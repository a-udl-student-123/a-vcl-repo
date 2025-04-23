# vcl implementation with gaussian likelihood for regression tasks
# handles both standard and coreset vcl

import torch
import time
from tqdm import tqdm
from pathlib import Path

from z_gaussian.utils_gaussian import (
    convert_to_onehot, gaussian_nll_loss, compute_rmse, compute_gradient_norm, 
    clean_memory, evaluate_model_gaussian, evaluate_all_tasks_gaussian, 
    print_eval_results_gaussian, compute_vcl_model_stats, compute_accuracy
)
from z_utils.training_utils import setup_directories, save_checkpoint, create_plots
from z_utils.utils import clean_loader
from z_core.coreset import (
    select_coreset, create_coreset_loader, update_coreset, 
    initialize_coreset, create_filtered_dataloader
)

def initialize_model_for_gaussian_vcl(model_class, train_loader, test_loader, device, 
                                   use_ml_initialization, ml_epochs, lr, init_std=0.001, 
                                   adaptive_std=False, adaptive_std_epsilon=0.01, 
                                   exp_dir=None, different_perm_init=False):
    from z_gaussian.ml_initialization_gaussian import get_ml_initialized_gaussian_vcl_model
    
    model = model_class().to(device)
    
    if use_ml_initialization:
        model = get_ml_initialized_gaussian_vcl_model(
            train_loader=train_loader,
            vcl_model=model,
            test_loader=test_loader,
            ml_epochs=ml_epochs,
            lr=lr,
            init_std=init_std,
            adaptive_std=adaptive_std,
            adaptive_std_epsilon=adaptive_std_epsilon,
            device=device,
            exp_dir=exp_dir,
            different_perm=different_perm_init
        )
    else:
        # Standard initialization
        model.set_init_std(init_std, adaptive_std, adaptive_std_epsilon)
        std_type = "adaptive" if adaptive_std else "fixed"
        print(f"\nUsing {std_type} std: {init_std}")
    
    return model

def train_batch_gaussian(model, x_batch, y_batch, optimizer, dataset_size, n_train_samples, compute_grad_norm=False):
    batch_size = x_batch.size(0)
    
    if y_batch.dim() == 1:
        y_batch = convert_to_onehot(y_batch, num_classes=model.mean_head.out_features)
    
    optimizer.zero_grad()
    mean, logvar, aleatoric_uncertainty, epistemic_uncertainty = model(x_batch, n_samples=n_train_samples)
    
    nll_loss = gaussian_nll_loss(mean, logvar, y_batch)
    kl = model.kl_loss()
    
    avg_variance = aleatoric_uncertainty.mean().item() + epistemic_uncertainty.mean().item()
    
    # scale kl by dataset size for stability
    loss = nll_loss + kl / dataset_size
    loss.backward()
    
    # clip grads to avoid explosions, high clipping as grads are naturally large
    max_grad_norm = 1000.0
    
    rmse = compute_rmse(mean, y_batch)
    accuracy = compute_accuracy(mean, y_batch)
    
    grad_norm = compute_gradient_norm(model) if compute_grad_norm else 0.0
    
    optimizer.step()
    
    return {
        'loss': loss.item() * batch_size,
        'kl_scaled': (kl.item() / dataset_size) * batch_size,
        'nll_loss': nll_loss.item() * batch_size,
        'rmse': rmse,
        'accuracy': accuracy,
        'grad_norm': grad_norm,
        'variance': avg_variance
    }

def train_epoch_gaussian(model, loader, optimizer, device, dataset_size, n_train_samples=5):
    model = model.to(device)
    model.train()
    
    total_loss = 0.0
    total_kl_scaled = 0.0
    total_nll_loss = 0.0
    total_rmse = 0.0
    total_accuracy = 0.0
    total_variance = 0.0
    total = 0
    grad_norms = []
    
    #  10% of batches for grad norm tracking
    total_batches = len(loader)
    sampling_interval = max(1, int(total_batches / 10))
    
    batch_pbar = tqdm(enumerate(loader), total=total_batches, desc="Batches", leave=False)
    for batch_idx, (x_batch, y_batch) in batch_pbar:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        batch_size = x_batch.size(0)

        compute_grad = (batch_idx % sampling_interval == 0)
        
        batch_metrics = train_batch_gaussian(
            model, x_batch, y_batch, optimizer, dataset_size, 
            n_train_samples, compute_grad_norm=compute_grad
        )
        
        total_loss += batch_metrics['loss']
        total_kl_scaled += batch_metrics['kl_scaled']
        total_nll_loss += batch_metrics['nll_loss']
        total_rmse += batch_metrics['rmse'] * batch_size
        total_accuracy += batch_metrics['accuracy'] * batch_size
        total_variance += batch_metrics['variance'] * batch_size
        total += batch_size
        
        if compute_grad:
            grad_norms.append(batch_metrics['grad_norm'])
            batch_pbar.set_postfix(
                loss=f"{batch_metrics['loss']/batch_size:.4f}",
                rmse=f"{batch_metrics['rmse']:.4f}",
                var=f"{batch_metrics['variance']:.4f}",
                acc=f"{batch_metrics['accuracy']:.4f}",
                grad=f"{batch_metrics['grad_norm']:.4f}"
            )
    
    # calc averages
    avg_loss = total_loss / total if total > 0 else 0
    avg_nll_loss = total_nll_loss / total if total > 0 else 0
    avg_kl_loss = total_kl_scaled / total if total > 0 else 0
    avg_rmse = total_rmse / total if total > 0 else 0
    avg_accuracy = total_accuracy / total if total > 0 else 0
    avg_variance = total_variance / total if total > 0 else 0
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
    
    return {
        'avg_loss': avg_loss,
        'avg_nll_loss': avg_nll_loss,
        'avg_kl_loss': avg_kl_loss,
        'avg_rmse': avg_rmse,
        'avg_accuracy': avg_accuracy,
        'avg_variance': avg_variance,
        'total_nll_loss': total_nll_loss,
        'avg_grad_norm': avg_grad_norm
    }

def record_epoch_metrics_gaussian(record_fn, t_idx, epoch, metrics, layer_stats, model_type):
    if not record_fn:
        return
        
    prefix = f'train_{model_type.lower()}' if model_type else 'train'
    
    record_fn(t_idx, epoch, f'{prefix}_loss', metrics['avg_loss'])
    record_fn(t_idx, epoch, f'{prefix}_nll_loss', metrics['avg_nll_loss'])
    record_fn(t_idx, epoch, f'{prefix}_kl_loss', metrics['avg_kl_loss'])
    record_fn(t_idx, epoch, f'{prefix}_rmse', metrics['avg_rmse'])
    record_fn(t_idx, epoch, f'{prefix}_accuracy', metrics['avg_accuracy'])
    record_fn(t_idx, epoch, f'{prefix}_variance', metrics['avg_variance'])
    record_fn(t_idx, epoch, f'{prefix}_total_nll_loss', metrics['total_nll_loss'])
    
    if 'avg_grad_norm' in metrics:
        record_fn(t_idx, epoch, f'{prefix}_grad_norm', metrics['avg_grad_norm'])
    
    record_fn(t_idx, epoch, f'{prefix}_weight_std', layer_stats['avg_weight_std'])
    record_fn(t_idx, epoch, f'{prefix}_bias_std', layer_stats['avg_bias_std'])
    record_fn(t_idx, epoch, f'{prefix}_weight_mu_abs', layer_stats['avg_weight_mu_abs'])
    record_fn(t_idx, epoch, f'{prefix}_bias_mu_abs', layer_stats['avg_bias_mu_abs'])

def train_gaussian_model_for_task(model, loader, epochs, lr, device, record_metric_fn=None, 
                               t_idx=0, model_type=None, exp_dir=None, n_train_samples=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset_size = len(loader.dataset)
    task_idx = t_idx + 1
    
    epoch_metrics = {
        'loss': [], 'nll_loss': [], 'kl': [], 'rmse': [], 'accuracy': [], 'variance': [], 'grad_norm': [],
        'weight_std': [], 'bias_std': [], 'weight_mu_abs': [], 'bias_mu_abs': []
    }
    
    model_type_str = f"{model_type} " if model_type else ""
    print(f"\nTraining {model_type_str}model on Task {task_idx}...")
    
    epoch_pbar = tqdm(range(1, epochs + 1), desc=f"{model_type_str}Epochs", leave=False)
    
    layer_stats = compute_vcl_model_stats(model)
    
    for epoch in epoch_pbar:
        epoch_metrics['weight_std'].append(layer_stats['avg_weight_std'])
        epoch_metrics['bias_std'].append(layer_stats['avg_bias_std'])
        epoch_metrics['weight_mu_abs'].append(layer_stats['avg_weight_mu_abs'])
        epoch_metrics['bias_mu_abs'].append(layer_stats['avg_bias_mu_abs'])
        
        metrics = train_epoch_gaussian(model, loader, optimizer, device, dataset_size, n_train_samples)
        
        epoch_metrics['loss'].append(metrics['avg_loss'])
        epoch_metrics['nll_loss'].append(metrics['avg_nll_loss'])
        epoch_metrics['kl'].append(metrics['avg_kl_loss'])
        epoch_metrics['rmse'].append(metrics['avg_rmse'])
        epoch_metrics['accuracy'].append(metrics['avg_accuracy'])
        epoch_metrics['variance'].append(metrics['avg_variance'])
        epoch_metrics['grad_norm'].append(metrics.get('avg_grad_norm', 0))
        
        epoch_pbar.set_postfix(
            RMSE=f"{metrics['avg_rmse']:.4f}", 
            NLL=f"{metrics['avg_nll_loss']:.4f}", 
            KL=f"{metrics['avg_kl_loss']:.4f}",
            Var=f"{metrics['avg_variance']:.4f}",
            Loss=f"{metrics['avg_loss']:.4f}",
            Grad=f"{metrics.get('avg_grad_norm', 0):.4f}"
        )
        
        record_epoch_metrics_gaussian(record_metric_fn, t_idx, epoch, metrics, layer_stats, model_type)
        
        layer_stats = compute_vcl_model_stats(model)
    
    clean_memory(device)
    
    print(f"\nFinished training {model_type_str}model on Task {task_idx}")
    
    # TODO: implement rmse plots
    
    model_type_prefix = f"{model_type} Model" if model_type else "Model"
    print(f"    Training Results for {model_type_prefix} after Task {task_idx}:")
    print(f"    Task {task_idx}: RMSE={metrics['avg_rmse']:.4f} | NLL={metrics['avg_nll_loss']:.4f} | " +
          f"KL={metrics['avg_kl_loss']:.4f} | Var={metrics['avg_variance']:.4f} | Loss={metrics['avg_loss']:.4f}")
    
    return model, metrics

def train_propagation_model_gaussian(model, loader, task_idx, epochs, lr, device, 
                                  record_metric_fn=None, exp_dir=None, n_train_samples=5):
    model, _ = train_gaussian_model_for_task(
        model=model, loader=loader, epochs=epochs, lr=lr, device=device,
        record_metric_fn=record_metric_fn, t_idx=task_idx, 
        model_type='Propagation', exp_dir=exp_dir, n_train_samples=n_train_samples
    )
    model.store_params_as_old()
    return model

def train_prediction_model_gaussian(model_class, propagation_model, coreset_ds, task_idx, 
                                 epochs, lr, device, num_workers=0,
                                 record_metric_fn=None, exp_dir=None, n_train_samples=5):
    pred_model = model_class().to(device)
    pred_model.load_state_dict(propagation_model.state_dict())
    
    coreset_loader = create_coreset_loader(coreset_ds, num_workers=num_workers)
    
    if coreset_loader is None:
        return pred_model
    
    pred_model, _ = train_gaussian_model_for_task(
        model=pred_model, loader=coreset_loader, epochs=epochs, lr=lr, 
        device=device, record_metric_fn=record_metric_fn, t_idx=task_idx,
        model_type='Prediction', exp_dir=exp_dir, n_train_samples=n_train_samples
    )
    
    clean_loader(coreset_loader)
    return pred_model

def handle_coreset_training_gaussian(model_class, propagation_model, train_loader, 
                                  coreset_size, use_kcenter, task_idx, epochs, 
                                  pred_epochs_multiplier, lr, device, num_workers=0,
                                  kcenter_batch_size=1024, record_metric_fn=None, 
                                  exp_dir=None, n_train_samples=5, coreset_ds=None):
    coreset_x, coreset_y, coreset_indices = select_coreset(
        train_loader, coreset_size, use_kcenter, device, kcenter_batch_size
    )
    
    coreset_ds = update_coreset(coreset_ds, coreset_x, coreset_y, device, task_idx)

    filtered_loader = create_filtered_dataloader(
        train_loader, coreset_indices, num_workers=num_workers
    )
    print(f"\nOriginal dataset size: {len(train_loader.dataset)}, "
          f"Filtered dataset size: {len(filtered_loader.dataset)}, "
          f"Moved {len(coreset_indices)} examples to Coreset")
    
    propagation_model = train_propagation_model_gaussian(
        propagation_model, filtered_loader, task_idx, epochs, lr, 
        device, record_metric_fn, exp_dir, n_train_samples
    )
    
    clean_loader(filtered_loader)
    clean_memory(device)
    
    pred_epochs = int(epochs * pred_epochs_multiplier)
    
    prediction_model = train_prediction_model_gaussian(
        model_class, propagation_model, coreset_ds, task_idx, 
        pred_epochs, lr, device, num_workers,
        record_metric_fn, exp_dir, n_train_samples
    )
    return propagation_model, prediction_model, coreset_ds

def initialize_vcl_gaussian_setup(model_class, train_loader_factories, test_loader_factories, 
                              device, use_ml_initialization, ml_epochs, lr, init_std, 
                              adaptive_std, adaptive_std_epsilon, exp_dir, 
                              different_perm_init, coreset_size, record_metric_fn):
    exp_dir, checkpoint_dir = setup_directories(exp_dir)
    
    using_coreset = coreset_size > 0
    coreset_ds = initialize_coreset() if using_coreset else None
    
    first_train_loader = train_loader_factories[0]()
    ml_test_loader = test_loader_factories[0](force_persistent=True)
    
    propagation_model = initialize_model_for_gaussian_vcl(
        model_class, first_train_loader, ml_test_loader, device,
        use_ml_initialization, ml_epochs, lr, init_std, 
        adaptive_std, adaptive_std_epsilon, exp_dir, 
        different_perm_init
    )
    
    clean_loader(ml_test_loader)
    first_test_loader = test_loader_factories[0]()
    
    test_loaders = [first_test_loader] + [None] * (len(test_loader_factories) - 1)
    
    return (propagation_model, first_train_loader, test_loaders, 
           using_coreset, coreset_ds, exp_dir, checkpoint_dir)

def process_task_gaussian(task_idx, model_class, train_loader_factories, first_train_loader, 
                       propagation_model, coreset_size, use_kcenter, epochs_per_task, 
                       pred_epochs_multiplier, lr, device, num_workers, kcenter_batch_size, 
                       record_metric_fn, exp_dir, n_train_samples, coreset_ds):
    t_idx = task_idx - 1
    print(f"\nStarting Task {task_idx}")
    
    current_train_loader = first_train_loader if t_idx == 0 else train_loader_factories[t_idx]()
    
    active_model = None
    
    if coreset_size > 0:
        propagation_model, active_model, coreset_ds = handle_coreset_training_gaussian(
            model_class, propagation_model, current_train_loader, 
            coreset_size, use_kcenter, t_idx, epochs_per_task, 
            pred_epochs_multiplier, lr, device, num_workers,
            kcenter_batch_size, record_metric_fn, 
            exp_dir, n_train_samples, coreset_ds
        )
    else:
        propagation_model = train_propagation_model_gaussian(
            propagation_model, current_train_loader, t_idx, 
            epochs_per_task, lr, device, record_metric_fn,
            exp_dir, n_train_samples
        )
        active_model = propagation_model
    
    clean_loader(current_train_loader)
    clean_memory(device)
    
    return propagation_model, active_model, coreset_ds

def load_test_loaders(test_loaders, test_loader_factories, task_idx):
    for i in range(task_idx):
        if test_loaders[i] is None:
            test_loaders[i] = test_loader_factories[i]()
    return test_loaders

def train_vcl_gaussian(
    model_class,
    train_loader_factories,
    test_loader_factories,
    epochs_per_task=5,
    coreset_size=0,
    use_kcenter=False,
    kcenter_batch_size=1024,
    lr=1e-3,
    device='cuda',
    num_workers=0,
    record_metric_fn=None,
    exp_dir=None,
    use_ml_initialization=False,
    ml_epochs=5,
    n_eval_samples=100,
    n_train_samples=5,
    pred_epochs_multiplier=1.0, 
    init_std=0.001, 
    adaptive_std=False,
    adaptive_std_epsilon=0.01,
    different_perm_init=False,
    early_stopping_threshold=None
):
    start_time = time.time()
    
    (propagation_model, first_train_loader, test_loaders, 
     using_coreset, coreset_ds, exp_dir, checkpoint_dir) = initialize_vcl_gaussian_setup(
        model_class, train_loader_factories, test_loader_factories, 
        device, use_ml_initialization, ml_epochs, lr, init_std, 
        adaptive_std, adaptive_std_epsilon, exp_dir, 
        different_perm_init, coreset_size, record_metric_fn
    )
    
    task_rmse_values = []
    avg_rmse_values = []
    num_tasks = len(train_loader_factories)
    
    task_pbar = tqdm(range(1, num_tasks+1), desc="Tasks", leave=False)
    for task_idx in task_pbar:
        task_pbar.set_description(f"Task {task_idx}")
        
        propagation_model, active_model, coreset_ds = process_task_gaussian(
            task_idx, model_class, train_loader_factories, first_train_loader,
            propagation_model, coreset_size, use_kcenter, epochs_per_task,
            pred_epochs_multiplier, lr, device, num_workers, kcenter_batch_size, 
            record_metric_fn, exp_dir, n_train_samples, coreset_ds
        )
        
        save_checkpoint(active_model, checkpoint_dir, task_idx)
        
        train_metrics = getattr(active_model, 'last_metrics', {})
        
        if task_idx == 1:
            first_train_loader = None
        
        test_loaders = load_test_loaders(test_loaders, test_loader_factories, task_idx)
        
        eval_metrics = evaluate_all_tasks_gaussian(
            active_model, test_loaders[:task_idx], task_idx, device, n_eval_samples,
            record_metric_fn
        )
        
        task_rmse_values.append(eval_metrics['rmse_values'])
        avg_rmse_values.append(eval_metrics['avg_rmse'])
        task_pbar.set_postfix(
            avg_rmse=f"{eval_metrics['avg_rmse']:.4f}", 
            avg_acc=f"{eval_metrics['avg_accuracy']:.4f}", 
            avg_nll=f"{eval_metrics['avg_nll']:.4f}"
        )
        
        print_eval_results_gaussian(task_idx, eval_metrics)
        
        print(f"Finished Task {task_idx}")
        
        if record_metric_fn:
            record_metric_fn(task_idx-1, -1, 'average_rmse', eval_metrics['avg_rmse'])
            record_metric_fn(task_idx-1, -1, 'average_accuracy', eval_metrics['avg_accuracy'])
            record_metric_fn(task_idx-1, -1, 'average_nll', eval_metrics['avg_nll'])
            record_metric_fn(task_idx-1, -1, 'average_loss', eval_metrics['avg_loss'])
            record_metric_fn(task_idx-1, -1, 'average_variance', eval_metrics['avg_variance'])
            record_metric_fn(task_idx-1, -1, 'average_kl', eval_metrics['avg_kl'])
            
        if early_stopping_threshold is not None and eval_metrics['avg_rmse'] > early_stopping_threshold:
            print(f"\nEarly stopping: RMSE {eval_metrics['avg_rmse']:.4f} > {early_stopping_threshold:.4f}")
            print("Stopping early to save compute")
            
            for i, loader in enumerate(test_loaders):
                if loader is not None:
                    clean_loader(loader)
                    test_loaders[i] = None
            
            try:
                import wandb
                if wandb.run is not None:
                    wandb.run.summary["early_stopped"] = True
                    wandb.run.summary["early_stopping_reason"] = f"RMSE {eval_metrics['avg_rmse']:.4f} above threshold {early_stopping_threshold:.4f}"
                    wandb.run.summary["early_stopping_task"] = task_idx
            except:
                pass
        
            total_time = time.time() - start_time
            print(f"\nTotal training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            
            if record_metric_fn:
                record_metric_fn(task_idx-1, -1, 'early_stopped', 1)
                record_metric_fn(task_idx-1, -1, 'early_stopping_task', task_idx)
                record_metric_fn(task_idx-1, -1, 'total_duration_seconds', total_time)
            
            return active_model
    
    for i, loader in enumerate(test_loaders):
        if loader is not None:
            clean_loader(loader)
            test_loaders[i] = None
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    if record_metric_fn:
        record_metric_fn(num_tasks-1, -1, 'total_duration_seconds', total_time)
    
    # TODO:  rmse plots
    
    return active_model 