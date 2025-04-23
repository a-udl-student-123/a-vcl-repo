# vcl core alg

import torch
import time
from tqdm import tqdm

from z_utils.training_utils import train_model_for_task, setup_directories, save_checkpoint, create_plots
from z_utils.utils import clean_memory, evaluate_all_tasks, evaluate_model, print_eval_results, clean_loader
from z_core.coreset import (
    select_coreset, create_coreset_loader, update_coreset, 
    initialize_coreset, create_filtered_dataloader
)

def initialize_model_for_vcl(model_class, train_loader, test_loader, device, 
                           use_ml_init, ml_epochs, lr, init_std=0.001, 
                           adaptive_std=False, adaptive_std_epsilon=0.01, 
                           exp_dir=None, initialize_model_fn=None,
                           different_perm_init=False):
    from z_utils.training_utils import initialize_model
    from z_models.ml_initialization import get_ml_initialized_vcl_model
    
    if use_ml_init and different_perm_init:
        model = model_class().to(device)
        return get_ml_initialized_vcl_model(
            train_loader=train_loader, vcl_model=model, test_loader=test_loader,
            ml_epochs=ml_epochs, lr=lr, init_std=init_std, adaptive_std=adaptive_std, 
            adaptive_std_epsilon=adaptive_std_epsilon, device=device,
            exp_dir=exp_dir, different_perm=True
        )
    
    init_fn = initialize_model_fn or initialize_model
    return init_fn(
        model_class, train_loader, test_loader, device, 
        use_ml_init, ml_epochs, lr, init_std, 
        adaptive_std, adaptive_std_epsilon, exp_dir
    )

def train_propagation_model(model, loader, task_idx, epochs, lr, device, 
                          record_metric_fn=None, exp_dir=None, n_train_samples=5):
    model, _ = train_model_for_task(
        model=model, loader=loader, epochs=epochs, lr=lr, device=device,
        record_metric_fn=record_metric_fn, t_idx=task_idx, 
        model_type='Propagation', exp_dir=exp_dir, n_train_samples=n_train_samples
    )
    model.store_params_as_old()
    return model

def train_prediction_model(model_class, propagation_model, coreset_ds, task_idx, 
                         epochs, lr, device, num_workers=0, pre_task_hook=None,
                         record_metric_fn=None, exp_dir=None, n_train_samples=5,
                         task_specific=False, task_seen_idx=None):
    pred_model = model_class().to(device)
    pred_model.load_state_dict(propagation_model.state_dict())
    
    if pre_task_hook:
        idx = task_seen_idx if task_specific else task_idx
        pred_model = pre_task_hook(pred_model, idx, is_prediction_model=True)
    
    model_type = f'Prediction_Task{task_seen_idx+1}' if task_specific else 'Prediction'
    task_filter = task_seen_idx if task_specific else None
    coreset_loader = create_coreset_loader(coreset_ds, num_workers=num_workers, task_idx=task_filter)
    
    if coreset_loader is None:
        if task_specific:
            print(f"Warning: No coreset data found for task {task_seen_idx+1}")
        return pred_model
    
    pred_model, _ = train_model_for_task(
        model=pred_model, loader=coreset_loader, epochs=epochs, lr=lr, 
        device=device, record_metric_fn=record_metric_fn, t_idx=task_idx,
        model_type=model_type, exp_dir=exp_dir, n_train_samples=n_train_samples
    )
    
    clean_loader(coreset_loader)
    return pred_model

def process_coreset(train_loader, coreset_size, use_kcenter, device, kcenter_batch_size, 
                  task_idx, coreset_ds, record_metric_fn=None, num_workers=0):
    coreset_x, coreset_y, coreset_indices = select_coreset(
        train_loader, coreset_size, use_kcenter, device, kcenter_batch_size
    )
    
    coreset_ds = update_coreset(coreset_ds, coreset_x, coreset_y, device, task_idx)
    
    if record_metric_fn:
        record_metric_fn(task_idx, -1, 'coreset_size', len(coreset_ds))
        record_metric_fn(task_idx, -1, 'coreset_method', 'kcenter' if use_kcenter else 'random')
    
    filtered_loader = create_filtered_dataloader(
        train_loader, coreset_indices, num_workers=num_workers
    )
    print(f"\nOriginal dataset size: {len(train_loader.dataset)}, "
          f"Filtered dataset size: {len(filtered_loader.dataset)}, "
          f"Moved {len(coreset_indices)} examples to Coreset")
    
    return coreset_ds, filtered_loader

def create_task_specific_models(model_class, propagation_model, coreset_ds, task_idx,
                              epochs, lr, device, num_workers, pre_task_hook,
                              record_metric_fn, exp_dir, n_train_samples):
    # TODO
    prediction_models = {}
    
    for task_seen_idx in range(task_idx + 1):
        pred_model = train_prediction_model(
            model_class, propagation_model, coreset_ds, task_idx, 
            epochs, lr, device, num_workers, pre_task_hook,
            record_metric_fn, exp_dir, n_train_samples, 
            task_specific=True, task_seen_idx=task_seen_idx
        )
        prediction_models[task_seen_idx] = pred_model
    
    active_model = prediction_models.get(task_idx, propagation_model)
    return propagation_model, active_model, prediction_models

def handle_coreset_training(model_class, propagation_model, train_loader, 
                          coreset_size, use_kcenter, task_idx, epochs, 
                          pred_epochs_multiplier, lr, device, num_workers=0,
                          kcenter_batch_size=1024, pre_task_hook=None,
                          record_metric_fn=None, exp_dir=None, 
                          n_train_samples=5, coreset_ds=None,
                          use_task_specific_prediction=False):
    coreset_ds, filtered_loader = process_coreset(
        train_loader, coreset_size, use_kcenter, device, kcenter_batch_size,
        task_idx, coreset_ds, record_metric_fn, num_workers
    )
    
    propagation_model = train_propagation_model(
        propagation_model, filtered_loader, task_idx, epochs, lr, 
        device, record_metric_fn, exp_dir, n_train_samples
    )
    
    clean_loader(filtered_loader)
    clean_memory(device)
    
    pred_epochs = int(epochs * pred_epochs_multiplier)
    
    if use_task_specific_prediction:
        return create_task_specific_models(
            model_class, propagation_model, coreset_ds, task_idx,
            pred_epochs, lr, device, num_workers, pre_task_hook,
            record_metric_fn, exp_dir, n_train_samples
        )
    
    prediction_model = train_prediction_model(
        model_class, propagation_model, coreset_ds, task_idx, 
        pred_epochs, lr, device, num_workers, pre_task_hook,
        record_metric_fn, exp_dir, n_train_samples
    )
    return propagation_model, prediction_model, None

def evaluate_models(models, test_loaders, task_idx, device, n_samples, 
                   record_metric_fn=None, evaluate_fn=None, 
                   use_task_specific=False):
    print(f"\nEvaluation after Task {task_idx}")
    
    if use_task_specific and isinstance(models, dict):
        accuracies, ce_losses, avg_accuracy, avg_ce_loss = eval_task_specific_models(
            models, test_loaders, task_idx, device, n_samples, record_metric_fn
        )
    else:
        eval_fn = evaluate_fn or evaluate_all_tasks
        accuracies, ce_losses, avg_accuracy, avg_ce_loss = eval_fn(
            models, test_loaders, task_idx, device, n_samples, record_metric_fn
        )
    
    avg_accuracy, avg_ce_loss = print_eval_results(task_idx, accuracies, ce_losses)
    
    if record_metric_fn:
        record_metric_fn(task_idx - 1, -1, 'average_accuracy', avg_accuracy)
        record_metric_fn(task_idx - 1, -1, 'average_ce_loss', avg_ce_loss)
        record_metric_fn(task_idx - 1, -1, 'num_tasks', len(accuracies))
        
    return accuracies, avg_accuracy

def eval_task_specific_models(prediction_models, test_loaders, num_tasks_seen, 
                            device, n_samples=100, record_metric_fn=None):
    accys = []
    ce_losses = []
    
    eval_pbar = tqdm(range(num_tasks_seen), desc="Task Eval", leave=False)
    
    for task_idx in eval_pbar:
        if task_idx not in prediction_models:
            print(f"Warning: No task-specific model found for task {task_idx+1}")
            accys.append(0.0)
            ce_losses.append(float('inf'))
            continue
            
        model = prediction_models[task_idx]
        
        if hasattr(model, 'set_current_task'):
            model.set_current_task(task_idx)
            
        accuracy, ce_loss = evaluate_model(
            model, test_loaders[task_idx], device, n_samples
        )
        
        if record_metric_fn:
            t_idx = num_tasks_seen - 1
            record_metric_fn(t_idx, -1, f'accuracy_on_task_{task_idx+1}', accuracy)
            record_metric_fn(t_idx, -1, f'ce_loss_on_task_{task_idx+1}', ce_loss)
        
        accys.append(accuracy)
        ce_losses.append(ce_loss)
        eval_pbar.set_postfix(accuracy=f"{accuracy:.4f}", task=f"{task_idx+1}")
        
    clean_memory(device)
    
    avg_accuracy = sum(accys) / len(accys) if accys else 0
    avg_ce_loss = sum(ce_losses) / len(ce_losses) if ce_losses else 0
    
    return accys, ce_losses, avg_accuracy, avg_ce_loss

def initialize_vcl_setup(model_class, train_loader_factories, test_loader_factories, 
                       device, use_ml_initialization, ml_epochs, lr, init_std, 
                       adaptive_std, adaptive_std_epsilon, exp_dir, 
                       initialize_model_fn, different_perm_init, 
                       coreset_size, record_metric_fn):
    exp_dir, checkpoint_dir = setup_directories(exp_dir)
    
    using_coreset = coreset_size > 0
    coreset_ds = initialize_coreset() if using_coreset else None
    
    if record_metric_fn:
        record_metric_fn(0, -1, 'init_std', init_std)
        record_metric_fn(0, -1, 'adaptive_std', 1 if adaptive_std else 0)
        record_metric_fn(0, -1, 'adaptive_std_epsilon', adaptive_std_epsilon)
        if different_perm_init:
            record_metric_fn(0, -1, 'different_perm_init', 1)
    
    first_train_loader = train_loader_factories[0]()
    ml_test_loader = test_loader_factories[0](force_persistent=True)
    
    propagation_model = initialize_model_for_vcl(
        model_class, first_train_loader, ml_test_loader, device,
        use_ml_initialization, ml_epochs, lr, init_std, 
        adaptive_std, adaptive_std_epsilon, exp_dir, 
        initialize_model_fn, different_perm_init
    )
    
    clean_loader(ml_test_loader)
    first_test_loader = test_loader_factories[0]()
    
    test_loaders = [first_test_loader] + [None] * (len(test_loader_factories) - 1)
    
    return (propagation_model, first_train_loader, test_loaders, 
           using_coreset, coreset_ds, exp_dir, checkpoint_dir)

def process_task(task_idx, model_class, train_loader_factories, first_train_loader, 
               propagation_model, coreset_size, use_kcenter, epochs_per_task, 
               pred_epochs_multiplier, lr, device, num_workers, kcenter_batch_size, 
               pre_task_hook, record_metric_fn, exp_dir, n_train_samples, 
               coreset_ds, use_task_specific_prediction, prediction_models):
    t_idx = task_idx - 1
    print(f"\nTask {task_idx}")
    
    if pre_task_hook:
        propagation_model = pre_task_hook(propagation_model, t_idx, is_prediction_model=False)
    
    current_train_loader = first_train_loader if t_idx == 0 else train_loader_factories[t_idx]()
    
    active_model = None

    if coreset_size > 0:
        propagation_model, active_model, task_models = handle_coreset_training(
            model_class, propagation_model, current_train_loader, 
            coreset_size, use_kcenter, t_idx, epochs_per_task, 
            pred_epochs_multiplier, lr, device, num_workers,
            kcenter_batch_size, pre_task_hook, record_metric_fn, 
            exp_dir, n_train_samples, coreset_ds, use_task_specific_prediction
        )
        
        if use_task_specific_prediction and task_models:
            prediction_models.update(task_models)
    else:
        propagation_model = train_propagation_model(
            propagation_model, current_train_loader, t_idx, 
            epochs_per_task, lr, device, record_metric_fn,
            exp_dir, n_train_samples
        )
        propagation_model._print_std_summary()
        active_model = propagation_model
    
    clean_loader(current_train_loader)
    clean_memory(device)
    
    return propagation_model, active_model, prediction_models

def load_test_loaders(test_loaders, test_loader_factories, task_idx):
    for i in range(task_idx):
        if test_loaders[i] is None:
            test_loaders[i] = test_loader_factories[i]()
    return test_loaders

def train_vcl(
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
    initialize_model_fn=None,
    evaluate_all_tasks_fn=None,
    pre_task_hook=None,
    use_task_specific_prediction=False,
    different_perm_init=False,
    early_stopping_threshold=None
):
    """Main training function for VCL models.
    
    Trains model sequentially on tasks, using coreset if specified. Evaluates after each task.
    
    Args:
        model_class: Model class to use
        train_loader_factories: List of functions that return train dataloaders
        test_loader_factories: List of functions that return test dataloaders
        epochs_per_task: Number of epochs per task
        coreset_size: Size of coreset (0 = no coreset)
        use_kcenter: Whether to use k-center coreset selection
        kcenter_batch_size: Batch size for k-center
        lr: Learning rate
        device: Device to use
        num_workers: Number of dataloader workers
        record_metric_fn: Optional metric recording function
        exp_dir: Directory for saving results
        use_ml_initialization: Whether to use ML initialization
        ml_epochs: Epochs for ML init
        n_eval_samples: Samples for evaluation
        n_train_samples: Samples during training
        pred_epochs_multiplier: Multiplier for prediction model epochs
        init_std: Initial weight std
        adaptive_std: Use adaptive std
        adaptive_std_epsilon: Epsilon for adaptive std
        initialize_model_fn: Custom init function
        evaluate_all_tasks_fn: Custom eval function
        pre_task_hook: Hook called before each task
        use_task_specific_prediction: Use task-specific models
        different_perm_init: Use different permutations for init
        early_stopping_threshold: Early stopping threshold
    """
    start_time = time.time()
    
    (propagation_model, first_train_loader, test_loaders, 
     using_coreset, coreset_ds, exp_dir, checkpoint_dir) = initialize_vcl_setup(
        model_class, train_loader_factories, test_loader_factories, 
        device, use_ml_initialization, ml_epochs, lr, init_std, 
        adaptive_std, adaptive_std_epsilon, exp_dir, 
        initialize_model_fn, different_perm_init, 
        coreset_size, record_metric_fn
    )

    task_accs = []
    avg_accs = []
    num_tasks = len(train_loader_factories)
    prediction_models = {}
    
    task_pbar = tqdm(range(1, num_tasks+1), desc="Tasks", leave=False)
    for task_idx in task_pbar:
        task_pbar.set_description(f"Task {task_idx}")
        
        propagation_model, active_model, prediction_models = process_task(
            task_idx, model_class, train_loader_factories, first_train_loader, 
            propagation_model, coreset_size, use_kcenter, epochs_per_task, 
            pred_epochs_multiplier, lr, device, num_workers, kcenter_batch_size, 
            pre_task_hook, record_metric_fn, exp_dir, n_train_samples, 
            coreset_ds, use_task_specific_prediction, prediction_models
        )
        
        save_checkpoint(active_model, checkpoint_dir, task_idx)
        
        if task_idx == 1:
            first_train_loader = None
        
        test_loaders = load_test_loaders(test_loaders, test_loader_factories, task_idx)
        
        models_to_eval = prediction_models if (use_task_specific_prediction and prediction_models) else active_model
            
        accuracies, avg_accuracy = evaluate_models(
            models_to_eval, test_loaders[:task_idx], task_idx, device, n_eval_samples,
            record_metric_fn, evaluate_all_tasks_fn, use_task_specific_prediction
        )
        
        task_accs.append(accuracies)
        avg_accs.append(avg_accuracy)
        task_pbar.set_postfix(avg_acc=f"{avg_accuracy:.4f}")
        
        print(f"Finished Task {task_idx}")
        
        if early_stopping_threshold is not None and avg_accuracy < early_stopping_threshold:
            print(f"\nWARNING: Average accuracy ({avg_accuracy:.4f}) fell below threshold ({early_stopping_threshold:.4f})")
            print("Terminating training early to save computation resources.")
            
            for i, loader in enumerate(test_loaders):
                if loader is not None:
                    clean_loader(loader)
                    test_loaders[i] = None
            
            try:
                import wandb
                if wandb.run is not None:
                    wandb.run.summary["early_stopped"] = True
                    wandb.run.summary["early_stopping_reason"] = f"Accuracy {avg_accuracy:.4f} below threshold {early_stopping_threshold:.4f}"
                    wandb.run.summary["early_stopping_task"] = task_idx
            except:
                pass
                
            total_time = time.time() - start_time
            print(f"\nTotal training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            
            if record_metric_fn:
                record_metric_fn(task_idx-1, -1, 'early_stopped', 1)
                record_metric_fn(task_idx-1, -1, 'early_stopping_task', task_idx)
                record_metric_fn(task_idx-1, -1, 'total_duration_seconds', total_time)
            
            create_plots(exp_dir, avg_accs, task_accs)
            
            return prediction_models if use_task_specific_prediction and prediction_models else active_model
    
    for i, loader in enumerate(test_loaders):
        if loader is not None:
            clean_loader(loader)
            test_loaders[i] = None
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    if record_metric_fn:
        record_metric_fn(num_tasks-1, -1, 'total_duration_seconds', total_time)
    
    create_plots(exp_dir, avg_accs, task_accs)
    
    return prediction_models if use_task_specific_prediction and prediction_models else active_model
