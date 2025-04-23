# training utils for vcl models 

import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

from z_utils.plotting_utils import create_accuracy_plot, create_task_specific_plots, create_task_training_plots
from z_utils.utils import clean_memory

def compute_layer_stats(layer):
    # get stats for a single vcl layer
    stats = {}
    
    stats['weight_mu_abs'] = torch.abs(layer.weight_mu).sum().item()
    stats['weight_mu_count'] = layer.weight_mu.numel()
    stats['bias_mu_abs'] = torch.abs(layer.bias_mu).sum().item() 
    stats['bias_mu_count'] = layer.bias_mu.numel()
    
    weight_std = layer._get_sigma(layer.weight_rho)
    bias_std = layer._get_sigma(layer.bias_rho)
    stats['weight_std'] = weight_std.sum().item()
    stats['weight_std_count'] = weight_std.numel()
    stats['bias_std'] = bias_std.sum().item()
    stats['bias_std_count'] = bias_std.numel()
    
    return stats

def compute_vcl_model_stats(model):
    stats = {
        'weight_mu_abs': 0.0, 'weight_mu_count': 0,
        'bias_mu_abs': 0.0, 'bias_mu_count': 0,
        'weight_std_avg': 0.0, 'weight_std_count': 0,
        'bias_std_avg': 0.0, 'bias_std_count': 0
    }
    
    if hasattr(model, 'shared_layers'):
        # multi-head case
        for layer in model.shared_layers:
            layer_stats = compute_layer_stats(layer)
            
            for key in ['weight_mu_abs', 'weight_mu_count', 'bias_mu_abs', 'bias_mu_count']:
                stats[key] += layer_stats[key]
            
            stats['weight_std_avg'] += layer_stats['weight_std']
            stats['weight_std_count'] += layer_stats['weight_std_count']
            stats['bias_std_avg'] += layer_stats['bias_std']
            stats['bias_std_count'] += layer_stats['bias_std_count']
    else:
        # standard model
        for i in range(1, 3):
            layer = getattr(model, f"lin{i}")
            layer_stats = compute_layer_stats(layer)
            
            for key in ['weight_mu_abs', 'weight_mu_count', 'bias_mu_abs', 'bias_mu_count']:
                stats[key] += layer_stats[key]
            
            stats['weight_std_avg'] += layer_stats['weight_std']
            stats['weight_std_count'] += layer_stats['weight_std_count']
            stats['bias_std_avg'] += layer_stats['bias_std']
            stats['bias_std_count'] += layer_stats['bias_std_count']
    
    is_multi_head = hasattr(model, 'heads')
    output_layer = model.heads[model.current_task] if is_multi_head else model.lin3
    
    layer_stats = compute_layer_stats(output_layer)
    for key in ['weight_mu_abs', 'weight_mu_count', 'bias_mu_abs', 'bias_mu_count']:
        stats[key] += layer_stats[key]
    
    stats['weight_std_avg'] += layer_stats['weight_std']
    stats['weight_std_count'] += layer_stats['weight_std_count']
    stats['bias_std_avg'] += layer_stats['bias_std']
    stats['bias_std_count'] += layer_stats['bias_std_count']
    
    results = {
        'avg_weight_mu_abs': stats['weight_mu_abs'] / stats['weight_mu_count'] if stats['weight_mu_count'] > 0 else 0,
        'avg_bias_mu_abs': stats['bias_mu_abs'] / stats['bias_mu_count'] if stats['bias_mu_count'] > 0 else 0,
        'avg_weight_std': stats['weight_std_avg'] / stats['weight_std_count'] if stats['weight_std_count'] > 0 else 0,
        'avg_bias_std': stats['bias_std_avg'] / stats['bias_std_count'] if stats['bias_std_count'] > 0 else 0
    }
    
    return results

def compute_gradient_norm(model):
    total_norm = 0.0
    total_params = 0
    
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm(2).item() ** 2
            total_params += 1
            
    avg_norm = (total_norm / total_params) ** 0.5 if total_params > 0 else 0
    return avg_norm

def print_vcl_model_stats(model, task_idx, model_type=None):
    stats = compute_vcl_model_stats(model)
    model_type_str = f"{model_type} " if model_type else ""
    
    print(f"\n  {model_type_str}Model Statistics After Task {task_idx}")
    print(f"    avg: weight_mu={stats['avg_weight_mu_abs']:.6f}, bias_mu={stats['avg_bias_mu_abs']:.6f}, " +
          f"weight_std={stats['avg_weight_std']:.6f}, bias_std={stats['avg_bias_std']:.6f}")

def train_batch(model, x_batch, y_batch, optimizer, ce_loss_fn, dataset_size, n_train_samples, compute_grad_norm=False):
    batch_size = x_batch.size(0)
    
    optimizer.zero_grad()
    logits = model(x_batch, n_samples=n_train_samples)
    ce_loss = ce_loss_fn(logits, y_batch)
    kl = model.kl_loss()
    
    loss = ce_loss + kl / dataset_size
    loss.backward()
    
    grad_norm = compute_gradient_norm(model) if compute_grad_norm else 0.0
    
    optimizer.step()

    _, pred = torch.max(logits, dim=1)
    batch_correct = (pred == y_batch).sum().item()
    
    return {
        'loss': loss.item() * batch_size,
        'kl_scaled': (kl.item() / dataset_size) * batch_size,
        'ce_loss': ce_loss.item() * batch_size,
        'correct': batch_correct,
        'grad_norm': grad_norm
    }

def train_epoch(model, loader, optimizer, device, dataset_size, n_train_samples=5):
    model = model.to(device)
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    
    total_loss = 0.0
    total_kl_scaled = 0.0
    total_ce_loss = 0.0
    correct = 0
    total = 0
    grad_norms = []
    
    layer_stats = compute_vcl_model_stats(model)
    
    # sample grad norm every 10% of batches
    total_batches = len(loader)
    sampling_interval = max(1, int(total_batches / 10))
    
    batch_pbar = tqdm(enumerate(loader), total=total_batches, desc="Batches", leave=False)
    for batch_idx, (x_batch, y_batch) in batch_pbar:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        batch_size = x_batch.size(0)

        compute_grad = (batch_idx % sampling_interval == 0)
        
        batch_metrics = train_batch(model, x_batch, y_batch, optimizer, ce_loss_fn, 
                                   dataset_size, n_train_samples, compute_grad_norm=compute_grad)
        
        total_loss += batch_metrics['loss']
        total_kl_scaled += batch_metrics['kl_scaled']
        total_ce_loss += batch_metrics['ce_loss']
        correct += batch_metrics['correct']
        total += batch_size
        
        batch_acc = batch_metrics['correct'] / batch_size if batch_size > 0 else 0
        
        if compute_grad:
            grad_norms.append(batch_metrics['grad_norm'])
            batch_pbar.set_postfix(
                loss=f"{batch_metrics['loss']/batch_size:.4f}",
                acc=f"{batch_acc:.4f}",
                grad=f"{batch_metrics['grad_norm']:.4f}"
            )
    
    avg_loss = total_loss / total if total > 0 else 0
    avg_ce_loss = total_ce_loss / total if total > 0 else 0
    avg_kl_loss = total_kl_scaled / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
    
    return {
        'avg_loss': avg_loss,
        'avg_ce_loss': avg_ce_loss,
        'avg_kl_loss': avg_kl_loss,
        'accuracy': accuracy,
        'total_ce_loss': total_ce_loss,
        'avg_grad_norm': avg_grad_norm
    }

def record_epoch_metrics(record_fn, t_idx, epoch, metrics, layer_stats, model_type):
    if not record_fn:
        return
        
    prefix = f'train_{model_type.lower()}' if model_type else 'train'
    
    record_fn(t_idx, epoch, f'{prefix}_loss', metrics['avg_loss'])
    record_fn(t_idx, epoch, f'{prefix}_ce_loss', metrics['avg_ce_loss'])
    record_fn(t_idx, epoch, f'{prefix}_kl_loss', metrics['avg_kl_loss'])
    record_fn(t_idx, epoch, f'{prefix}_accuracy', metrics['accuracy'])
    record_fn(t_idx, epoch, f'{prefix}_total_ce_loss', metrics['total_ce_loss'])
    
    if 'avg_grad_norm' in metrics:
        record_fn(t_idx, epoch, f'{prefix}_grad_norm', metrics['avg_grad_norm'])
    
    record_fn(t_idx, epoch, f'{prefix}_weight_std', layer_stats['avg_weight_std'])
    record_fn(t_idx, epoch, f'{prefix}_bias_std', layer_stats['avg_bias_std'])
    record_fn(t_idx, epoch, f'{prefix}_weight_mu_abs', layer_stats['avg_weight_mu_abs'])
    record_fn(t_idx, epoch, f'{prefix}_bias_mu_abs', layer_stats['avg_bias_mu_abs'])

def get_model_description(model_type, task_idx):
    if model_type and model_type.startswith('Prediction_Task'):
        start_msg = f"\nTraining {model_type} model (after completing Task {task_idx})..."
        end_msg = f"\nFinished training {model_type} model (after completing Task {task_idx})"
    elif model_type and model_type == 'Prediction':
        start_msg = f"\nTraining Prediction model on complete coreset (after Task {task_idx})..."
        end_msg = f"\nFinished training Prediction model on complete coreset (after Task {task_idx})"
    else:
        model_type_str = f"{model_type} " if model_type else ""
        start_msg = f"\nTraining {model_type_str}model on Task {task_idx}..."
        end_msg = f"\nFinished training {model_type_str}model on Task {task_idx}"
    
    return start_msg, end_msg

def train_model_for_task(model, loader, epochs, lr, device, record_metric_fn=None, 
                        t_idx=0, model_type=None, exp_dir=None, n_train_samples=5):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset_size = len(loader.dataset)
    task_idx = t_idx + 1  # 1-based 
    
    epoch_metrics = {
        'loss': [], 'ce_loss': [], 'kl': [], 'accuracy': [], 'grad_norm': [],
        'weight_std': [], 'bias_std': [], 'weight_mu_abs': [], 'bias_mu_abs': []
    }
    
    start_msg, end_msg = get_model_description(model_type, task_idx)
    print(start_msg)
    
    model_type_str = f"{model_type} " if model_type else ""
    epoch_pbar = tqdm(range(1, epochs + 1), desc=f"{model_type_str}Epochs", leave=False)
    
    layer_stats = compute_vcl_model_stats(model)
    
    for epoch in epoch_pbar:
        epoch_metrics['weight_std'].append(layer_stats['avg_weight_std'])
        epoch_metrics['bias_std'].append(layer_stats['avg_bias_std'])
        epoch_metrics['weight_mu_abs'].append(layer_stats['avg_weight_mu_abs'])
        epoch_metrics['bias_mu_abs'].append(layer_stats['avg_bias_mu_abs'])
        
        metrics = train_epoch(model, loader, optimizer, device, dataset_size, n_train_samples)
        
        epoch_metrics['loss'].append(metrics['avg_loss'])
        epoch_metrics['ce_loss'].append(metrics['avg_ce_loss'])
        epoch_metrics['kl'].append(metrics['avg_kl_loss'])
        epoch_metrics['accuracy'].append(metrics['accuracy'])
        epoch_metrics['grad_norm'].append(metrics.get('avg_grad_norm', 0))
        
        epoch_pbar.set_postfix(
            Acc=f"{metrics['accuracy']:.4f}", 
            CE=f"{metrics['avg_ce_loss']:.4f}", 
            KL=f"{metrics['avg_kl_loss']:.4f}",
            Loss=f"{metrics['avg_loss']:.4f}",
            Grad=f"{metrics.get('avg_grad_norm', 0):.4f}"
        )
        
        record_epoch_metrics(record_metric_fn, t_idx, epoch, metrics, layer_stats, model_type)
        
        layer_stats = compute_vcl_model_stats(model)
    
    clean_memory(device)
    
    print(end_msg)
    
    if exp_dir:
        model_type_suffix = model_type.lower() if model_type else None
        create_task_training_plots(task_idx, epoch_metrics, dataset_size, exp_dir, model_type=model_type_suffix)
    
    if model_type is None or model_type.lower() == 'propagation':
        print_vcl_model_stats(model, task_idx, model_type)
    
    model_type_prefix = f"{model_type} Model" if model_type else "Model"
    print(f"    Training Results for {model_type_prefix} after Task {task_idx}:")
    print(f"    Task {task_idx}: Acc={metrics['accuracy']:.4f} | average CE={metrics['avg_ce_loss']:.4f} | " +
          f"Total CE={metrics['total_ce_loss']:.4f} | KL={metrics['avg_kl_loss']:.4f} | Loss={metrics['avg_loss']:.4f}")
    
    return model, metrics

def setup_directories(exp_dir):
    if not exp_dir:
        return None, None
        
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return exp_dir, checkpoint_dir

def initialize_model(model_class, train_loader, test_loader, device, use_ml_initialization, ml_epochs, lr,
                  init_std=0.001, adaptive_std=False, adaptive_std_epsilon=0.01, exp_dir=None,
                  different_perm_init=False):
    model = model_class().to(device)
    
    if use_ml_initialization:
        if different_perm_init:
            print("\nUsing ML initialization with a DIFFERENT permutation than Task #1...")
            from z_models.ml_initialization import get_ml_initialized_vcl_model
            model = get_ml_initialized_vcl_model(
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
                different_perm=True
            )
        else:
            print("\nUsing ML initialization for the first task...")
            from z_models.ml_initialization import get_ml_initialized_vcl_model
            model = get_ml_initialized_vcl_model(
                train_loader=train_loader,
                vcl_model=model,
                test_loader=test_loader,
                ml_epochs=ml_epochs,
                lr=lr,
                init_std=init_std,
                adaptive_std=adaptive_std,
                adaptive_std_epsilon=adaptive_std_epsilon,
                device=device,
                exp_dir=exp_dir
            )
        clean_memory(device)
    else:
        model.set_init_std(init_std, adaptive_std, adaptive_std_epsilon)
        std_type = "adaptive" if adaptive_std else "fixed"
        print(f"\nInitializing model with {std_type} standard deviation: {init_std}")
    
    return model

def save_checkpoint(model, checkpoint_dir, task_idx):
    if checkpoint_dir:
        checkpoint_path = checkpoint_dir / f"model_task{task_idx}.pt"
        torch.save(model.state_dict(), checkpoint_path)

def create_plots(exp_dir, avg_accuracies, task_accuracies):
    if not (exp_dir and avg_accuracies):
        return
        
    try:
        create_accuracy_plot(avg_accuracies, exp_dir)
        if task_accuracies:
            create_task_specific_plots(task_accuracies, exp_dir)
    except Exception as e:
        print(f"Error creating plots: {str(e)}")