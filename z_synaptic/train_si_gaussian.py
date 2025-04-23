# training functions for synaptic intelligence with gaussian likelihood
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from tqdm import tqdm
import warnings
import math

warnings.filterwarnings("ignore", message="A newer version of deeplake.*")
warnings.filterwarnings("ignore", category=UserWarning, module="deeplake")

from z_utils.utils import clean_memory, clean_loader
from z_utils.plotting_utils import create_accuracy_plot, create_task_specific_plots
from z_synaptic.si import SynapticIntelligence
from z_synaptic.evaluation import evaluate_model, evaluate_all_tasks, print_eval_results

def convert_to_onehot(labels, num_classes=10):
    # already 1hot
    if labels.dim() > 1:
        return labels
        
    onehot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    return onehot.scatter_(1, labels.unsqueeze(1), 1.0)

def gaussian_nll_loss(mean, logvar, target, reduction='mean'):
    # negative ll
    var = torch.exp(logvar) + 1e-10
    nll = 0.5 * (logvar + (target - mean).pow(2) / var)
    
    if reduction == 'none':
        return nll
    elif reduction == 'sum':
        return nll.sum()
    else:  
        return nll.mean()

def train_epoch_with_si_gaussian(model, loader, optimizer, si, device, dataset_size):
    model.train()
    
    total_loss = 0.0
    total_nll_loss = 0.0
    total_si_loss = 0.0
    total_rmse = 0.0
    total_variance = 0.0
    correct = 0
    total = 0
    
    batch_pbar = tqdm(loader, desc="Batches", leave=False)
    for x_batch, y_batch in batch_pbar:
        x_batch = x_batch.to(device)
        
        y_batch = convert_to_onehot(y_batch, num_classes=model.output_size)
        y_batch = y_batch.to(device)
        
        batch_size = x_batch.size(0)
        
        optimizer.zero_grad()
        means, logvars = model(x_batch)
        nll_loss = gaussian_nll_loss(means, logvars, y_batch)
        
        batch_variance = torch.exp(logvars).mean().item()
        
        si_loss = si.compute_regularization_loss()
        
        # catch unstable si loss
        if torch.isnan(si_loss) or torch.isinf(si_loss) or si_loss > 1000:
            print(f"WARNING: SI loss unstable ({si_loss}), zeroing")
            si_loss = torch.tensor(0.0, device=device)
        
        loss = nll_loss + si_loss
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: total loss is NaN/Inf, using only NLL")
            loss = nll_loss
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        si.before_step()
        optimizer.step()
        si.accumulate_path_integral()
        
        pred = torch.argmax(means, dim=1)
        target = torch.argmax(y_batch, dim=1)
        batch_correct = (pred == target).sum().item()
        
        batch_rmse = torch.sqrt(((means - y_batch) ** 2).mean()).item()
        
        total_loss += loss.item() * batch_size
        total_nll_loss += nll_loss.item() * batch_size
        si_loss_value = si_loss.item() if isinstance(si_loss, torch.Tensor) else si_loss
        total_si_loss += si_loss_value * batch_size
        total_rmse += batch_rmse * batch_size
        total_variance += batch_variance * batch_size
        correct += batch_correct
        total += batch_size
        
        batch_pbar.set_postfix(
            loss=f"{loss.item():.4f}", 
            nll=f"{nll_loss.item():.4f}", 
            si=f"{si_loss_value:.4f}" if isinstance(si_loss, torch.Tensor) else f"{si_loss:.4f}", 
            rmse=f"{batch_rmse:.4f}",
            var=f"{batch_variance:.4f}",
            acc=f"{batch_correct/batch_size:.4f}"
        )
    
    avg_loss = total_loss / total if total > 0 else 0
    avg_nll_loss = total_nll_loss / total if total > 0 else 0
    avg_si_loss = total_si_loss / total if total > 0 else 0
    avg_rmse = total_rmse / total if total > 0 else 0
    avg_variance = total_variance / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return {
        'loss': avg_loss,
        'nll_loss': avg_nll_loss,
        'si_loss': avg_si_loss,
        'rmse': avg_rmse,
        'variance': avg_variance,
        'accuracy': accuracy
    }

def train_task_with_si_gaussian(model, loader, optimizer, si, task_idx, epochs, device, record_metric_fn=None):
    dataset_size = len(loader.dataset)
    
    print(f"\nTraining SI Gaussian model on Task {task_idx+1} for {epochs} epochs...")
    
    if hasattr(model, 'set_current_task'):
        model.set_current_task(task_idx)
        print(f"Set model to use head for task {task_idx+1}")
    
    epoch_pbar = tqdm(range(1, epochs + 1), desc="Epochs", leave=False)
    
    for epoch in epoch_pbar:
        metrics = train_epoch_with_si_gaussian(model, loader, optimizer, si, device, dataset_size)
        
        epoch_pbar.set_postfix(
            loss=f"{metrics['loss']:.4f}", 
            nll=f"{metrics['nll_loss']:.4f}", 
            si=f"{metrics['si_loss']:.4f}",
            rmse=f"{metrics['rmse']:.4f}",
            var=f"{metrics['variance']:.4f}",
            acc=f"{metrics['accuracy']:.4f}"
        )
        
        if record_metric_fn:
            record_metric_fn(task_idx, epoch, 'train_loss', metrics['loss'])
            record_metric_fn(task_idx, epoch, 'train_nll_loss', metrics['nll_loss'])
            record_metric_fn(task_idx, epoch, 'train_si_loss', metrics['si_loss'])
            record_metric_fn(task_idx, epoch, 'train_rmse', metrics['rmse'])
            record_metric_fn(task_idx, epoch, 'train_variance', metrics['variance'])
            record_metric_fn(task_idx, epoch, 'train_accuracy', metrics['accuracy'])
            
            if hasattr(si, '_debug_prev_state') and hasattr(model, 'parameters'):
                param_norm_sum = 0.0
                param_count = 0
                for name, param in model.named_parameters():
                    if name in si.omega:
                        param_norm_sum += torch.norm(param.detach() - si.old_params[name]).item()
                        param_count += 1
                
                avg_param_change = param_norm_sum / param_count if param_count > 0 else 0
                record_metric_fn(task_idx, epoch, 'train_param_change', avg_param_change)
                
                omega_norm = 0.0
                omega_count = 0
                for name, omega_val in si.omega.items():
                    omega_norm += torch.norm(omega_val).item()
                    omega_count += omega_val.numel()
                
                avg_omega = omega_norm / omega_count if omega_count > 0 else 0
                record_metric_fn(task_idx, epoch, 'train_avg_omega', avg_omega)
    
    clean_memory(device)
    
    print(f"\nFinished training SI Gaussian model on Task {task_idx+1}")
    
    print(f"  Final metrics for Task {task_idx+1}:")
    print(f"  Loss: {metrics['loss']:.4f} | NLL: {metrics['nll_loss']:.4f} | "
          f"SI: {metrics['si_loss']:.4f} | RMSE: {metrics['rmse']:.4f} | Variance: {metrics['variance']:.4f} | Accuracy: {metrics['accuracy']:.4f}")
    
    return metrics

def train_with_si_gaussian(model, train_loader_factories, test_loader_factories, 
                 epochs_per_task=5, lr=1e-3, device='cuda', 
                 lambda_reg=1.0, epsilon=1e-4, omega_decay=0.9, momentum=0.9,
                 optimizer_type='sgd', record_metric_fn=None, exp_dir=None,
                 early_stopping_threshold=None, n_eval_samples=100, evaluate_all_tasks_fn=None):
    
    si = SynapticIntelligence(model, lambda_reg=lambda_reg, epsilon=epsilon, 
                             omega_decay=omega_decay, device=device)
    
    task_accuracies = []
    avg_accuracies = []
    test_loaders = []
    
    num_tasks = len(train_loader_factories)
    task_pbar = tqdm(range(num_tasks), desc="Tasks", leave=True)
    for task_idx in task_pbar:
        task_pbar.set_description(f"Task {task_idx+1}")
        
        if hasattr(model, 'set_current_task'):
            model.set_current_task(task_idx)
            print(f"Set model to task {task_idx+1}")
        
        train_loader = train_loader_factories[task_idx]()
        
        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        
        train_metrics = train_task_with_si_gaussian(model, train_loader, optimizer, si, task_idx, 
                          epochs_per_task, device, record_metric_fn)
        
        si.update_omega()
        
        while len(test_loaders) <= task_idx:
            test_loaders.append(test_loader_factories[len(test_loaders)]())
        
        model.eval()
        
        if evaluate_all_tasks_fn is not None:
            metrics = evaluate_all_tasks_fn(
                model, test_loaders[:task_idx+1], task_idx+1, device, n_eval_samples, record_metric_fn
            )
        else:
            metrics = evaluate_all_tasks_gaussian(
                model, test_loaders[:task_idx+1], task_idx+1, device, n_eval_samples, record_metric_fn
            )
        
        print_eval_results_gaussian(task_idx+1, metrics)
        
        task_accuracies.append(metrics['accuracy_values'])
        avg_accuracies.append(metrics['avg_accuracy'])
        
        if record_metric_fn:
            record_metric_fn(task_idx, -1, 'average_accuracy', metrics['avg_accuracy'])
            record_metric_fn(task_idx, -1, 'average_rmse', metrics['avg_rmse'])
            record_metric_fn(task_idx, -1, 'average_nll', metrics['avg_nll'])
            record_metric_fn(task_idx, -1, 'average_variance', metrics['avg_variance'])

        if early_stopping_threshold is not None and metrics['avg_accuracy'] < early_stopping_threshold:
            print(f"\n⚠️ EARLY STOPPING: Average accuracy ({metrics['avg_accuracy']:.4f}) fell below threshold ({early_stopping_threshold:.4f})")
            print("Terminating training early to save computation resources.")
            
            for loader in test_loaders:
                clean_loader(loader)
            clean_loader(train_loader)
            #TODO
            try:
                import wandb
                if wandb.run is not None:
                    wandb.run.summary["early_stopped"] = True
                    wandb.run.summary["early_stopping_reason"] = f"Accuracy {metrics['avg_accuracy']:.4f} below threshold {early_stopping_threshold:.4f}"
                    wandb.run.summary["early_stopping_task"] = task_idx + 1
            except:
                pass
                
            if exp_dir:
                create_accuracy_plot(avg_accuracies, exp_dir)
                create_task_specific_plots(task_accuracies, exp_dir)
                
            return model, avg_accuracies, task_accuracies
        
        clean_loader(train_loader)
    
    for loader in test_loaders:
        clean_loader(loader)
    
    if exp_dir:
        create_accuracy_plot(avg_accuracies, exp_dir)
        create_task_specific_plots(task_accuracies, exp_dir)
    
    return model, avg_accuracies, task_accuracies

def evaluate_model_gaussian(model, dataloader, device='cuda', n_samples=100):
    model.eval()
    model = model.to(device)
    
    total_squared_error = 0.0
    total_correct = 0
    total_nll = 0.0
    total_variance = 0.0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            
            labels = convert_to_onehot(labels, num_classes=model.output_size)
            labels = labels.to(device, non_blocking=True)
            
            means, logvars = model(inputs)
            
            squared_errors = ((means - labels) ** 2).sum()
            
            batch_nll = gaussian_nll_loss(means, logvars, labels)
            
            batch_variance = torch.exp(logvars).mean().item()
            
            pred_indices = torch.argmax(means, dim=1)
            target_indices = torch.argmax(labels, dim=1)
            batch_correct = (pred_indices == target_indices).float().sum()
            
            batch_size = inputs.size(0)
            total_squared_error += squared_errors.item()
            total_nll += batch_nll.item() * batch_size
            total_variance += batch_variance * batch_size
            total_correct += batch_correct.item()
            total += batch_size
            
            del inputs, labels, means, logvars
    
    dim = model.mean_head.out_features
    rmse = math.sqrt(total_squared_error / (total * dim)) if total > 0 else 0
    accuracy = total_correct / total if total > 0 else 0
    nll_loss = total_nll / total if total > 0 else 0
    avg_variance = total_variance / total if total > 0 else 0
    
    clean_memory(device)
    
    return rmse, accuracy, nll_loss, avg_variance

def evaluate_all_tasks_gaussian(model, test_loaders, current_task, device, n_eval_samples, record_metric_fn=None):
    from tqdm import tqdm
    
    model.eval()
    rmse_values = []
    accuracy_values = []
    nll_values = []
    variance_values = []
    
    num_tasks_to_eval = min(current_task, len(test_loaders))
    
    eval_pbar = tqdm(range(num_tasks_to_eval), desc="Task Eval", leave=False)
    for task_idx in eval_pbar:
        rmse, accuracy, nll, variance = evaluate_model_gaussian(
            model, test_loaders[task_idx], device, n_eval_samples
        )
        
        if record_metric_fn:
            t_idx = current_task - 1
            record_metric_fn(t_idx, -1, f'rmse_on_task_{task_idx+1}', rmse)
            record_metric_fn(t_idx, -1, f'accuracy_on_task_{task_idx+1}', accuracy)
            record_metric_fn(t_idx, -1, f'nll_on_task_{task_idx+1}', nll)
            record_metric_fn(t_idx, -1, f'variance_on_task_{task_idx+1}', variance)
        
        rmse_values.append(rmse)
        accuracy_values.append(accuracy)
        nll_values.append(nll)
        variance_values.append(variance)
        
        eval_pbar.set_postfix(
            rmse=f"{rmse:.4f}", 
            acc=f"{accuracy:.4f}", 
            nll=f"{nll:.4f}",
            var=f"{variance:.4f}",
            task=f"{task_idx+1}"
        )
    
    clean_memory(device)
    
    avg_rmse = sum(rmse_values) / len(rmse_values) if rmse_values else 0
    avg_accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0
    avg_nll = sum(nll_values) / len(nll_values) if nll_values else 0
    avg_variance = sum(variance_values) / len(variance_values) if variance_values else 0
    
    metrics = {
        'rmse_values': rmse_values,
        'accuracy_values': accuracy_values,
        'nll_values': nll_values,
        'variance_values': variance_values,
        'avg_rmse': avg_rmse,
        'avg_accuracy': avg_accuracy,
        'avg_nll': avg_nll,
        'avg_variance': avg_variance
    }
    
    return metrics

def print_eval_results_gaussian(task_idx, metrics):
    print(f"    Evaluation Results after Task {task_idx}:")
    
    rmse_values = metrics['rmse_values']
    accuracy_values = metrics['accuracy_values']
    nll_values = metrics['nll_values']
    variance_values = metrics['variance_values']
    
    for idx, (rmse, acc, nll, var) in enumerate(zip(
        rmse_values, accuracy_values, nll_values, variance_values), start=1):
        print(f"    Task {idx}: RMSE={rmse:.4f}, Acc={acc:.4f}, NLL={nll:.4f}, Variance={var:.4f}")
    
    print(f"    Average: RMSE={metrics['avg_rmse']:.4f}, Acc={metrics['avg_accuracy']:.4f}, " +
          f"NLL={metrics['avg_nll']:.4f}, Variance={metrics['avg_variance']:.4f}")
    print("    " + "-" * 80 + "\n")