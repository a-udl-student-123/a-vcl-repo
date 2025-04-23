# training functions for ssi
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message="A newer version of deeplake.*")
warnings.filterwarnings("ignore", category=UserWarning, module="deeplake")

from z_utils.utils import clean_memory, clean_loader
from z_utils.plotting_utils import create_accuracy_plot, create_task_specific_plots
from z_synaptic.si import SynapticIntelligence
from z_synaptic.evaluation import evaluate_model, evaluate_all_tasks, print_eval_results

def train_epoch_with_si(model, loader, optimizer, si, device, dataset_size):
    # train one epoch with SI 
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_si_loss = 0.0
    correct = 0
    total = 0
    
    batch_pbar = tqdm(loader, desc="Batches", leave=False)
    for x_batch, y_batch in batch_pbar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        batch_size = x_batch.size(0)
        
        optimizer.zero_grad()
        logits = model(x_batch)
        ce_loss = criterion(logits, y_batch)
        
        si_loss = si.compute_regularization_loss()
        
        if torch.isnan(si_loss) or torch.isinf(si_loss) or si_loss > 1000:
            print(f"WARNING: SI loss unstable: {si_loss}. Setting to 0 for this batch.")
            si_loss = torch.tensor(0.0, device=device)
        
        loss = ce_loss + si_loss
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Total loss is NaN/Inf. Using only CE loss.")
            loss = ce_loss
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        si.before_step()
        optimizer.step()
        si.accumulate_path_integral()
        
        _, pred = torch.max(logits, dim=1)
        batch_correct = (pred == y_batch).sum().item()
        
        total_loss += loss.item() * batch_size
        total_ce_loss += ce_loss.item() * batch_size
        si_loss_value = si_loss.item() if isinstance(si_loss, torch.Tensor) else si_loss
        total_si_loss += si_loss_value * batch_size
        correct += batch_correct
        total += batch_size
        
        batch_pbar.set_postfix(
            loss=f"{loss.item():.4f}", 
            ce=f"{ce_loss.item():.4f}", 
            si=f"{si_loss_value:.4f}" if isinstance(si_loss, torch.Tensor) else f"{si_loss:.4f}", 
            acc=f"{batch_correct/batch_size:.4f}"
        )
    
    avg_loss = total_loss / total if total > 0 else 0
    avg_ce_loss = total_ce_loss / total if total > 0 else 0
    avg_si_loss = total_si_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return {
        'loss': avg_loss,
        'ce_loss': avg_ce_loss,
        'si_loss': avg_si_loss,
        'accuracy': accuracy
    }

def train_task_with_si(model, loader, optimizer, si, task_idx, epochs, device, record_metric_fn=None):
    # train single task with sI
    dataset_size = len(loader.dataset)
    
    print(f"\nTraining SI model on Task {task_idx+1} for {epochs} epochs...")
    
    if hasattr(model, 'set_current_task'):
        model.set_current_task(task_idx)
        print(f"Set model to use head for task {task_idx+1}")
    
    epoch_pbar = tqdm(range(1, epochs + 1), desc="Epochs", leave=False)
    
    for epoch in epoch_pbar:
        metrics = train_epoch_with_si(model, loader, optimizer, si, device, dataset_size)
        
        epoch_pbar.set_postfix(
            loss=f"{metrics['loss']:.4f}", 
            ce=f"{metrics['ce_loss']:.4f}", 
            si=f"{metrics['si_loss']:.4f}", 
            acc=f"{metrics['accuracy']:.4f}"
        )
        
        if record_metric_fn:
            record_metric_fn(task_idx, epoch, 'train_loss', metrics['loss'])
            record_metric_fn(task_idx, epoch, 'train_ce_loss', metrics['ce_loss'])
            record_metric_fn(task_idx, epoch, 'train_si_loss', metrics['si_loss'])
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
                
                #  omega stats
                omega_norm = 0.0
                omega_count = 0
                for name, omega_val in si.omega.items():
                    omega_norm += torch.norm(omega_val).item()
                    omega_count += omega_val.numel()
                
                avg_omega = omega_norm / omega_count if omega_count > 0 else 0
                record_metric_fn(task_idx, epoch, 'train_avg_omega', avg_omega)
    
    clean_memory(device)
    
    print(f"\nFinished training SI model on Task {task_idx+1}")
    print(f"  Final metrics for Task {task_idx+1}:")
    print(f"  Loss: {metrics['loss']:.4f} | CE: {metrics['ce_loss']:.4f} | "
          f"SI: {metrics['si_loss']:.4f} | Accuracy: {metrics['accuracy']:.4f}")
    
    return metrics

def train_with_si(model, train_loader_factories, test_loader_factories, 
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
        
        # fresh optimizer per task
        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        
        train_task_with_si(model, train_loader, optimizer, si, task_idx, 
                          epochs_per_task, device, record_metric_fn)
        
        si.update_omega()
        
        while len(test_loaders) <= task_idx:
            test_loaders.append(test_loader_factories[len(test_loaders)]())
        
        model.eval()
        
        if evaluate_all_tasks_fn is not None:
            accuracies, ce_losses, avg_accuracy, avg_ce_loss = evaluate_all_tasks_fn(
                model, test_loaders[:task_idx+1], task_idx+1, device
            )
        else:
            accuracies, ce_losses, avg_accuracy, avg_ce_loss = evaluate_all_tasks(
                model, test_loaders[:task_idx+1], task_idx+1, device, record_metric_fn
            )
        
        print_eval_results(task_idx+1, accuracies, ce_losses)
        
        task_accuracies.append(accuracies)
        avg_accuracies.append(avg_accuracy)
        
        if record_metric_fn:
            record_metric_fn(task_idx, -1, 'average_accuracy', avg_accuracy)
            record_metric_fn(task_idx, -1, 'average_ce_loss', avg_ce_loss)
        
        # early stopping if acc too low - run useless
        if early_stopping_threshold is not None and avg_accuracy < early_stopping_threshold:
            print(f"\n⚠️ EARLY STOPPING: Average accuracy ({avg_accuracy:.4f}) fell below threshold ({early_stopping_threshold:.4f})")
            print("Terminating training early to save computation resources.")
            
            for loader in test_loaders:
                clean_loader(loader)
            clean_loader(train_loader)
            
            try:
                import wandb
                if wandb.run is not None:
                    wandb.run.summary["early_stopped"] = True
                    wandb.run.summary["early_stopping_reason"] = f"Accuracy {avg_accuracy:.4f} below threshold {early_stopping_threshold:.4f}"
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