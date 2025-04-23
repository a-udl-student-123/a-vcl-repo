# wandb integration for gaussian vcl experiments

import torch
import wandb
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # avoid thread issues
import matplotlib.pyplot as plt

def init_wandb(config, project_name="vcl-gaussian", tags=None):
    if tags is None:
        tags = [config.get("method", "unknown")]
    
    method = config.get("method", "unknown")
    num_tasks = config.get("num_tasks", 10)
    coreset_size = config.get("coreset_size", 0)
    
    run_name = f"{method}_tasks-{num_tasks}"
    if method != "standard_vcl":
        run_name += f"_coreset-{coreset_size}"
    
    wandb.init(
        project=project_name,
        config=config,
        name=run_name,
        tags=tags
    )
    
    print(f"Initialized wandb run: {run_name}")
    return wandb.run

def log_training_metrics_gaussian(metrics, task_idx, epoch, epochs_per_task):
    wandb.log({
        "train/loss": metrics.get("avg_loss", 0),
        "train/nll_loss": metrics.get("avg_nll_loss", 0),
        "train/kl_loss": metrics.get("avg_kl_loss", 0),
        "train/rmse": metrics.get("avg_rmse", 0),
        "train/accuracy": metrics.get("avg_accuracy", 0),
        "train/grad_norm": metrics.get("avg_grad_norm", 0),
        "train/variance": metrics.get("avg_variance", 0),
        
        "params/weight_std": metrics.get("avg_weight_std", 0),
        "params/bias_std": metrics.get("avg_bias_std", 0),
        "params/weight_mu_abs": metrics.get("avg_weight_mu_abs", 0),
        "params/bias_mu_abs": metrics.get("avg_bias_mu_abs", 0)
    }, step=task_idx*epochs_per_task+epoch)

def log_evaluation_metrics_gaussian(metrics, task_idx, epochs_per_task):
    rmse_by_task = metrics.get("task_rmse_values", [])
    accuracy_by_task = metrics.get("task_accuracy_values", [])
    nll_by_task = metrics.get("task_nll_values", [])
    kl_by_task = metrics.get("task_kl_values", [])
    loss_by_task = metrics.get("task_loss_values", [])
    
    avg_rmse = metrics.get("avg_rmse", 0)
    avg_accuracy = metrics.get("avg_accuracy", 0)
    avg_nll = metrics.get("avg_nll", 0)
    avg_kl = metrics.get("avg_kl", 0)
    avg_loss = metrics.get("avg_loss", 0)
    avg_variance = metrics.get("avg_variance", 0)


    # log per-task metrics
    for i, values in enumerate(zip(
        rmse_by_task, accuracy_by_task, nll_by_task, kl_by_task, loss_by_task
    )):
        rmse, acc, nll, kl, loss = values
        wandb.log({
            f"eval/task{i+1}_rmse": rmse,
            f"eval/task{i+1}_accuracy": acc,
            f"eval/task{i+1}_nll": nll,
            f"eval/task{i+1}_kl": kl,
            f"eval/task{i+1}_loss": loss
        }, step= (task_idx + 1) * epochs_per_task)
    
    wandb.log({
        "eval/avg_rmse": avg_rmse,
        "eval/avg_accuracy": avg_accuracy,
        "eval/avg_nll": avg_nll,
        "eval/avg_kl": avg_kl,
        "eval/avg_loss": avg_loss,
        "eval/avg_variance": avg_variance,
    }, step= (task_idx + 1) * epochs_per_task)
    
def create_wandb_metric_recorder_gaussian(config, metrics):
    def record_metric_with_wandb(task, epoch, name, value):
        value_to_record = float(value) if isinstance(value, (int, float)) else value
        metrics.append({
            'task': task,
            'epoch': epoch,
            'metric': name,
            'value': value_to_record
        })
        
        # skip prediction model metrics
        if name.startswith('train_prediction'):
            return 
        
        # log training metrics (propagation model only)
        if epoch >= 0 and task >= 0 and name.startswith('train_') and '_bias_mu_abs' in name:
            prefix = name.rsplit('_bias_mu_abs', 1)[0]
            
            epoch_metrics = {
                'avg_weight_std': next((m['value'] for m in metrics if m['task'] == task and 
                                      m['epoch'] == epoch and m['metric'] == f"{prefix}_weight_std"), 0),
                'avg_bias_std': next((m['value'] for m in metrics if m['task'] == task and 
                                    m['epoch'] == epoch and m['metric'] == f"{prefix}_bias_std"), 0),
                'avg_weight_mu_abs': next((m['value'] for m in metrics if m['task'] == task and 
                                        m['epoch'] == epoch and m['metric'] == f"{prefix}_weight_mu_abs"), 0),
                'avg_bias_mu_abs': value_to_record,
                'avg_accuracy': next((m['value'] for m in metrics if m['task'] == task and 
                                m['epoch'] == epoch and m['metric'] == f"{prefix}_accuracy"), 0),
                'avg_loss': next((m['value'] for m in metrics if m['task'] == task and 
                                m['epoch'] == epoch and m['metric'] == f"{prefix}_loss"), 0),
                'avg_nll_loss': next((m['value'] for m in metrics if m['task'] == task and 
                                    m['epoch'] == epoch and m['metric'] == f"{prefix}_nll_loss"), 0),
                'avg_kl_loss': next((m['value'] for m in metrics if m['task'] == task and 
                                   m['epoch'] == epoch and m['metric'] == f"{prefix}_kl_loss"), 0),
                'avg_rmse': next((m['value'] for m in metrics if m['task'] == task and 
                                m['epoch'] == epoch and m['metric'] == f"{prefix}_rmse"), 0),
                'avg_grad_norm': next((m['value'] for m in metrics if m['task'] == task and 
                                     m['epoch'] == epoch and m['metric'] == f"{prefix}_grad_norm"), 0),
                'avg_variance': next((m['value'] for m in metrics if m['task'] == task and 
                                    m['epoch'] == epoch and m['metric'] == f"{prefix}_variance"), 0),
            }
            log_training_metrics_gaussian(epoch_metrics, task, epoch, config['epochs'])
        
        # log eval metrics
        if epoch < 0 and name == 'average_kl':
            log_wandb_evaluation_metrics_gaussian(metrics, task, config['epochs'])
            
    return record_metric_with_wandb

def log_wandb_evaluation_metrics_gaussian(metrics, task, epochs_per_task):
    task_rmse_values = []
    task_accuracy_values = []
    task_nll_values = []
    task_kl_values = []
    task_loss_values = []
    max_task_idx = task + 1
    
    for t in range(max_task_idx):
        rmse = next((m['value'] for m in metrics if m['task'] == task and 
                  m['metric'] == f'rmse_on_task_{t+1}'), None)
        accuracy = next((m['value'] for m in metrics if m['task'] == task and 
                      m['metric'] == f'accuracy_on_task_{t+1}'), None)
        nll = next((m['value'] for m in metrics if m['task'] == task and 
                 m['metric'] == f'nll_on_task_{t+1}'), None)
        kl = next((m['value'] for m in metrics if m['task'] == task and 
                m['metric'] == f'kl_on_task_{t+1}'), None)
        loss = next((m['value'] for m in metrics if m['task'] == task and 
                  m['metric'] == f'loss_on_task_{t+1}'), None)
        
        if rmse is not None:
            task_rmse_values.append(rmse)
        if accuracy is not None:
            task_accuracy_values.append(accuracy)
        if nll is not None:
            task_nll_values.append(nll)
        if kl is not None:
            task_kl_values.append(kl)
        if loss is not None:
            task_loss_values.append(loss)
    
    avg_rmse = next((m['value'] for m in metrics if m['task'] == task and 
                  m['metric'] == 'average_rmse'), 0)
    avg_accuracy = next((m['value'] for m in metrics if m['task'] == task and 
                      m['metric'] == 'average_accuracy'), 0)
    avg_nll = next((m['value'] for m in metrics if m['task'] == task and 
                 m['metric'] == 'average_nll'), 0)
    avg_kl = next((m['value'] for m in metrics if m['task'] == task and 
                m['metric'] == 'average_kl'), 0)
    avg_loss = next((m['value'] for m in metrics if m['task'] == task and 
                  m['metric'] == 'average_loss'), 0)
    avg_variance = next((m['value'] for m in metrics if m['task'] == task and 
                  m['metric'] == 'average_variance'), 0)
    
    log_evaluation_metrics_gaussian({
        "task_rmse_values": task_rmse_values,
        "task_accuracy_values": task_accuracy_values,
        "task_nll_values": task_nll_values,
        "task_kl_values": task_kl_values,
        "task_loss_values": task_loss_values,
        "avg_rmse": avg_rmse,
        "avg_accuracy": avg_accuracy,
        "avg_nll": avg_nll,
        "avg_kl": avg_kl,
        "avg_loss": avg_loss,
        "avg_variance": avg_variance,
    }, task, epochs_per_task)

def finish_wandb():
    if wandb.run is not None:
        wandb.finish()