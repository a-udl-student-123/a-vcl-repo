# wandb logging 

import torch
import wandb
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

def init_wandb(config, project_name="si-gaussian", tags=None):
    if tags is None:
        tags = ["synaptic_intelligence", "gaussian"]
    
    optimizer_type = config.get("optimizer_type", "sgd")
    num_tasks = config.get("num_tasks", 10)
    
    run_name = f"si_gaussian_{optimizer_type}_"
    run_name += f"ep{config.get('epochs', 5)}_"
    run_name += f"lr{config.get('lr', 1e-3)}_"
    run_name += f"lambda{config.get('si_lambda', 1.0)}_"
    run_name += f"eps{config.get('si_epsilon', 1e-4)}_"
    run_name += f"tasks{num_tasks}"
    
    wandb.init(
        project=project_name,
        config=config,
        name=run_name,
        tags=tags
    )
    
    print(f"Started wandb: {run_name}")
    return wandb.run

def log_training_metrics_gaussian(metrics, task_idx, epoch, epochs_per_task):
    wandb.log({
        "train/loss": metrics.get("avg_loss", 0),
        "train/nll_loss": metrics.get("avg_nll_loss", 0),
        "train/si_loss": metrics.get("avg_si_loss", 0),
        "train/rmse": metrics.get("avg_rmse", 0),
        "train/accuracy": metrics.get("avg_accuracy", 0),
        "train/grad_norm": metrics.get("avg_grad_norm", 0),
        "train/variance": metrics.get("avg_variance", 0),
        
        "params/omega": metrics.get("avg_omega", 0),
        "params/change": metrics.get("avg_param_change", 0),
    }, step=task_idx*epochs_per_task+epoch)

def log_evaluation_metrics_gaussian(metrics, task_idx, epochs_per_task):
    rmse_by_task = metrics.get("task_rmse_values", [])
    accuracy_by_task = metrics.get("task_accuracy_values", [])
    nll_by_task = metrics.get("task_nll_values", [])
    
    avg_rmse = metrics.get("avg_rmse", 0)
    avg_accuracy = metrics.get("avg_accuracy", 0)
    avg_nll = metrics.get("avg_nll", 0)
    avg_variance = metrics.get("avg_variance", 0)
    
    # log per-task metrics
    for i, values in enumerate(zip(
        rmse_by_task, accuracy_by_task, nll_by_task
    )):
        rmse, acc, nll = values
        wandb.log({
            f"eval/task{i+1}_rmse": rmse,
            f"eval/task{i+1}_accuracy": acc,
            f"eval/task{i+1}_nll": nll,
        }, step= (task_idx + 1) * epochs_per_task)
    
    wandb.log({
        "eval/avg_rmse": avg_rmse,
        "eval/avg_accuracy": avg_accuracy,
        "eval/avg_nll": avg_nll,
        "eval/avg_variance": avg_variance,
    }, step= (task_idx + 1) * epochs_per_task)

def create_wandb_metric_recorder_si_gaussian(config, metrics):
    epochs_per_task = config.get('epochs', 5)
    
    def record_metric_with_wandb(task, epoch, name, value):
        value_to_record = float(value) if isinstance(value, (int, float)) else value
        metrics.append({
            'task': task,
            'epoch': epoch,
            'metric': name,
            'value': value_to_record
        })
        
        # skip pred model metrics
        if name.startswith('train_prediction'):
            return 
        
        # training metrics
        if epoch >= 0 and task >= 0 and name.startswith('train_'):
            if name == 'train_avg_omega':
                training_metrics = {
                    'avg_loss': value_to_record,
                    'avg_nll_loss': next((m['value'] for m in metrics if m['task'] == task and 
                                      m['epoch'] == epoch and m['metric'] == 'train_nll_loss'), 0),
                    'avg_si_loss': next((m['value'] for m in metrics if m['task'] == task and 
                                      m['epoch'] == epoch and m['metric'] == 'train_si_loss'), 0),
                    'avg_rmse': next((m['value'] for m in metrics if m['task'] == task and 
                                   m['epoch'] == epoch and m['metric'] == 'train_rmse'), 0),
                    'avg_accuracy': next((m['value'] for m in metrics if m['task'] == task and 
                                       m['epoch'] == epoch and m['metric'] == 'train_accuracy'), 0),
                    'avg_grad_norm': next((m['value'] for m in metrics if m['task'] == task and 
                                        m['epoch'] == epoch and m['metric'] == 'train_grad_norm'), 0),
                    'avg_omega': next((m['value'] for m in metrics if m['task'] == task and 
                                    m['epoch'] == epoch and m['metric'] == 'train_avg_omega'), 0),
                    'avg_param_change': next((m['value'] for m in metrics if m['task'] == task and 
                                           m['epoch'] == epoch and m['metric'] == 'train_param_change'), 0),
                    'avg_variance': next((m['value'] for m in metrics if m['task'] == task and 
                                    m['epoch'] == epoch and m['metric'] == 'train_variance'), 0),
                }
                log_training_metrics_gaussian(training_metrics, task, epoch, epochs_per_task)
        
        # eval metrics
        if epoch < 0 and name == 'average_variance':
            task_rmse_values = []
            task_accuracy_values = []
            task_nll_values = []
            
            for t in range(task + 1):
                rmse = next((m['value'] for m in metrics if m['task'] == task and 
                          m['metric'] == f'rmse_on_task_{t+1}'), None)
                accuracy = next((m['value'] for m in metrics if m['task'] == task and 
                              m['metric'] == f'accuracy_on_task_{t+1}'), None)
                nll = next((m['value'] for m in metrics if m['task'] == task and 
                         m['metric'] == f'nll_on_task_{t+1}'), None)
                
                if rmse is not None:
                    task_rmse_values.append(rmse)
                if accuracy is not None:
                    task_accuracy_values.append(accuracy)
                if nll is not None:
                    task_nll_values.append(nll)
            
            avg_rmse = next((m['value'] for m in metrics if m['task'] == task and 
                          m['metric'] == 'average_rmse'), 0)
            avg_accuracy = next((m['value'] for m in metrics if m['task'] == task and 
                              m['metric'] == 'average_accuracy'), 0)
            avg_nll = next((m['value'] for m in metrics if m['task'] == task and 
                         m['metric'] == 'average_nll'), 0)
            avg_variance = next((m['value'] for m in metrics if m['task'] == task and 
                         m['metric'] == 'average_variance'), 0)
            
            log_evaluation_metrics_gaussian({
                "task_rmse_values": task_rmse_values,
                "task_accuracy_values": task_accuracy_values,
                "task_nll_values": task_nll_values,
                "avg_rmse": avg_rmse,
                "avg_accuracy": avg_accuracy,
                "avg_variance": avg_variance,
                "avg_nll": avg_nll,
            }, task, epochs_per_task)
    
    return record_metric_with_wandb

def finish_wandb():
    if wandb.run is not None:
        wandb.finish()