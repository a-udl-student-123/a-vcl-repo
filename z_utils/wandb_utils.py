
import wandb
import torch
import numpy as np
from pathlib import Path

def init_wandb(config, project_name="vcl-experiments", group=None, tags=None):
    return wandb.init(
        project=project_name,
        config=config,
        group=group,
        tags=tags
    )

def log_training_metrics(metrics, task_idx, epoch, epochs_per_task):
    # calc global step
    global_step = (task_idx * epochs_per_task) + epoch
    
    wandb.log({
        "training/losses/cross_entropy_loss": metrics["avg_ce_loss"],
        "training/losses/kl_divergence": metrics["avg_kl_loss"], 
        "training/losses/total_loss": metrics["avg_loss"],
        "training/accuracy": metrics["accuracy"],
        "training/gradient_norm": metrics.get("avg_grad_norm", 0.0),
        
        "parameters/weights/standard_deviation": metrics["avg_weight_std"],
        "parameters/biases/standard_deviation": metrics["avg_bias_std"],
        "parameters/weights/mean_absolute": metrics["avg_weight_mu_abs"],
        "parameters/biases/mean_absolute": metrics["avg_bias_mu_abs"],
    }, step=global_step)

def log_evaluation_metrics(metrics, task_idx, epochs_per_task):
    task_end_step = (task_idx + 1) * epochs_per_task
    
    for eval_task_idx, accuracy in enumerate(metrics["task_accuracies"]):
        wandb.log({
            f"evaluation/task_{eval_task_idx+1}/accuracy": accuracy,
            f"evaluation/task_{eval_task_idx+1}/ce_loss": metrics["task_ce_losses"][eval_task_idx],
        }, step=task_end_step)
    
    wandb.log({
        "evaluation/average_accuracy": metrics["avg_accuracy"],
        "evaluation/average_ce_loss": metrics["avg_ce_loss"]
    }, step=task_end_step)
    
    task_forgetting = metrics.get("task_forgetting", [])
    if task_forgetting and isinstance(task_forgetting, list) and len(task_forgetting) > 0:
        for eval_task_idx, forgetting in enumerate(task_forgetting):
            wandb.log({
                f"evaluation/forgetting/task_{eval_task_idx+1}": forgetting,
            }, step=task_end_step)

def log_layer_statistics(layer_stats, task_idx, epoch, epochs_per_task):
    global_step = (task_idx * epochs_per_task) + epoch
    
    metrics = {}
    for layer_idx, stats in layer_stats.items():
        layer_name = f"layer_{layer_idx+1}"
        metrics[f"parameters/layers/{layer_name}/standard_deviation"] = stats["std"]
        metrics[f"parameters/layers/{layer_name}/mean_absolute"] = stats["mean_abs"]
    
    wandb.log(metrics, step=global_step)

def finish_wandb():
    if wandb.run is not None:
        wandb.finish() 