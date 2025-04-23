
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
from pathlib import Path
import gc
import time
import warnings
import wandb

warnings.filterwarnings("ignore", message="A newer version of deeplake.*")
warnings.filterwarnings("ignore", category=UserWarning, module="deeplake")

from z_data.datasets import generate_permutations, create_permuted_mnist_loader_factories
from z_synaptic.train import train_with_si
from z_synaptic.si import SynapticIntelligence, StandardMLP
from z_utils.utils import clean_memory
from z_utils.plotting_utils import create_accuracy_plot, create_task_specific_plots
from z_synaptic.evaluation import evaluate_model
from z_utils.wandb_utils import init_wandb, finish_wandb

warnings.filterwarnings("ignore", message="A newer version of deeplake.*", category=UserWarning, module="deeplake.util.check_latest_version")


def setup_experiment_dirs(experiment_dir):
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "plots").mkdir(exist_ok=True)
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    return experiment_dir

def create_metric_recorder():
    metrics = []
    
    def record_metric(task, epoch, name, value):
        value_to_record = float(value) if isinstance(value, (int, float)) else value
        metrics.append({
            'task': task,
            'epoch': epoch,
            'metric': name,
            'value': value_to_record
        })
        
    return record_metric, metrics

def cleanup_resources(train_loader_factories=None, test_loader_factories=None, permutations=None):

    if train_loader_factories is not None:
        del train_loader_factories
    if test_loader_factories is not None:
        del test_loader_factories
    if permutations is not None:
        del permutations
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    gc.collect()

def create_data_loader_factories(config, num_workers):
    experiment_type = config.get("experiment_type", "permuted_mnist")
    batch_size = config.get("batch_size", 256)
    num_tasks = config.get("num_tasks", 10)
    
    if experiment_type == "permuted_mnist":
        permutations = generate_permutations(num_tasks=num_tasks, device='cpu')
        
        train_loader_factories = create_permuted_mnist_loader_factories(
            root='data/',
            permutations=permutations,
            batch_size=batch_size,
            train=True,
            num_workers=num_workers
        )
        
        test_loader_factories = create_permuted_mnist_loader_factories(
            root='data/',
            permutations=permutations,
            batch_size=batch_size*2,  
            train=False,
            num_workers=num_workers
        )
        
        return train_loader_factories, test_loader_factories, permutations
    
    raise ValueError(f"Experiment type {experiment_type} not supported")

def create_model_factory(config, device):
    experiment_type = config.get("experiment_type", "permuted_mnist")
    hidden_size = config.get("hidden_size", 100)
    
    if experiment_type == "permuted_mnist":
        def create_model():
            return StandardMLP(
                input_size=784, 
                hidden_size=hidden_size, 
                output_size=10
            ).to(device)
            
        return create_model
    
    raise ValueError(f"Experiment type {experiment_type} not supported")

def save_error_log(error, config, experiment_dir):
    import traceback
    from datetime import datetime
    
    error_file = experiment_dir / "error_log.txt"
    with open(error_file, "w") as f:
        f.write(f"Error at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Error message: {str(error)}\n\n")
        f.write("Traceback:\n")
        f.write(traceback.format_exc())
        
        f.write("\nConfig:\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

def create_wandb_metric_recorder(config, metrics):
    epochs_per_task = config.get('epochs', 20)
    
    def record_metric_with_wandb(task, epoch, name, value):
        value_to_record = float(value) if isinstance(value, (int, float)) else value
        metrics.append({
            'task': task,
            'epoch': epoch,
            'metric': name,
            'value': value_to_record
        })
        
        if wandb.run is None:
            return
            
        if epoch >= 0:  # training metrics
            global_step = task * epochs_per_task + epoch
            
            if name.startswith('train_loss'):
                wandb.log({"training/total_loss": value_to_record}, step=global_step)
            elif name.startswith('train_ce_loss'):
                wandb.log({"training/ce_loss": value_to_record}, step=global_step)
            elif name.startswith('train_si_loss'):
                wandb.log({"training/si_loss": value_to_record}, step=global_step)
            elif name.startswith('train_accuracy'):
                wandb.log({"training/accuracy": value_to_record}, step=global_step)
            elif name.startswith('train_param_change'):
                wandb.log({"parameters/mean_change": value_to_record}, step=global_step)
            elif name.startswith('train_avg_omega'):
                wandb.log({"parameters/importance": value_to_record}, step=global_step)
        else:  # eval metrics
            task_end_step = (task + 1) * epochs_per_task
            
            if name.startswith('accuracy_on_task_'):
                task_num = name.split('_')[-1]
                wandb.log({
                    f"evaluation/task_{task_num}/accuracy": value_to_record
                }, step=task_end_step)
            elif name == 'average_accuracy':
                wandb.log({
                    "evaluation/average_accuracy": value_to_record
                }, step=task_end_step)
            elif name == 'total_duration_seconds':
                wandb.log({
                    "runtime/total_seconds": value_to_record,
                    "runtime/minutes": value_to_record / 60.0
                }, step=task_end_step)
    
    return record_metric_with_wandb

def run_si_experiment(config, experiment_dir, device='cuda'):
    experiment_dir = setup_experiment_dirs(experiment_dir)
    
    record_metric, metrics = create_metric_recorder()
    
    train_loader_factories, test_loader_factories, permutations = create_data_loader_factories(
        config, config['num_workers']
    )
    model_factory = create_model_factory(config, device)
    
    try:
        model = model_factory()
        
        si_params = {
            'model': model,
            'train_loader_factories': train_loader_factories,
            'test_loader_factories': test_loader_factories,
            'epochs_per_task': config['epochs'],
            'lr': config['lr'],
            'device': device,
            'lambda_reg': config.get('si_lambda', 1.0),
            'epsilon': config.get('si_epsilon', 1e-3),
            'omega_decay': config.get('omega_decay', 0.9),
            'momentum': config.get('momentum', 0.9),
            'optimizer_type': config.get('optimizer_type', 'sgd'),
            'record_metric_fn': record_metric,
            'exp_dir': experiment_dir,
            'n_eval_samples': config.get('n_eval_samples', 100),
            'early_stopping_threshold': config.get('early_stopping_threshold', None)
        }
        
        start_time = time.time()
        
        trained_model, avg_accuracies, task_accuracies = train_with_si(**si_params)
        
        total_time = time.time() - start_time
        record_metric(config['num_tasks']-1, -1, 'total_duration_seconds', total_time)
        print(f"\nTotal training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(experiment_dir / "metrics.csv", index=False)
        
        torch.save(trained_model.state_dict(), experiment_dir / "checkpoints" / "final_model.pt")
        
        return metrics_df
    
    except Exception as e:
        save_error_log(e, config, experiment_dir)
        raise
    
    finally:
        cleanup_resources(train_loader_factories, test_loader_factories, permutations)

def run_si_experiment_with_wandb(config, experiment_dir, project_name, device='cuda'):
    experiment_dir = setup_experiment_dirs(experiment_dir)
    
    init_wandb(config, project_name=project_name, 
               tags=["synaptic_intelligence", f"tasks={config['num_tasks']}"])
    
    metrics = []
    record_metric_with_wandb = create_wandb_metric_recorder(config, metrics)
    
    train_loader_factories, test_loader_factories, permutations = create_data_loader_factories(
        config, config['num_workers']
    )
    model_factory = create_model_factory(config, device)
    
    try:
        model = model_factory()
        
        si_params = {
            'model': model,
            'train_loader_factories': train_loader_factories,
            'test_loader_factories': test_loader_factories,
            'epochs_per_task': config['epochs'],
            'lr': config['lr'],
            'device': device,
            'lambda_reg': config.get('si_lambda', 1.0),
            'epsilon': config.get('si_epsilon', 1e-3),
            'omega_decay': config.get('omega_decay', 0.9),
            'momentum': config.get('momentum', 0.9),
            'optimizer_type': config.get('optimizer_type', 'sgd'),
            'record_metric_fn': record_metric_with_wandb,
            'exp_dir': experiment_dir,
            'n_eval_samples': config.get('n_eval_samples', 100),
            'early_stopping_threshold': config.get('early_stopping_threshold', None)
        }
        
        start_time = time.time()
        
        trained_model, avg_accuracies, task_accuracies = train_with_si(**si_params)
        
        total_time = time.time() - start_time
        record_metric_with_wandb(config['num_tasks']-1, -1, 'total_duration_seconds', total_time)
        print(f"\nTotal training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        # log plots
        for plot_file in (experiment_dir / "plots").glob("*.png"):
            wandb.log({f"plots/{plot_file.stem}": wandb.Image(str(plot_file))})
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(experiment_dir / "metrics.csv", index=False)
        
        torch.save(trained_model.state_dict(), experiment_dir / "checkpoints" / "final_model.pt")
        
        return metrics_df
    
    except Exception as e:
        save_error_log(e, config, experiment_dir)
        raise
    
    finally:
        finish_wandb()
        cleanup_resources(train_loader_factories, test_loader_factories, permutations)

def main():
    config = {
        "experiment_type": "permuted_mnist",
        "num_tasks": 10,
        "epochs": 5,
        "batch_size": 256,
        "lr": 1e-3,
        "hidden_size": 100,
        "num_workers": 4,
        "si_lambda": 0.1,  
    }
    
    run_si_experiment(config, "experiments/si_permuted_mnist", device='cuda')

if __name__ == "__main__":
    main() 