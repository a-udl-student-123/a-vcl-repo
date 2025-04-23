

import torch
import pandas as pd
import warnings
from pathlib import Path
import gc

warnings.filterwarnings("ignore", message="A newer version of deeplake.*", category=UserWarning, module="deeplake.util.check_latest_version")

from z_data.datasets import generate_permutations, create_permuted_mnist_loader_factories
from z_synaptic.model_si_gaussian import HeteroscedasticSI_MLP
from z_synaptic.train_si_gaussian import train_with_si_gaussian
from z_utils.wandb_utils import finish_wandb

def setup_experiment_dirs(experiment_dir):
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "plots").mkdir(exist_ok=True)
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    return experiment_dir

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
    num_tasks = config.get("num_tasks", 10)
    batch_size = config.get("batch_size", 256)
    
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

def create_model_factory(config, device):
    hidden_size = config.get("hidden_size", 100)
    
    def create_model():
        return HeteroscedasticSI_MLP(
            input_size=784, 
            hidden_size=hidden_size, 
            output_size=10
        ).to(device)
        
    return create_model

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

def save_error_log(error, config, experiment_dir):
    import traceback
    from datetime import datetime
    
    error_file = experiment_dir / "error_log.txt"
    with open(error_file, "w") as f:
        f.write(f"Error occurred at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Error message: {str(error)}\n\n")
        f.write("Traceback:\n")
        f.write(traceback.format_exc())
        
        f.write("\nExperiment Config:\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

def run_si_gaussian_experiment(config, experiment_dir, device='cuda'):

    experiment_dir = setup_experiment_dirs(experiment_dir)
    
    record_metric, metrics = create_metric_recorder()
    
    train_loader_factories, test_loader_factories, permutations = create_data_loader_factories(
        config, config['num_workers']
    )
    model_factory = create_model_factory(config, device)
    
    record_metric(0, -1, 'init_std', config.get('init_std', 0.001))
    record_metric(0, -1, 'si_lambda', config.get('si_lambda', 1.0))
    record_metric(0, -1, 'si_epsilon', config.get('si_epsilon', 1e-4))
    
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
            'epsilon': config.get('si_epsilon', 1e-4),
            'omega_decay': config.get('omega_decay', 0.9),
            'momentum': config.get('momentum', 0.9),
            'optimizer_type': config.get('optimizer_type', 'sgd'),
            'record_metric_fn': record_metric,
            'exp_dir': experiment_dir,
            'n_eval_samples': config.get('n_eval_samples', 100),
            'early_stopping_threshold': config.get('early_stopping_threshold', None)
        }
        
        trained_model, avg_accuracies, task_accuracies = train_with_si_gaussian(**si_params)
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(experiment_dir / "metrics_gaussian.csv", index=False)
        
        return metrics_df
    
    except Exception as e:
        save_error_log(e, config, experiment_dir)
        raise
    
    finally:
        cleanup_resources(train_loader_factories, test_loader_factories, permutations)

def run_si_gaussian_experiment_with_wandb(config, experiment_dir, project_name, device='cuda'):
    from z_synaptic.wandb_utils_si_gaussian import (
        init_wandb, create_wandb_metric_recorder_si_gaussian, finish_wandb
    )
    
    experiment_dir = setup_experiment_dirs(experiment_dir)
    
    init_wandb(config, project_name=project_name, 
               tags=["synaptic_intelligence", f"tasks={config['num_tasks']}", "gaussian"])
    
    metrics = []
    record_metric_with_wandb = create_wandb_metric_recorder_si_gaussian(config, metrics)
    
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
            'epsilon': config.get('si_epsilon', 1e-4),
            'omega_decay': config.get('omega_decay', 0.9),
            'momentum': config.get('momentum', 0.9),
            'optimizer_type': config.get('optimizer_type', 'sgd'),
            'record_metric_fn': record_metric_with_wandb,
            'exp_dir': experiment_dir,
            'n_eval_samples': config.get('n_eval_samples', 100),
            'early_stopping_threshold': config.get('early_stopping_threshold', None)
        }
        
        trained_model, avg_accuracies, task_accuracies = train_with_si_gaussian(**si_params)
        
        try:
            import wandb
            for plot_file in (experiment_dir / "plots").glob("*.png"):
                wandb.log({plot_file.stem: wandb.Image(str(plot_file))})
        except Exception as e:
            print(f"Warning: Error logging plots to wandb: {str(e)}")
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(experiment_dir / "metrics_gaussian.csv", index=False)
        
        return metrics_df
    
    except Exception as e:
        save_error_log(e, config, experiment_dir)
        raise
    
    finally:
        finish_wandb()
        cleanup_resources(train_loader_factories, test_loader_factories, permutations) 