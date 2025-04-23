import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import numpy as np
import random
from pathlib import Path
import gc
import time
import warnings
import wandb

warnings.filterwarnings("ignore", message="A newer version of deeplake.*")
warnings.filterwarnings("ignore", category=UserWarning, module="deeplake")

from z_data.datasets_dgm import (
    create_digit_mnist_loader_factories,
    create_letter_notmnist_loader_factories
)

from z_utils.utils import clean_memory, clean_loader
from z_utils.wandb_utils import init_wandb, finish_wandb

from z_synaptic_dgm.train import train_si_vae

from z_classifiers.classifier_utils import get_classifier

def set_seed(seed):
    if seed is None:
        return
        
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed} for reproducibility")

def setup_experiment_dirs(experiment_dir):
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    sample_dir = experiment_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    
    return experiment_dir

def cleanup_resources(train_loader_factories=None, test_loader_factories=None):
    if train_loader_factories is not None:
        del train_loader_factories
    if test_loader_factories is not None:
        del test_loader_factories
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    gc.collect()

def save_error_log(error, config, experiment_dir):
    import traceback
    from datetime import datetime
    
    error_file = experiment_dir / "error_log.txt"
    with open(error_file, "w") as f:
        f.write(f"Error occurred at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Error message: {str(error)}\n\n")
        f.write("Traceback:\n")
        f.write(traceback.format_exc())
        
        f.write("\nExperiment Configuration:\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

def create_wandb_metric_recorder(config):
    epochs_per_task = config.get('epochs', 20)
    
    def record_metric_with_wandb(task, epoch, name, value):
        if wandb.run is None:
            return
            
        if epoch >= 0:
            global_step = task * epochs_per_task + epoch
            
            if name.startswith('loss'):
                wandb.log({"training/total_loss": value}, step=global_step)
            elif name.startswith('recon_loss'):
                wandb.log({"training/recon_loss": value}, step=global_step)
            elif name.startswith('kl_latent'):
                wandb.log({"training/kl_latent": value}, step=global_step)
            elif name.startswith('si_loss'):
                wandb.log({"training/si_loss": value}, step=global_step)
            elif name.startswith('elbo'):
                wandb.log({"training/elbo": value}, step=global_step)
        else:
            task_end_step = (task + 1) * epochs_per_task
            
            if name.startswith('log_likelihood_task_'):
                task_num = name.split('_')[-1]
                wandb.log({
                    f"evaluation/tasks/{task_num}/log_likelihood": value
                }, step=task_end_step)
            elif name.startswith('recon_error_task_'):
                task_num = name.split('_')[-1]
                wandb.log({
                    f"evaluation/tasks/{task_num}/recon_error": value
                }, step=task_end_step)
            elif name.startswith('cls_uncertainty_task_'):
                task_num = name.split('_')[-1]
                wandb.log({
                    f"evaluation/tasks/{task_num}/cls_uncertainty": value
                }, step=task_end_step)
            elif name == 'average_log_likelihood':
                wandb.log({"evaluation/average_log_likelihood": value}, step=task_end_step)
            elif name == 'average_recon_error':
                wandb.log({"evaluation/average_recon_error": value}, step=task_end_step)
            elif name == 'average_classifier_uncertainty':
                wandb.log({"evaluation/average_classifier_uncertainty": value}, step=task_end_step)
            elif name == 'total_duration_seconds':
                wandb.log({
                    "runtime/total_seconds": value,
                    "runtime/minutes": value / 60.0
                }, step=task_end_step)
            elif name == 'early_stopped':
                wandb.run.summary["early_stopped"] = True
            elif name == 'early_stopping_task':
                wandb.run.summary["early_stopping_task"] = value
    
    return record_metric_with_wandb

def log_wandb_images(experiment_dir):
    if wandb.run is None:
        return
        
    try:
        for montage_file in (experiment_dir / "samples").glob("*montage*.png"):
            task = int(montage_file.stem.split('_')[0].replace('task', ''))
            montage_num = int(montage_file.stem.split('montage')[1])
            wandb.log({f"samples/task{task}_montage{montage_num}": wandb.Image(str(montage_file))})
        
        for recon_file in (experiment_dir / "samples").glob("*reconstructions.png"):
            task = int(recon_file.stem.split('_')[0].replace('task', ''))
            wandb.log({f"reconstructions/task{task}": wandb.Image(str(recon_file))})
    except Exception as e:
        print(f"Warning: Error logging images to wandb: {str(e)}")

def run_si_vae_experiment(config, experiment_dir, device='cuda'):
    experiment_dir = setup_experiment_dirs(experiment_dir)
    
    set_seed(config.get('seed', None))
    
    try:
        data_root = 'data/'
        
        if config.get('experiment_type', 'digit_mnist') == 'digit_mnist':
            train_loader_factories = create_digit_mnist_loader_factories(
                root=data_root,
                batch_size=config.get('batch_size', 128),
                train=True, 
                num_workers=config.get('num_workers', 4)
            )
            
            test_loader_factories = create_digit_mnist_loader_factories(
                root=data_root,
                batch_size=config.get('batch_size', 128) * 2,
                train=False,
                num_workers=config.get('num_workers', 4)
            )
            
        elif config.get('experiment_type') == 'letter_notmnist':
            train_loader_factories = create_letter_notmnist_loader_factories(
                root=data_root,
                batch_size=config.get('batch_size', 128),
                train=True, 
                num_workers=config.get('num_workers', 4)
            )
            
            test_loader_factories = create_letter_notmnist_loader_factories(
                root=data_root,
                batch_size=config.get('batch_size', 128) * 2,
                train=False,
                num_workers=config.get('num_workers', 4)
            )
        else:
            raise ValueError(f"Unsupported experiment type: {config.get('experiment_type')}")
        
        classifier = None
        if config.get('use_classifier', True):
            try:
                classifier = get_classifier(config.get('experiment_type', 'digit_mnist'), device)
                print(f"Loaded classifier for {config.get('experiment_type', 'digit_mnist')}")
            except Exception as e:
                print(f"Failed to load classifier: {str(e)}")
        
        metrics = []
        
        def record_metric(task, epoch, name, value):
            metrics.append({
                'task': task,
                'epoch': epoch,
                'metric': name,
                'value': value
            })
        
        model, task_metrics = train_si_vae(
            train_loader_factories=train_loader_factories,
            test_loader_factories=test_loader_factories,
            num_tasks=config.get('num_tasks', 10),
            hidden_size=config.get('hidden_size', 500),
            latent_size=config.get('latent_size', 50),
            input_size=config.get('input_size', 784),
            epochs=config.get('epochs', 20),
            learning_rate=config.get('lr', 1e-3),
            optimizer_type=config.get('optimizer_type', 'sgd'),
            momentum=config.get('momentum', 0.9),
            si_lambda=config.get('si_lambda', 0.1),
            si_epsilon=config.get('si_epsilon', 1e-3),
            omega_decay=config.get('omega_decay', 0.9),
            device=device,
            output_dir=experiment_dir,
            classifier=classifier,
            record_metric_fn=record_metric,
            early_stopping_threshold=config.get('early_stopping_threshold', None)
        )
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(experiment_dir / "metrics.csv", index=False)
        
        torch.save(model.state_dict(), experiment_dir / "checkpoints" / "final_model.pt")
        
        return metrics_df
    
    except Exception as e:
        save_error_log(e, config, experiment_dir)
        raise
    
    finally:
        cleanup_resources()

def run_si_vae_experiment_with_wandb(config, experiment_dir, project_name, device='cuda'):
    experiment_dir = setup_experiment_dirs(experiment_dir)
    
    set_seed(config.get('seed', None))
    
    init_wandb(config, project_name=project_name, 
               tags=["synaptic_intelligence", "dgm", f"tasks={config.get('num_tasks', 10)}"])
    
    try:
        data_root = 'data/'
        
        if config.get('experiment_type', 'digit_mnist') == 'digit_mnist':
            train_loader_factories = create_digit_mnist_loader_factories(
                root=data_root,
                batch_size=config.get('batch_size', 128),
                train=True, 
                num_workers=config.get('num_workers', 4)
            )
            
            test_loader_factories = create_digit_mnist_loader_factories(
                root=data_root,
                batch_size=config.get('batch_size', 128) * 2,
                train=False,
                num_workers=config.get('num_workers', 4)
            )
            
        elif config.get('experiment_type') == 'letter_notmnist':
            train_loader_factories = create_letter_notmnist_loader_factories(
                root=data_root,
                batch_size=config.get('batch_size', 128),
                train=True, 
                num_workers=config.get('num_workers', 4)
            )
            
            test_loader_factories = create_letter_notmnist_loader_factories(
                root=data_root,
                batch_size=config.get('batch_size', 128) * 2,
                train=False,
                num_workers=config.get('num_workers', 4)
            )
        else:
            raise ValueError(f"Unsupported experiment type: {config.get('experiment_type')}")
        
        classifier = None
        if config.get('use_classifier', True):
            try:
                classifier = get_classifier(config.get('experiment_type', 'digit_mnist'), device)
                print(f"Loaded classifier for {config.get('experiment_type', 'digit_mnist')}")
            except Exception as e:
                print(f"Failed to load classifier: {str(e)}")
        
        record_metric_with_wandb = create_wandb_metric_recorder(config)
        
        model, task_metrics = train_si_vae(
            train_loader_factories=train_loader_factories,
            test_loader_factories=test_loader_factories,
            num_tasks=config.get('num_tasks', 10),
            hidden_size=config.get('hidden_size', 500),
            latent_size=config.get('latent_size', 50),
            input_size=config.get('input_size', 784),
            epochs=config.get('epochs', 20),
            learning_rate=config.get('lr', 1e-3),
            optimizer_type=config.get('optimizer_type', 'sgd'),
            momentum=config.get('momentum', 0.9),
            si_lambda=config.get('si_lambda', 0.1),
            si_epsilon=config.get('si_epsilon', 1e-3),
            omega_decay=config.get('omega_decay', 0.9),
            device=device,
            output_dir=experiment_dir,
            classifier=classifier,
            record_metric_fn=record_metric_with_wandb,
            early_stopping_threshold=config.get('early_stopping_threshold', None)
        )
        
        log_wandb_images(experiment_dir)
        
        torch.save(model.state_dict(), experiment_dir / "checkpoints" / "final_model.pt")
        
        return model, task_metrics
    
    except Exception as e:
        save_error_log(e, config, experiment_dir)
        raise
    
    finally:
        finish_wandb()
        cleanup_resources()

def main():
    config = {
        "experiment_type": "digit_mnist",
        "num_tasks": 10,
        "input_size": 784,
        
        "hidden_size": 500,
        "latent_size": 50,
        
        "epochs": 20,
        "batch_size": 128,
        "lr": 1e-3,
        "optimizer_type": "sgd",
        "momentum": 0.9,
        
        "si_lambda": 0.1,
        "si_epsilon": 1e-3,
        "omega_decay": 0.9,
        
        "num_workers": 4,
        "eval_samples": 100,
        
        "early_stopping_threshold": 0.15
    }
    
    print("\nStarting SI-VAE experiment...")
    print(f"Experiment type: {config['experiment_type']}")
    print(f"Number of tasks: {config['num_tasks']}")
    print(f"Early stopping threshold: {config['early_stopping_threshold']}")
    
    run_si_vae_experiment_with_wandb(
        config, "experiments/si_vae_wandb", "si-vae-project", device='cuda'
    )
    
    print("Experiment completed.")

if __name__ == "__main__":
    from tqdm.auto import tqdm
    torch.tqdm = tqdm
    main()