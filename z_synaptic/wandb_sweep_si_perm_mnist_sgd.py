

import yaml
import wandb
import torch
import numpy as np
import random
from pathlib import Path
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="A newer version of deeplake.*", category=UserWarning, module="deeplake.util.check_latest_version")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from z_synaptic.experiment_runner import run_si_experiment_with_wandb

OPTIMIZER_TYPE = "sgd"
OUTPUT_DIR = "SI_PermutedMNIST/SGD"
FORCE_RERUN = True
COUNT = 150
PROJECT_NAME = "SI-permuted-mnist-fixed-si"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_experiment_completed(exp_dir):
    metrics_file = exp_dir / "metrics.csv"
    return metrics_file.exists()

def create_sweep_function(base_dir, device, force_rerun=False):
    
    def sweep_iteration():
        run = wandb.init(resume=False)
        
        config = dict(run.config)
        
        set_seed(config.get('seed', 42))
        
        experiment_name = (
            f"sgd_"
            f"ep{config.get('epochs', 10)}_"
            f"lr{config.get('lr', 0.1)}_"
            f"lambda{config.get('si_lambda', 0.5)}_"
            f"eps{config.get('si_epsilon', 1e-4)}_"
            f"decay{config.get('omega_decay', 0.9)}_"
            f"seed{config.get('seed', 42)}"
        )
        
        if 'early_stopping_threshold' in config:
            experiment_name += f"_early{config['early_stopping_threshold']}"
        
        experiment_dir = Path(base_dir) / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        if not force_rerun and is_experiment_completed(experiment_dir):
            print(f"Skipping already completed experiment: {experiment_name}")
            wandb.finish()
            return
        
        with open(experiment_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        
        print(f"Running experiment: {experiment_name}")
        print(f"  Optimizer: SGD")
        print(f"  Learning rate: {config.get('lr', 0.1)}")
        print(f"  SI lambda: {config.get('si_lambda', 0.5)}")
        print(f"  SI epsilon: {config.get('si_epsilon', 1e-4)}")
        print(f"  Omega decay: {config.get('omega_decay', 0.9)}")
        print(f"  Tasks: {config.get('num_tasks', 10)}")
        print(f"  Epochs: {config.get('epochs', 10)}")
        
        run_si_experiment_with_wandb(
            config, 
            experiment_dir,
            project_name=run.project,
            device=device
        )
        
        print(f"Experiment completed. Results saved to {experiment_dir}")
    
    return sweep_iteration

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sweep_config_path = os.path.join(root_dir, f"z_synaptic/sweep_{OPTIMIZER_TYPE}.yaml")
    output_dir = os.path.join(root_dir, OUTPUT_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Starting SGD sweep with {COUNT} iterations")
    print(f"Output directory: {output_dir}")
    print(f"Force rerun: {FORCE_RERUN}")
    
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    print(f"Created sweep with ID: {sweep_id}")
    
    sweep_fn = create_sweep_function(
        base_dir=output_dir,
        device=device,
        force_rerun=FORCE_RERUN
    )
    
    wandb.agent(sweep_id, function=sweep_fn)

if __name__ == "__main__":
    main()