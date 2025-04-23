

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

from z_ewc_dgm.experiment_runner_ewc_dgm import run_ewc_vae_experiment
from z_utils.experiment_utils import is_experiment_completed

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_sweep_function(base_dir, device, force_rerun=False):
    def sweep_iteration():
        run = wandb.init(resume=False)
        config = dict(run.config)
        
        set_seed(config.get('seed', 42))
        
        experiment_name = (
            f"ewc_notmnist_"
            f"ep{config.get('epochs', 100)}_"
            f"lr{config.get('lr', 0.01)}_"
            f"lambda{config.get('ewc_lambda', 1.0)}_"
            f"samples{config.get('n_fisher_samples', 600)}_"
            f"mom{config.get('momentum', 0.9)}_"
            f"opt{config.get('optimizer_type', 'sgd')}_"
            f"tasks{config.get('num_tasks', 10)}"
        )
        
        experiment_dir = Path(base_dir) / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        if not force_rerun and is_experiment_completed(experiment_dir, config.get('num_tasks', 10)):
            print(f"Skipping already completed experiment: {experiment_name}")
            return
        
        with open(experiment_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        
        print(f"Running experiment: {experiment_name}")
        print(f"  Experiment type: {config['experiment_type']}")
        print(f"  Learning rate: {config.get('lr', 0.01)}")
        print(f"  EWC lambda: {config.get('ewc_lambda', 1.0)}")
        print(f"  Fisher samples: {config.get('n_fisher_samples', 600)}")
        print(f"  Tasks: {config.get('num_tasks', 10)}")
        print(f"  Epochs: {config.get('epochs', 100)}")
        
        run_ewc_vae_experiment(
            config, 
            experiment_dir,
            device=device,
            use_wandb=True
        )
        
        print(f"Experiment completed. Results saved to {experiment_dir}")
    
    return sweep_iteration

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    sweep_config_path = os.path.join(root_dir, "sweep_dgm_ewc_notmnist.yaml")
    output_dir = os.path.join(root_dir, "..", "EWC_DGM_notMNIST")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    force_rerun = False  
    
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    sweep_id = wandb.sweep(sweep_config, project="EWC-dgm-notmnist")
    print(f"Created sweep with ID: {sweep_id}")
    
    sweep_fn = create_sweep_function(
        base_dir=output_dir,
        device=device,
        force_rerun=force_rerun
    )
    
    wandb.agent(sweep_id, function=sweep_fn, count=4)

if __name__ == "__main__":
    main() 