# sweep launcher for permuted mnist vcl w/ active coreset selection

import yaml
import wandb
import torch
import numpy as np
import random
from pathlib import Path
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from z_active_coreset.experiment_runner_active_coreset import run_vcl_active_experiment
from z_utils.experiment_utils import is_experiment_completed


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_sweep_function(base_dir, device, force_rerun=False):
    """creates sweep fn for permuted mnist active coreset"""
    
    def sweep_iteration():
        run = wandb.init(resume=False)
        config = dict(run.config)
        set_seed(config.get('seed', 42))
        
        experiment_name = (
            f"active_perm_"
            f"lr{config.get('lr', 0.001)}_"
            f"lambda{config.get('lambda_mix', 0.5)}_"
            f"coreset{config.get('coreset_size', 500)}_"
            f"kc{config.get('use_kcenter', True)}_"
            f"ep{config.get('epochs_per_task', 100)}_"
            f"init{config.get('init_std', 0.001)}_"
            f"diff_perm{config.get('different_perm_init', False)}_"
            f"seed{config.get('seed', 42)}"
        )
        
        experiment_dir = Path(base_dir) / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)  
        
        if not force_rerun and is_experiment_completed(experiment_dir, config.get('num_tasks', 10)):
            print(f"Skipping already completed experiment: {experiment_name}")
            return
        
        with open(experiment_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        
        # print experiment details
        print(f"Running experiment: {experiment_name}")
        print(f"  Experiment type: {config['experiment_type']}")
        print(f"  Learning rate: {config.get('lr', 0.001)}")
        print(f"  Lambda mix: {config.get('lambda_mix', 0.5)}")
        print(f"  Coreset size: {config.get('coreset_size', 500)}")
        print(f"  Use K-center: {config.get('use_kcenter', True)}")
        print(f"  Tasks: {config.get('num_tasks', 10)}")
        print(f"  Epochs per task: {config.get('epochs_per_task', 100)}")

        run_vcl_active_experiment(config, experiment_dir, device=device, use_wandb=True)
        print(f"Experiment completed. Results saved to {experiment_dir}")
    
    return sweep_iteration


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sweep_config_path=os.path.join(root_dir, "z_active_coreset/active_coreset_perm.yaml")
    output_dir = os.path.join(root_dir, "Active_Coreset_Permuted_MNIST")
    project_name = "vcl-active-coreset-permuted"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    force_rerun = False

    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Created sweep with ID: {sweep_id}")

    sweep_fn = create_sweep_function(base_dir=output_dir, device=device, force_rerun=force_rerun)
    wandb.agent(sweep_id, function=sweep_fn, count=36)


if __name__ == "__main__":
    main() 