

import yaml
import wandb
import torch
import numpy as np
import random
from pathlib import Path
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg') 

warnings.filterwarnings("ignore", message="A newer version of deeplake.*", category=UserWarning, module="deeplake.util.check_latest_version")



sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiment_runner_dgm import run_experiment_with_wandb
from z_utils.experiment_utils import is_experiment_completed

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_sweep_function(base_dir, device, force_rerun=False):
    """Create a sweep function for notMNIST DGM experiments."""
    
    def sweep_iteration():
        """Run a single iteration of the sweep."""

        run = wandb.init(resume=False)
        
 
        config = dict(run.config)
        

        set_seed(config.get('seed', 42))
        

        experiment_name = (
            f"lr{config.get('lr', 1e-3)}_"
            f"std{config.get('init_std', 0.001)}_"
            f"bs{config.get('batch_size', 256)}_"
            f"ep{config.get('epochs', 200)}_"
            f"seed{config.get('seed', 42)}_"
            f"epochs{config.get('epochs', 400)}"
        )


        experiment_dir = Path(base_dir) / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        

        if not force_rerun and is_experiment_completed(experiment_dir, config.get('num_tasks', 10)):
            print(f"Skipping already completed experiment: {experiment_name}")
            return
        
  
        with open(experiment_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        
        print(f"Running experiment: {experiment_name}")
        print(f"  Method: {config['method']}")
        print(f"  Learning rate: {config.get('lr', 1e-3)}")
        print(f"  Initial std: {config.get('init_std', 0.001)}")
        print(f"  Batch size: {config.get('batch_size', 256)}")
        print(f"  Tasks: {config.get('num_tasks', 10)}")
        print(f"  Epochs: {config.get('epochs', 200)}")
        

        run_experiment_with_wandb(
            config, 
            experiment_dir,
            project_name=run.project,
            device=device
        )
        
        print(f"Experiment completed. Results saved to {experiment_dir}")
    
    return sweep_iteration

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sweep_config_path = os.path.join(root_dir, "z_sweep_configs/dgm_notmnist/random_sweep.yaml")
    output_dir = os.path.join(root_dir, "DGM_notMNIST/StandardVCL")
    project_name = "vcl-dgm-notmnist"
    device = "cpu"
    force_rerun = False

    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Created sweep with ID: {sweep_id}")
    
    
    sweep_fn = create_sweep_function(
        base_dir=output_dir,
        device=device,
        force_rerun=force_rerun
    )
    wandb.agent(sweep_id, function=sweep_fn, count=18)

if __name__ == "__main__":
    main() 