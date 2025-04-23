
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

from z_lp.experiment_runner_lp import run_lp_experiment
from z_utils.experiment_utils import is_experiment_completed

OUTPUT_DIR = "LP_SplitNotMNIST"
FORCE_RERUN = False
COUNT = 48  
PROJECT_NAME = "LP-split-notmnist"

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
        # Initialize wandb
        run = wandb.init(resume=False)
        
        config = dict(run.config)
        
        set_seed(config.get('seed', 42))
        
        experiment_name = (
            f"lp_split_notmnist_"
            f"h{config.get('hidden_size', 150)}_"
            f"ep{config.get('epochs_per_task', 20)}_"
            f"lr{config.get('lr', 0.01)}_"
            f"lambda{config.get('lp_lambda', 100.0)}_"
            f"samples{config.get('n_train_samples', 2000)}_"
            f"seed{config.get('seed', 42)}"
        )
        
        experiment_dir = Path(base_dir) / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        if not force_rerun and is_experiment_completed(experiment_dir):
            print(f"Skipping already completed experiment: {experiment_name}")
            wandb.finish()
            return
        
        with open(experiment_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        
        print(f"Running Split notMNIST experiment: {experiment_name}")
        print(f"  Hidden size: {config.get('hidden_size', 150)}")
        print(f"  Learning rate: {config.get('lr', 0.01)}")
        print(f"  LP lambda: {config.get('lp_lambda', 100.0)}")
        print(f"  Hessian samples: {config.get('n_train_samples', 2000)}")
        print(f"  Epochs per task: {config.get('epochs_per_task', 20)}")
        
        run_lp_experiment(
            config, 
            experiment_dir,
            device=device,
            use_wandb=True
        )
        
        print(f"Experiment completed. Results saved to {experiment_dir}")
    
    return sweep_iteration

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sweep_config_path = os.path.join(root_dir, "z_lp/sweep_lp_split_notmnist.yaml")
    output_dir = os.path.join(root_dir, OUTPUT_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Starting LP Split notMNIST sweep with {COUNT} iterations")
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
    
    wandb.agent(sweep_id, function=sweep_fn, count=COUNT)

if __name__ == "__main__":
    main()