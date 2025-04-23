

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

from z_synaptic.multi_head_experiment_runner import run_multi_head_si_experiment_with_wandb
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
            f"lr{config.get('lr', 1e-3)}_"
            f"si{config.get('si_lambda', 0.5)}_"
            f"eps{config.get('si_epsilon', 1e-3)}_"
            f"decay{config.get('omega_decay', 0.9)}_"
            f"ep{config.get('epochs', 15)}_"
            f"seed{config.get('seed', 42)}"
        )
        
        experiment_dir = Path(base_dir) / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        if not force_rerun and is_experiment_completed(experiment_dir, config.get('num_tasks', 5)):
            print(f"Skipping already completed experiment: {experiment_name}")
            return
        
        with open(experiment_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        
        print(f"Running experiment: {experiment_name}")
        print(f"  Experiment type: {config['experiment_type']}")
        print(f"  Learning rate: {config.get('lr', 1e-3)}")
        print(f"  SI lambda: {config.get('si_lambda', 0.5)}")
        print(f"  SI epsilon: {config.get('si_epsilon', 1e-3)}")
        print(f"  Omega decay: {config.get('omega_decay', 0.9)}")
        print(f"  Tasks: {config.get('num_tasks', 5)}")
        print(f"  Epochs: {config.get('epochs', 15)}")
        
        run_multi_head_si_experiment_with_wandb(
            config, 
            experiment_dir,
            project_name="SI-split-mnist-sweep",
            device=device
        )
        
        print(f"Experiment completed. Results saved to {experiment_dir}")
    
    return sweep_iteration

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    sweep_config_path = os.path.join(root_dir, "..", "z_synaptic/sweep_split_mnist.yaml")
    output_dir = os.path.join(root_dir, "..", "Split_MNIST/SI")
    project_name = "SI-split-mnist-sweep-fixed-si"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    force_rerun = True
    
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Created sweep with ID: {sweep_id}")
    
    sweep_fn = create_sweep_function(
        base_dir=output_dir,
        device=device,
        force_rerun=force_rerun
    )
    
    wandb.agent(sweep_id, function=sweep_fn)

if __name__ == "__main__":
    main()