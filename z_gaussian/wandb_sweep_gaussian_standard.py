

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

from z_gaussian.experiment_runner_gaussian import run_experiment_gaussian_with_wandb
from z_utils.experiment_utils import is_experiment_completed, create_experiment_name

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
        
        try:
            config = dict(run.config)
            
            set_seed(config.get('seed', 42))
            
            experiment_name = (
                f"lr{config.get('lr', 1e-3)}_"
                f"std{config.get('init_std', 0.001)}_"
                f"coreset{config.get('coreset_size', 0)}_"
                f"kc{config.get('use_kcenter', False)}_"
                f"adapt{config.get('adaptive_std', False)}_"
                f"eps{config.get('adaptive_std_epsilon', 0)}_"
                f"diffperm{config.get('different_perm_init', False)}_"
                f"ep{config.get('epochs', 100)}_"
                f"seed{config.get('seed', 42)}"
            )
            
            experiment_dir = Path(base_dir) / experiment_name
            experiment_dir.mkdir(parents=True, exist_ok=True)
            
            if not force_rerun and is_experiment_completed(experiment_dir, config.get('num_tasks', 10)):
                print(f"Skipping already completed experiment: {experiment_name}")
                wandb.run.summary["skipped"] = True
                wandb.run.summary["skipped_reason"] = "Experiment already completed"
                return
            
            with open(experiment_dir / "config.yaml", "w") as f:
                yaml.dump(config, f)
            
            print(f"Running experiment: {experiment_name}")
            print(f"  Method: {config['method']}")
            print(f"  Learning rate: {config.get('lr', 1e-3)}")
            print(f"  Initial std: {config.get('init_std', 0.001)}")
            print(f"  Adaptive std: {config.get('adaptive_std', False)}")
            print(f"  Epsilon: {config.get('adaptive_std_epsilon', 0)}")
            print(f"  Different perm init: {config.get('different_perm_init', False)}")
            print(f"  Tasks: {config.get('num_tasks', 10)}")
            print(f"  Epochs: {config.get('epochs', 100)}")
            
            run_experiment_gaussian_with_wandb(
                config, 
                experiment_dir,
                project_name=run.project,
                device=device
            )
            
            print(f"Experiment completed. Results saved to {experiment_dir}")
            
        except Exception as e:
            # log error but keep sweep running
            error_message = f"Error in run {run.id}: {str(e)}"
            print(f"\n⚠️ {error_message}")
            
            import traceback
            traceback.print_exc()
            
            wandb.log({"error": error_message})
            wandb.run.summary["failed"] = True
            wandb.run.summary["error_message"] = error_message
            
            # write error details to file
            try:
                experiment_dir = Path(base_dir) / f"failed_run_{run.id}"
                experiment_dir.mkdir(parents=True, exist_ok=True)
                
                error_file = experiment_dir / "error_log.txt"
                with open(error_file, "w") as f:
                    f.write(f"Error in run {run.id}: {str(e)}\n\n")
                    f.write("Traceback:\n")
                    f.write(traceback.format_exc())
                    f.write("\nConfig:\n")
                    for key, val in config.items():
                        f.write(f"{key}: {val}\n")
            except:
                print("Failed to write error log file")
        
        finally:
            print(f"Run {run.id} finished")
            wandb.finish()
    
    return sweep_iteration

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sweep_config_path = os.path.join(os.path.dirname(__file__), "standard_vcl_sweep.yaml")
    output_dir = os.path.join(root_dir, "Gauss_Permuted_MNIST_fixed/StandardVCL")
    project_name = "vcl-gaussian-permuted-mnist-fixed"
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    
    wandb.agent(sweep_id, function=sweep_fn, count=24)

if __name__ == "__main__":
    main() 