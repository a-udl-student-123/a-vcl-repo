import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import gc
import pandas as pd
import wandb

from z_data.datasets import (
    generate_permutations, create_permuted_mnist_loader_factories,
    create_split_mnist_loader_factories, create_split_notmnist_loader_factories
)
from z_models.vcl_models import VCL_MLP, VCL_FlexibleMultiHead_MLP
from z_active_coreset.vcl_active_coreset import train_vcl_active
from z_utils.utils import clean_memory

def prepare_task(model, task_idx, is_prediction_model=False):
    # set task and init new head if needed
    if hasattr(model, 'set_current_task'):
        model.set_current_task(task_idx)
        
    if not is_prediction_model and task_idx > 0 and hasattr(model, 'initialize_new_head'):
        model.initialize_new_head(task_idx)
        
    return model

def init_wandb(config, project_name="vcl-active-coreset"):
    return wandb.init(project=project_name, config=config,
        tags=[f"lambda_mix={config['lambda_mix']}", 
              f"coreset_size={config['coreset_size']}", 
              f"tasks={config['num_tasks']}"])

def create_wandb_metric_recorder(config, metrics):
    def record_metric_with_wandb(task, epoch, name, value):
        value_to_record = float(value) if isinstance(value, (int, float)) else value
        metrics.append({
            'task': task,
            'epoch': epoch,
            'metric': name,
            'value': value_to_record
        })
        
        global_step = task*config['epochs_per_task'] + epoch
        
        if epoch >= 0:
            wandb.log({f"train/{name}": value_to_record}, step=global_step)
        
        if epoch < 0:
            if name == 'average_accuracy':
                wandb.log({"eval/average_accuracy": value_to_record}, 
                    step=(task+1) * config['epochs_per_task'])
            elif name=='average_ce_loss':
                wandb.log({"eval/average_ce_loss": value_to_record}, 
                    step=(task+1)*config['epochs_per_task'])
            elif name.startswith('accuracy_on_task_'):
                task_num = name.split('_')[-1]
                wandb.log({f"eval/task_{task_num}/accuracy": value_to_record}, 
                    step=(task+1)*config['epochs_per_task'])
            else:
                wandb.log({f"eval/{name}": value_to_record}, 
                    step=(task+1)*config['epochs_per_task'])  
            
    return record_metric_with_wandb

def run_vcl_active_experiment(config, experiment_dir, device="cuda", use_wandb=True):
    # runs vcl experiment w/ active coreset selection
    exp_dir = Path(experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)

    if use_wandb:
        init_wandb(config)

    metrics = []
    if use_wandb:
        record_metric_fn = create_wandb_metric_recorder(config, metrics)
    else:
        record_metric_fn = lambda t,e,n,v: metrics.append(
            {"task": t, "epoch": e, "metric": n, "value": float(v)}
        )

    exp_type = config.get("experiment_type", "permuted_mnist").lower()
    permutations = None

    # TODO

    if exp_type == "permuted_mnist":
        permutations = generate_permutations(config["num_tasks"], device="cpu")

        train_loader_factories = create_permuted_mnist_loader_factories(
            root="data/",
            permutations=permutations,
            batch_size=config["batch_size"],
            train=True,
            num_workers=config["num_workers"],
        )
        test_loader_factories = create_permuted_mnist_loader_factories(
            root="data/",
            permutations=permutations,
            batch_size=config["batch_size"]*2,
            train=False,
            num_workers=config["num_workers"],
        )

        def model_factory():
            return VCL_MLP(
                input_size=784,
                hidden_size=config.get("hidden_size", 100),
                output_size=10
            ).to(device)

    elif exp_type == "split_mnist":
        train_loader_factories = create_split_mnist_loader_factories(
            root="data/",
            batch_size=config["batch_size"],
            train=True,
            num_workers=config["num_workers"],
            single_batch=config.get("single_batch", False),
        )
        test_loader_factories = create_split_mnist_loader_factories(
            root="data/",
            batch_size=config["batch_size"]*2,
            train=False,
            num_workers=config["num_workers"],
            single_batch=False,
        )

        def model_factory():
            return VCL_FlexibleMultiHead_MLP(
                input_size=784,
                hidden_sizes=[256,256],
                num_tasks=5,
                head_size=2,
                init_std=config.get("init_std", 0.001)
            ).to(device)

    elif exp_type == "split_notmnist":
        train_loader_factories = create_split_notmnist_loader_factories(
            root="data/", 
            batch_size=config["batch_size"],
            train=True,
            num_workers=config["num_workers"],
            single_batch=config.get("single_batch", False),
        )
        test_loader_factories = create_split_notmnist_loader_factories(
            root="data/",
            batch_size=config["batch_size"]*2,
            train=False,
            num_workers=config["num_workers"],
            single_batch=False,
        )

        def model_factory():
            return VCL_FlexibleMultiHead_MLP(
                input_size=784,
                hidden_sizes=[150,150,150,150],
                num_tasks=5,
                head_size=2,
                init_std=config.get("init_std", 0.001)
            ).to(device)
    else:
        raise ValueError(f"Unknown experiment_type '{exp_type}'")

    trained_model = train_vcl_active(
        model_class=model_factory,
        train_loader_factories=train_loader_factories,
        test_loader_factories=test_loader_factories,
        coreset_size=config["coreset_size"],
        lambda_mix=config["lambda_mix"],
        use_kcenter=config.get("use_kcenter", False),
        kcenter_batch_size=config.get("kcenter_batch_size", 1024),
        epochs_per_task=config["epochs_per_task"],
        pred_epochs_multiplier=config.get("pred_epochs_multiplier", 1.0),
        lr=config["lr"],
        device=device,
        record_metric_fn=record_metric_fn,
        exp_dir=exp_dir,
        num_workers=config["num_workers"],
        use_ml_initialization=config.get("use_ml_initialization", False),
        ml_epochs=config.get("ml_epochs", 5),
        n_train_samples=config.get("n_train_samples", 5),
        n_eval_samples=config.get("n_eval_samples", 100),
        init_std=config.get("init_std", 0.001),
        adaptive_std=config.get("adaptive_std", False),
        adaptive_std_epsilon=config.get("adaptive_std_epsilon", 0.01),
        different_perm_init=config.get("different_perm_init", False),
        use_task_specific_prediction=config.get("use_task_specific_prediction", False),
        pre_task_hook=prepare_task if config.get("use_task_specific_prediction", False) else None,
        early_stopping_threshold=config.get("early_stopping_threshold", None)
    )

    pd.DataFrame(metrics).to_csv(exp_dir / "metrics.csv", index=False)

    if permutations is not None:
        del permutations
    del train_loader_factories, test_loader_factories
    clean_memory(device)

    if use_wandb and wandb.run is not None:
        wandb.finish()

    return metrics

if __name__ == "__main__":
    config = {
        "num_tasks": 10,
        "epochs_per_task": 100,
        "lr": 0.001,
        "coreset_size": 1000,
        "lambda_mix": 0.5,
        "use_kcenter": True,
        "batch_size": 256,
        "num_workers": 6,
        "hidden_size": 100,
        "n_train_samples": 1,
        "n_eval_samples": 100,
        "use_task_specific_prediction": False,
        "experiment_type": "permuted_mnist",
        "ml_epochs": 100,
        "use_ml_initialization": True,
        "init_std": 0.000001,
        "adaptive_std": False,
        "adaptive_std_epsilon": 0.0,
        "different_perm_init": True,
    }
    
    run_vcl_active_experiment(config, "VCL_Active_Permuted_MNIST", device="cuda", use_wandb=True)