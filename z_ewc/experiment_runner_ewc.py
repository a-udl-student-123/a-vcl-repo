import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import gc
import pandas as pd
import wandb

from z_data.datasets import generate_permutations, create_permuted_mnist_loader_factories, create_split_mnist_loader_factories, create_split_notmnist_loader_factories
from z_ewc.ewc_models import EWC_MLP, MultiHeadEWC_MLP
from z_ewc.ewc import train_ewc
from z_utils.utils import clean_memory

def init_wandb(config, project_name="ewc-experiments"):
    return wandb.init(
        project=project_name,
        config=config,
        tags=[f"ewc_lambda={config['ewc_lambda']}", f"tasks={config['num_tasks']}"],
    )

def create_wandb_metric_recorder(config, metrics):
    def record_metric_with_wandb(task, epoch, name, value):
        value_to_record = float(value) if isinstance(value, (int, float)) else value
        metrics.append({
            'task': task, 
            'epoch': epoch,
            'metric': name,
            'value': value_to_record
        })
        
        global_step = (task * config['epochs_per_task']) + epoch
        
        if epoch >= 0 and name.startswith('train_'):
            wandb.log({
                f"training/{name}": value_to_record
            }, step=global_step)
        
        if epoch < 0:
            if name == 'average_accuracy':
                wandb.log({
                    "evaluation/average_accuracy": value_to_record
                }, step=(task+1) * config['epochs_per_task'])
            elif name=='average_ce_loss':
                wandb.log({
                    "evaluation/average_ce_loss": value_to_record
                }, step=(task+1) * config['epochs_per_task'])
            elif name.startswith('accuracy_on_task_'):
                task_num = name.split('_')[-1]
                wandb.log({
                    f"evaluation/task_{task_num}/accuracy": value_to_record
                }, step=(task+1) * config['epochs_per_task'])
            
    return record_metric_with_wandb

def run_ewc_experiment(config, experiment_dir, device="cuda", use_wandb=True):
    # runs ewc experiment on permuted/split mnist or notmnist
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
        record_metric_fn = lambda t, e, n, v: metrics.append(
            {"task": t, "epoch": e, "metric": n, "value": float(v)}
        )

    exp_type = config.get("experiment_type", "permuted_mnist").lower()

    permutations = None
    if exp_type == "permuted_mnist":
        permutations = generate_permutations(
            num_tasks=config["num_tasks"], device="cpu"
        )

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
            batch_size=config["batch_size"] * 2,
            train=False,
            num_workers=config["num_workers"],
        )

        def model_factory():
            return EWC_MLP(
                input_size=784,
                hidden_size=config.get("hidden_size", 100),
                output_size=10,
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
            batch_size=config["batch_size"] * 2,
            train=False,
            num_workers=config["num_workers"],
            single_batch=False,
        )

        def model_factory():
            return MultiHeadEWC_MLP(
                input_size=784,
                hidden_layers=config.get("hidden_layers", 2),
                hidden_size=config.get("hidden_size", 256),
                num_tasks=config.get("num_tasks", 5),
                head_size=2,  # binary classification
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
            batch_size=config["batch_size"] * 2,
            train=False,
            num_workers=config["num_workers"],
            single_batch=False,
        )

        def model_factory():
            return MultiHeadEWC_MLP(
                input_size=784,
                hidden_layers=config.get("hidden_layers", 4),
                hidden_size=config.get("hidden_size", 150),
                num_tasks=config.get("num_tasks", 5),
                head_size=2,  
            ).to(device)

    else:
        raise ValueError(f"Unknown experiment_type '{exp_type}'")

    trained_model = train_ewc(
        model_class=model_factory,
        train_loader_factories=train_loader_factories,
        test_loader_factories=test_loader_factories,
        epochs_per_task=config["epochs_per_task"],
        lr=config["lr"],
        ewc_lambda=config["ewc_lambda"],
        device=device,
        record_metric_fn=record_metric_fn,
        exp_dir=exp_dir,
        num_workers=config["num_workers"],
        batch_size=config["batch_size"],
        n_train_samples=config.get("n_train_samples", 600),
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
    # default config 
    config = {
        "num_tasks": 10,
        "epochs_per_task": 40,
        "lr": 0.01,
        "ewc_lambda": 100.0,
        "batch_size": 256,
        "num_workers": 4,
        "hidden_size": 100,
        "n_train_samples": 600  # samples for fisher estimation
    }
    
    run_ewc_experiment(config, "EWC_Permuted_MNIST", device="cuda", use_wandb=True)