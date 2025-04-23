
import sys, gc
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch, pandas as pd, wandb
from z_utils.utils import clean_memory

from z_data.datasets import (
    generate_permutations,
    create_permuted_mnist_loader_factories,
    create_split_mnist_loader_factories,
    create_split_notmnist_loader_factories,
)

from z_lp.lp import train_lp
from z_lp.lp_models import LP_MLP, MultiHeadLP_MLP


def init_wandb(cfg, project="lp-experiments"):
    return wandb.init(project=project, config=cfg,
                      tags=[f"lp_lambda={cfg['lp_lambda']}",
                            f"tasks={cfg['num_tasks']}"])


def make_wandb_recorder(cfg, buf):
    def rec(task, ep, name, val):
        v = float(val) if isinstance(val, (int, float)) else val
        buf.append({'task': task, 'epoch': ep, 'metric': name, 'value': v})
        if wandb.run is None:  
            return
        step = task*cfg['epochs_per_task'] + ep
        if ep >= 0 and name.startswith("train_"):
            wandb.log({f"training/{name}": v}, step=step)
        if ep < 0:
            if name == "average_accuracy":
                wandb.log({"evaluation/average_accuracy": v},
                          step=(task+1)*cfg['epochs_per_task'])
            elif name == "average_ce_loss":
                wandb.log({"evaluation/average_ce_loss": v},
                          step=(task+1)*cfg['epochs_per_task'])
            elif name.startswith("accuracy_on_task_"):
                wandb.log({f"evaluation/{name}": v},
                          step=(task+1)*cfg['epochs_per_task'])
    return rec


def run_lp_experiment(cfg, out_dir, device="cuda", use_wandb=True):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out/"plots").mkdir(exist_ok=True); (out/"checkpoints").mkdir(exist_ok=True)

    if use_wandb: init_wandb(cfg)

    metrics = []
    rec = make_wandb_recorder(cfg, metrics) if use_wandb else (lambda t,e,n,v: metrics.append(
                            {'task': t,'epoch': e,'metric': n,'value': float(v)}))

    exp = cfg.get("experiment_type", "permuted_mnist").lower()

    permutations = None
    if exp == "permuted_mnist":
        permutations = generate_permutations(cfg["num_tasks"], device="cpu")
        train_fact = create_permuted_mnist_loader_factories(
            "data/", permutations, cfg["batch_size"], True, cfg["num_workers"])
        test_fact  = create_permuted_mnist_loader_factories(
            "data/", permutations, cfg["batch_size"]*2, False, cfg["num_workers"])
        model_fac  = lambda: LP_MLP(784, cfg.get("hidden_size", 100), 10).to(device)

    elif exp == "split_mnist":
        train_fact = create_split_mnist_loader_factories(
            "data/", cfg["batch_size"], True, cfg["num_workers"],
            single_batch=cfg.get("single_batch", False))
        test_fact  = create_split_mnist_loader_factories(
            "data/", cfg["batch_size"]*2, False, cfg["num_workers"], single_batch=False)
        model_fac  = lambda: MultiHeadLP_MLP(
            784, cfg.get("hidden_layers", 2), cfg.get("hidden_size", 256),
            cfg["num_tasks"], 2).to(device)

    elif exp == "split_notmnist":
        train_fact = create_split_notmnist_loader_factories(
            "data/", cfg["batch_size"], True, cfg["num_workers"],
            single_batch=cfg.get("single_batch", False))
        test_fact  = create_split_notmnist_loader_factories(
            "data/", cfg["batch_size"]*2, False, cfg["num_workers"], single_batch=False)
        model_fac  = lambda: MultiHeadLP_MLP(
            784, cfg.get("hidden_layers", 4), cfg.get("hidden_size", 150),
            cfg["num_tasks"], 2).to(device)
    else:
        raise ValueError(f"unknown experiment_type {exp}")

    train_lp(
        model_class=model_fac,
        train_loader_factories=train_fact,
        test_loader_factories=test_fact,
        epochs_per_task=cfg["epochs_per_task"],
        lr=cfg["lr"],
        lp_lambda=cfg["lp_lambda"],
        device=device,
        record_metric_fn=rec,
        exp_dir=out,
        num_workers=cfg["num_workers"],
        batch_size=cfg["batch_size"],
        n_train_samples=cfg.get("n_train_samples", 600),
    )

    pd.DataFrame(metrics).to_csv(out/"metrics.csv", index=False)
    clean_memory(device)
    if use_wandb and wandb.run is not None:
        wandb.finish()
    return metrics


if __name__ == "__main__":
    cfg = {
        "experiment_type": "split_mnist",
        "num_tasks": 10,
        "epochs_per_task": 20,
        "lr": 0.01,
        "lp_lambda": 1.0,
        "batch_size": 256,
        "num_workers": 4,
        "hidden_size": 100,
        "n_train_samples": 600,
    }
    run_lp_experiment(cfg, "LP_Permuted_MNIST", device="cuda", use_wandb=True)
