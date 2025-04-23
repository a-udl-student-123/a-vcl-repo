

import sys, gc
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pandas as pd
import wandb
from z_utils.utils import clean_memory
from z_data.datasets_dgm import (
    create_digit_mnist_loader_factories,
    create_letter_notmnist_loader_factories,
)
from z_classifiers.classifier_utils import get_classifier

from z_ewc_dgm.ewc_dgm import train_ewc_vae

def init_wandb(cfg, project="ewc-vae-dgm"):
    return wandb.init(
        project=project,
        config=cfg,
        tags=[f"λ={cfg['ewc_lambda']}", f"tasks={cfg['num_tasks']}"],
    )

def _wandb_rec(cfg, buf):
    ep_per = cfg["epochs"]
    def rec(task, ep, name, val):
        buf.append(dict(task=task, epoch=ep, metric=name, value=float(val)))
        if wandb.run is None:
            return
        
        # prefix eval metrics with eval/, train with train/
        if ep == -1:
            prefixed_name = f"eval/{name}"
        else:
            prefixed_name = f"train/{name}"
            
        step = task*ep_per + max(ep, 0)
        wandb.log({prefixed_name: val}, step=step)
    return rec

def run_ewc_vae_experiment(cfg, out_dir, device="cuda", use_wandb=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if use_wandb:
        init_wandb(cfg)

    if cfg["experiment_type"] == "digit_mnist":
        train_fact = create_digit_mnist_loader_factories(
            "data/", cfg["batch_size"], True, cfg["num_workers"]
        )
        test_fact = create_digit_mnist_loader_factories(
            "data/", cfg["batch_size"]*2, False, cfg["num_workers"]
        )
    elif cfg["experiment_type"] == "letter_notmnist":
        train_fact = create_letter_notmnist_loader_factories(
            "data/", cfg["batch_size"], True, cfg["num_workers"]
        )
        test_fact = create_letter_notmnist_loader_factories(
            "data/", cfg["batch_size"]*2, False, cfg["num_workers"]
        )
    else:
        raise ValueError(f"unknown experiment_type {cfg['experiment_type']}")

    classifier = None
    if cfg.get('use_classifier', True):
        try:
            classifier = get_classifier(
                cfg["experiment_type"], device
            )
            print(f"Loaded classifier for {cfg['experiment_type']}")
        except Exception as e:
            print(f"Failed to load classifier: {e}")

    metrics = []
    if use_wandb:
        rec = _wandb_rec(cfg, metrics)
    else:
        rec = lambda t,e,n,v: metrics.append(
            {"task": t, "epoch": e, "metric": n, "value": float(v)}
        )

    train_ewc_vae(
        train_loader_factories=train_fact,
        test_loader_factories=test_fact,
        num_tasks=cfg["num_tasks"],
        hidden_size=cfg["hidden_size"],
        latent_size=cfg["latent_size"],
        input_size=cfg["input_size"],
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        ewc_lambda=cfg["ewc_lambda"],
        n_fisher_samples=cfg["n_train_samples"],
        device=device,
        output_dir=out,
        classifier=classifier,
        record_metric_fn=rec
    )

    pd.DataFrame(metrics).to_csv(out/"metrics.csv", index=False)
    clean_memory(device)
    if use_wandb and wandb.run is not None:
        wandb.finish()
    return metrics

if __name__ == "__main__":
    cfg = dict(
        experiment_type = "digit_mnist",
        num_tasks       = 3,
        epochs          = 20,
        batch_size      = 128,
        lr              = 1e-3,
        ewc_lambda      = 10.0,
        n_train_samples = 100,
        hidden_size     = 500,
        latent_size     = 50,
        input_size      = 784,
        num_workers     = 4,
        use_classifier  = True,
    )
    run_ewc_vae_experiment(
        cfg,
        "experiments/ewc_vae_example", 
        device="cuda",
        use_wandb=True
    )
