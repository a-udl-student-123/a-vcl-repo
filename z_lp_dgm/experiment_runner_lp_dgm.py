

import sys, gc
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch, pandas as pd, wandb
from z_utils.utils import clean_memory
from z_data.datasets_dgm import (
    create_digit_mnist_loader_factories,
    create_letter_notmnist_loader_factories,
)
from z_classifiers.classifier_utils import get_classifier
from z_lp_dgm.lp_dgm import train_lp_vae

def init_wandb(cfg, project="lp-vae-dgm"):
    return wandb.init(project=project, config=cfg,
                     tags=[f"λ={cfg['lp_lambda']}", f"tasks={cfg['num_tasks']}"])

def _wandb_rec(cfg, buf):
    ep_per = cfg["epochs"]
    def rec(task, ep, name, val):
        buf.append(dict(task=task, epoch=ep, metric=name, value=float(val)))
        if wandb.run is None:
            return
        step = task*ep_per + max(ep, 0)
        wandb.log({name: val}, step=step)
    return rec

def run_lp_vae_experiment(cfg, out_dir, device="cuda", use_wandb=True):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if use_wandb:
        init_wandb(cfg)

    if cfg["experiment_type"] == "digit_mnist":
        train_fact = create_digit_mnist_loader_factories(
            "data/", cfg["batch_size"], True, cfg["num_workers"])
        test_fact = create_digit_mnist_loader_factories(
            "data/", cfg["batch_size"]*2, False, cfg["num_workers"])
    elif cfg["experiment_type"] == "letter_notmnist":
        train_fact = create_letter_notmnist_loader_factories(
            "data/", cfg["batch_size"], True, cfg["num_workers"])
        test_fact = create_letter_notmnist_loader_factories(
            "data/", cfg["batch_size"]*2, False, cfg["num_workers"])
    else:
        raise ValueError("unknown experiment_type")

    classifier = None
    if cfg.get('use_classifier', True):
        try:
            classifier = get_classifier(cfg.get('experiment_type', 'digit_mnist'), device)
            print(f"loaded classifier for {cfg.get('experiment_type', 'digit_mnist')}")
        except Exception as e:
            print(f"failed to load classifier: {str(e)}")

    metrics = []
    rec = _wandb_rec(cfg, metrics) if use_wandb else (
        lambda t,e,n,v: metrics.append(dict(task=t, epoch=e, metric=n, value=v)))

    train_lp_vae(
        train_loader_factories=train_fact,
        test_loader_factories=test_fact,
        num_tasks=cfg["num_tasks"],
        hidden_size=cfg["hidden_size"],
        latent_size=cfg["latent_size"],
        input_size=cfg["input_size"],
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        lp_lambda=cfg["lp_lambda"],
        n_hessian_samples=cfg["n_train_samples"],
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
        num_tasks = 10,
        epochs = 20,
        batch_size = 128,
        lr = 1e-3,
        lp_lambda = 10.0,
        n_train_samples = 100,
        hidden_size = 500,
        latent_size = 50,
        input_size = 784,
        num_workers = 4,
        use_classifier = True,
    )
    run_lp_vae_experiment(cfg, "experiments/lp_vae_example", device="cuda")
