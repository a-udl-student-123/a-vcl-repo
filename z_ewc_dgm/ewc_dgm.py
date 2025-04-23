"""
Training for Elastic Weight Consolidation on deep generative models
(MNIST digit-by-digit or notMNIST letter-by-letter).

same visualizations and metrics as Laplace-Propagation impl, just 
different loss/fisher penalty and registration
"""

import os
import time
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import wandb

from z_utils.utils import clean_memory, clean_loader
from z_utils.utils_dgm import evaluate_model_dgm
from .ewc_vae import EWC_DGM_VAE 

def _train_epoch(model, loader, optimiser, ewc_lambda, device):
    model.train().to(device)
    data_size = len(loader.dataset)
    totals = dict(total_loss=0., recon_loss=0., kl_latent=0., ewc_penalty=0.)
    seen = 0

    for xb, _ in tqdm(loader, leave=False, desc="batches"):
        xb = xb.to(device)
        optimiser.zero_grad(set_to_none=True)
        loss, m = model.compute_loss(xb, ewc_lambda, data_size)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()

        bs = xb.size(0)
        seen += bs
        for k in ("total_loss", "recon_loss", "kl_latent", "ewc_penalty"):
            totals[k]+= m.get(k, 0.) * bs

    display_stats = {
        "loss": totals["total_loss"] / seen,
        "recon": totals["recon_loss"] / seen,
        "kl": totals["kl_latent"] / seen,
        "ewc": totals["ewc_penalty"] / seen
    }
    return display_stats

def _train_task(model, loader, task_idx, epochs, lr, ewc_lambda, device,
                optimiser_type="adam", momentum=0.9, record=None):
    model.set_task(task_idx)
    if optimiser_type.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    ep_bar = tqdm(range(1, epochs + 1), desc="epochs", leave=False)
    for ep in ep_bar:
        stats = _train_epoch(model, loader, opt, ewc_lambda, device)
        ep_bar.set_postfix({k: f"{v:.3f}" for k, v in stats.items()})
        if record:
            for k, v in stats.items():
                record(task_idx, ep, k, v)
    return stats

def _create_montage(model, out_dir, task_idx, montage_idx=1):
    out_dir.mkdir(exist_ok=True, parents=True)
    samples = []
    for t in range(task_idx + 1):
        model.set_task(t)
        with torch.no_grad():
            samples.append(model.sample(1, task_idx=t)[0].cpu())

    n = len(samples)
    fig, axs = plt.subplots(1, n, figsize=(n * 2, 2))
    if n == 1:
        axs = [axs]

    for i, im in enumerate(samples):
        axs[i].imshow(im.view(28, 28), cmap="gray")
        axs[i].set_title(f"T{i+1}")
        axs[i].axis("off")

    fig.tight_layout()
    p = out_dir / f"task{task_idx+1}_montage{montage_idx}.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p

def _visualise_reconstructions(model, loader, out_dir, task_idx, k=5):
    out_dir.mkdir(exist_ok=True, parents=True)
    model.eval()
    xb, _ = next(iter(loader))
    xb = xb[:k].to(next(model.parameters()).device)
    with torch.no_grad():
        recon = model.sample(k, task_idx=task_idx)

    fig, axs = plt.subplots(2, k, figsize=(k * 2, 4))
    for i in range(k):
        axs[0, i].imshow(xb[i].cpu().view(28, 28), cmap="gray")
        axs[0, i].axis("off")
        axs[1, i].imshow(recon[i].cpu().view(28, 28), cmap="gray")
        axs[1, i].axis("off")
    fig.tight_layout()
    p = out_dir / f"task{task_idx+1}_reconstructions.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p

def _print_eval(task_idx, results):
    print(f"\n=== eval after task {task_idx+1} ===")
    has_classifier = any('classifier_uncertainty' in r for r in results)

    for i, r in enumerate(results[:-1]):
        output = (
            f"  Task {i+1} | recon {r['recon_error']:.4f} "
            f"| LL {r['log_likelihood']:.4f}"
        )
        if has_classifier:
            output += f" | CU {r.get('classifier_uncertainty', 0.0):.4f}"
        print(output)

    avg = results[-1]
    out = (
        f"  Avg  | recon {avg.get('average_recon_error', 0.0):.4f} "
        f"| LL {avg.get('average_log_likelihood', 0.0):.4f}"
    )
    if has_classifier:
        out += f" | CU {avg.get('average_classifier_uncertainty', 0.0):.4f}"
    print(out)
    print("-" * 60)

def train_ewc_vae(
    train_loader_factories, test_loader_factories,
    num_tasks=10,
    hidden_size=500,
    latent_size=50,
    input_size=784,
    epochs=20,
    lr=1e-3,
    optimiser_type="adam",
    momentum=0.9,
    ewc_lambda=10.0,
    n_fisher_samples=600,
    device="cuda",
    output_dir="experiments/ewc_vae",
    classifier=None,
    record_metric_fn=None,
    early_stopping_thr=None
):
    """trains ewc vae model on sequence of tasks
    
    params:
        lr: learning rate
        optimiser_type: adam or sgd
        momentum: sgd momentum if using sgd
        ewc_lambda: ewc penalty strength
        n_fisher_samples: num samples for fisher computation
        device: cuda or cpu
        classifier: optional classifier for metrics
        early_stopping_thr: optional early stopping threshold
    """
    start_time = time.time()
    out = Path(output_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)

    model = EWC_DGM_VAE(input_size, hidden_size, latent_size, num_tasks).to(device)

    test_loaders = []
    for t in range(num_tasks):
        print(f"\n---- Task {t+1}/{num_tasks} ----")
        train_loader = train_loader_factories[t]()

        # train
        _train_task(
            model, train_loader, t,
            epochs, lr, ewc_lambda, device,
            optimiser_type, momentum, record_metric_fn
        )

        # fisher & checkpoint
        print("  estimating fisher...")
        fisher = model.estimate_fisher(
            train_loader, device, n_samples=n_fisher_samples
        )
        model.register_ewc_task(fisher)
        torch.save(
            model.state_dict(),
            out / "checkpoints" / f"model_t{t+1}.pt"
        )

        # visuals
        for m in range(5):
            _create_montage(model, out / "samples", t, m+1)
        try:
            if wandb.run is not None:
                step = (t+1) * epochs
                wandb.log({
                    f"samples/task{t+1}_montage":
                        wandb.Image(str(out/"samples"/f"task{t+1}_montage1.png"))
                }, step=step)
        except Exception:
            pass

        # eval
        if len(test_loaders) <= t:
            test_loaders.append(test_loader_factories[t]())
        _visualise_reconstructions(model, test_loaders[t],
                                   out / "samples", t)

        eval_res = evaluate_model_dgm(
            model, test_loaders[:t+1], device,
            num_tasks_seen=t+1, classifier=classifier
        )
        _print_eval(t, eval_res)
        if record_metric_fn:
            for key, val in eval_res[-1].items():
                record_metric_fn(t+1, -1, key, val)

        if early_stopping_thr is not None:
            if eval_res[-1]['average_recon_error'] > early_stopping_thr:
                print("early stopping criterion reached.")
                break

        clean_loader(train_loader)
        clean_memory(device)


    total = time.time() - start_time
    if record_metric_fn:
        record_metric_fn(num_tasks, -1, "total_duration_seconds", total)
    print(f"\nTraining finished in {total/60:.1f} min  "
          f"|  checkpoints -> {out/'checkpoints'}")
    return model
