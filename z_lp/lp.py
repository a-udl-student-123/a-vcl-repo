
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from z_utils.utils import (evaluate_all_tasks, evaluate_model, clean_loader,
                           clean_memory)


def compute_lp_loss(model, criterion, outputs, targets, lp_lambda, dataset_size):
    # cross entropy + sum_t (lambda/2|T|) * (theta - theta_t)^T H_t (theta - theta_t)
    ce = criterion(outputs, targets)

    penalty = torch.zeros(1, device=outputs.device)
    for hessian, theta_star in getattr(model, "lp_data", []):
        for n, p in model.named_parameters():
            penalty += (hessian[n] * (p - theta_star[n]).pow(2)).sum()

    n_prev = max(len(model.lp_data), 1)
    lp_term = 0.5 * (lp_lambda / n_prev) * penalty
    return ce + lp_term, ce.item(), lp_term.item()


def train_lp(model_class,
             train_loader_factories,
             test_loader_factories,
             epochs_per_task,
             lr,
             lp_lambda,
             device,
             record_metric_fn,
             exp_dir,
             num_workers=4,
             batch_size=256,
             n_train_samples=600,
             **_ignored):
    model = model_class().to(device)

    exp_dir = Path(exp_dir)
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    prev_train_loader = None
    test_loaders = [None] * len(test_loader_factories)

    n_tasks = len(train_loader_factories)
    task_bar = tqdm(range(1, n_tasks + 1), desc="Tasks", leave=False)

    for task_idx in task_bar:
        idx = task_idx - 1
        task_bar.set_description(f"Task {task_idx}/{n_tasks}")

        # finish prev task - estimate hessian and register
        if task_idx > 1 and prev_train_loader is not None:
            if hasattr(model, "set_current_task"):
                model.set_current_task(idx - 1)
            print(f"> Estimating Hessian for task {task_idx-1}")
            hess = model.estimate_hessian(prev_train_loader, device,
                                          n_samples=n_train_samples)
            model.register_lp_task(hess)
            clean_loader(prev_train_loader)
            prev_train_loader = None
            clean_memory(device)

        # switch head if multi-head model
        if hasattr(model, "set_current_task"):
            model.set_current_task(idx)

        train_loader = train_loader_factories[idx]()
        if test_loaders[idx] is None:
            test_loaders[idx] = test_loader_factories[idx]()
        prev_train_loader = train_loader

        optimiser = optim.SGD(model.parameters(), lr=lr)
        epoch_bar = tqdm(range(1, epochs_per_task + 1),
                         desc="Epochs", leave=False)

        for epoch in epoch_bar:
            tot_loss = tot_ce = tot_pen = tot_gn = 0.0
            correct = total = 0
            for xb, yb in tqdm(train_loader, desc=f"Ep {epoch}", leave=False):
                xb, yb = xb.to(device), yb.to(device)
                bs = xb.size(0)

                optimiser.zero_grad(set_to_none=True)
                out = model(xb)
                loss, ce_val, pen_val = compute_lp_loss(
                    model, criterion, out, yb, lp_lambda
                )
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    max_norm=100.0)
                optimiser.step()

                _, pred = out.max(1)
                correct += (pred == yb).sum().item()
                total += bs
                tot_loss += loss.item() * bs
                tot_ce += ce_val * bs
                tot_pen += pen_val * bs
                tot_gn += gn.item() * bs

            avg_loss = tot_loss / total
            avg_ce = tot_ce / total
            avg_pen = tot_pen / total
            avg_gn = tot_gn / total
            acc = correct / total

            epoch_bar.set_postfix(
                loss=f"{avg_loss:.3f}", ce=f"{avg_ce:.3f}",
                lp=f"{avg_pen:.3f}", acc=f"{100*acc:.1f}%",
                grad=f"{avg_gn:.3f}"
            )

            if record_metric_fn:
                record_metric_fn(idx, epoch, "train_loss", avg_loss)
                record_metric_fn(idx, epoch, "train_ce_loss", avg_ce)
                record_metric_fn(idx, epoch, "train_lp_loss", avg_pen)
                record_metric_fn(idx, epoch, "train_accuracy", acc)
                record_metric_fn(idx, epoch, "train_grad_norm", avg_gn)

        print(f"\n=== Task {task_idx} | Final epoch ===")
        print(f"accuracy {acc:.4f} | loss {avg_loss:.4f} | LP‑penalty {avg_pen:.4f}")

        model.eval()
        if hasattr(model, "heads"):
            saved = model.current_task
            accs, ces = [], []
            for t in range(task_idx):
                model.set_current_task(t)
                a, l = evaluate_model(model, test_loaders[t], device)
                accs.append(a); ces.append(l)
            model.set_current_task(saved)
        else:
            accs, ces, _, _ = evaluate_all_tasks(model, test_loaders,
                                                 task_idx, device, None)

        avg_acc = sum(accs)/len(accs)
        avg_ce = sum(ces)/len(ces)
        print("\n--- Eval on seen tasks ---")
        for i, (a, l) in enumerate(zip(accs, ces)):
            print(f"Task {i+1}: acc {a:.4f} | ce {l:.4f}")
        print(f"Average: acc {avg_acc:.4f} | ce {avg_ce:.4f}\n")

        if record_metric_fn:
            record_metric_fn(idx, -1, "average_accuracy", avg_acc)
            record_metric_fn(idx, -1, "average_ce_loss", avg_ce)
            for i, a in enumerate(accs):
                record_metric_fn(idx, -1, f"accuracy_on_task_{i+1}", a)

        torch.save(model.state_dict(), ckpt_dir / f"task_{task_idx}.pt")
        clean_memory(device)

    # get hessian for final task
    if prev_train_loader is not None:
        print("> Estimating Hessian for final task")
        if hasattr(model, "set_current_task"):
            model.set_current_task(n_tasks - 1)
        hess = model.estimate_hessian(prev_train_loader, device,
                                      n_samples=n_train_samples)
        model.register_lp_task(hess)

    for tl in test_loaders:
        if tl is not None:
            clean_loader(tl)
    if prev_train_loader:
        clean_loader(prev_train_loader)
    clean_memory(device)
    return model
