import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

from z_utils.utils import evaluate_all_tasks, clean_memory, clean_loader, evaluate_model


def compute_ewc_loss(model, criterion, outputs, targets, ewc_lambda):
    ce = criterion(outputs, targets)

    # init penalty on same device
    penalty = torch.zeros(1, device=outputs.device)
    for fisher, old_params in getattr(model, "ewc_data", []):
        for n, p in model.named_parameters():
            penalty += (fisher[n] * (p - old_params[n]).pow(2)).sum()

    n_prev = max(len(model.ewc_data), 1)
    eff_lambda = ewc_lambda/n_prev
    ewc_term = 0.5 * eff_lambda * penalty
    return ce + ewc_term, ce.item(), ewc_term.item()


def train_ewc(model_class,
              train_loader_factories,
              test_loader_factories, 
              epochs_per_task,
              lr,
              ewc_lambda,
              device,
              record_metric_fn,
              exp_dir,
              num_workers=4,
              batch_size=256,
              n_train_samples=600,
              **kw):
    # trains model sequentially on tasks using ewc regularization
    model = model_class().to(device)

    exp_dir = Path(exp_dir)
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    prev_train_loader = None
    test_loaders = [None] * len(test_loader_factories)


    num_tasks = len(train_loader_factories)
    task_pbar = tqdm(range(1, num_tasks + 1), desc="Tasks", leave=False)

    for task_idx in task_pbar:
        t_idx = task_idx - 1
        task_pbar.set_description(f"Task {task_idx}/{num_tasks}")

        if hasattr(model, "set_current_task"):
            model.set_current_task(t_idx)

        # save fisher before moving on
        if task_idx > 1 and prev_train_loader is not None:
            print(f"> Estimating Fisher for task {task_idx-1}")
            fisher = model.estimate_fisher(prev_train_loader,
                                           device,
                                           n_samples=n_train_samples)
            model.register_ewc_task(fisher)
            clean_loader(prev_train_loader)
            prev_train_loader = None
            clean_memory(device)

        train_loader = train_loader_factories[t_idx]()
        if test_loaders[t_idx] is None:
            test_loaders[t_idx] = test_loader_factories[t_idx]()
        prev_train_loader = train_loader

        optimizer = optim.SGD(model.parameters(), lr=lr)

        model.train()
        epoch_pbar = tqdm(range(1, epochs_per_task + 1),
                          desc="Epochs", leave=False)

        avg_loss = avg_ce = avg_pen = acc = 0.0

        for epoch in epoch_pbar:
            tot_loss = tot_ce = tot_pen = tot_grad_norm = 0.0
            correct = total = 0

            batch_pbar = tqdm(train_loader,
                              desc=f"Epoch {epoch}", leave=False)

            for x, y in batch_pbar:
                x, y = x.to(device), y.to(device)
                bsz = x.size(0)

                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                loss, ce_val, pen_val = compute_ewc_loss(
                    model, criterion, out, y, ewc_lambda
                )
                loss.backward()
                
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
                optimizer.step()

                _, preds = out.max(1)
                correct += (preds == y).sum().item()
                total += bsz

                tot_loss += loss.item() * bsz
                tot_ce   += ce_val     * bsz
                tot_pen  += pen_val    * bsz
                tot_grad_norm += grad_norm.item() * bsz

                batch_pbar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    ce=f"{ce_val:.3f}",
                    ewc=f"{pen_val:.3f}",
                    acc=f"{100*(preds==y).float().mean():.1f}%"
                )

            avg_loss = tot_loss / total
            avg_ce   = tot_ce   / total  
            avg_pen  = tot_pen  / total
            acc      = correct / total
            avg_grad_norm = tot_grad_norm / total

            epoch_pbar.set_postfix(
                loss=f"{avg_loss:.3f}",
                ce=f"{avg_ce:.3f}",
                ewc=f"{avg_pen:.3f}",
                acc=f"{100*acc:.1f}%",
                grad_norm=f"{avg_grad_norm:.3f}"
            )

            if record_metric_fn:
                record_metric_fn(t_idx, epoch, "train_loss",     avg_loss)
                record_metric_fn(t_idx, epoch, "train_ce_loss",  avg_ce)
                record_metric_fn(t_idx, epoch, "train_ewc_loss", avg_pen)
                record_metric_fn(t_idx, epoch, "train_accuracy", acc)
                record_metric_fn(t_idx, epoch, "train_grad_norm", avg_grad_norm)

        print(f"\n=== Task {task_idx} | Final epoch ===")
        print(f"accuracy {acc:.4f} | loss {avg_loss:.4f} | "
              f"EWC-penalty {avg_pen:.4f}")

        # TODO: add early stopping?
        model.eval()
        if hasattr(model, 'heads'):
            accs, ce_losses, avg_acc, avg_ce = evaluate_multi_head(
                model, test_loaders, task_idx, device
            )
        else:
            accs, ce_losses, avg_acc, avg_ce = evaluate_all_tasks(
                model, test_loaders, task_idx, device, 1, None
            )

        print("\n--- Evaluation on seen tasks ---")
        for i, (a, l) in enumerate(zip(accs, ce_losses)):
            print(f"Task {i+1}: acc {a:.4f}, ce-loss {l:.4f}")
        print(f"Average : acc {avg_acc:.4f}, ce-loss {avg_ce:.4f}\n")

        if record_metric_fn:
            record_metric_fn(t_idx, -1, "average_accuracy", avg_acc)
            record_metric_fn(t_idx, -1, "average_ce_loss",  avg_ce)
            for i, a in enumerate(accs):
                record_metric_fn(t_idx, -1, f"accuracy_on_task_{i+1}", a)

        torch.save(model.state_dict(), ckpt_dir / f"task_{task_idx}.pt")
        clean_memory(device)

    for tl in test_loaders:
        if tl is not None:
            clean_loader(tl)
    if prev_train_loader:
        clean_loader(prev_train_loader)
    clean_memory(device)

    return model


def evaluate_multi_head(model, test_loaders, task_idx, device):
    model.eval()
    accuracies, ce_losses = [], []

    saved = model.current_task  # remember active head

    for t in tqdm(range(task_idx), desc="Task Eval", leave=False):
        model.set_current_task(t)
        acc, ce = evaluate_model(model, test_loaders[t], device)
        accuracies.append(acc)
        ce_losses.append(ce)

    model.set_current_task(saved)
    clean_memory(device)

    avg_acc = sum(accuracies) / len(accuracies)
    avg_ce  = sum(ce_losses) / len(ce_losses)
    return accuracies, ce_losses, avg_acc, avg_ce