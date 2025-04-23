# training functions for si + VAE

import os
import sys
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import numpy as np
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.filterwarnings("ignore", message="A newer version of deeplake.*")
warnings.filterwarnings("ignore", category=UserWarning, module="deeplake")

from z_utils.utils import clean_memory, clean_loader
from z_utils.utils_dgm import evaluate_model_dgm
from z_synaptic_dgm.si_vae import SI_DGM_VAE

def train_dgm_epoch_with_si(model, loader, optimizer, device):

    model.train()
    model = model.to(device)

    dataset_size = len(loader.dataset)
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_latent = 0.0
    total_si_loss = 0.0
    total_elbo = 0.0
    total_samples = 0
    
    batch_pbar = tqdm(loader, desc="Batches", leave=False)
    for x_batch, _ in batch_pbar:
        x_batch = x_batch.to(device)
        batch_size = x_batch.size(0)
        
        if x_batch.dim() > 2:
            x_batch = x_batch.view(batch_size, -1)
        
        optimizer.zero_grad()
        loss, metrics = model.compute_loss(x_batch, dataset_size)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: Loss is NaN/Inf. Skipping batch.")
            continue
        
        model.before_optimizer_step()
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        model.after_optimizer_step()
        
        total_loss += loss.item() * batch_size
        total_recon_loss += metrics['recon_loss'] * batch_size
        total_kl_latent += metrics['kl_latent'] * batch_size
        total_si_loss += metrics['si_loss'] * batch_size
        total_elbo += metrics['elbo'] * batch_size
        total_samples += batch_size
        
        batch_pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'recon': f"{metrics['recon_loss']:.4f}", 
            'kl_z': f"{metrics['kl_latent']:.4f}",
            'si': f"{metrics['si_loss']:.4f}"
        })
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    avg_recon_loss = total_recon_loss / total_samples if total_samples > 0 else 0
    avg_kl_latent = total_kl_latent / total_samples if total_samples > 0 else 0
    avg_si_loss = total_si_loss / total_samples if total_samples > 0 else 0
    avg_elbo = total_elbo / total_samples if total_samples > 0 else 0
    
    return {
        'loss': avg_loss,
        'recon_loss': avg_recon_loss,
        'kl_latent': avg_kl_latent,
        'si_loss': avg_si_loss,
        'elbo': avg_elbo,
        'n_samples': total_samples
    }

def train_dgm_model_for_task_with_si(model, loader, task_idx, epochs, lr, device,
                                    optimizer_type="sgd", momentum=0.9,
                                    record_metric_fn=None, exp_dir=None):
    # train VAE on one task
    model.set_task(task_idx)

    if optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    epoch_metrics = {
        'loss': [], 'recon_loss': [], 'kl_latent': [], 'si_loss': [], 'elbo': []
    }
    
    print(f"\nTraining generative model with SI on Task {task_idx+1}...")
    
    epoch_pbar = tqdm(range(1, epochs + 1), desc="Epochs", leave=False)
    
    for epoch in epoch_pbar:
        metrics = train_dgm_epoch_with_si(model, loader, optimizer, device)
        
        epoch_metrics['loss'].append(metrics['loss'])
        epoch_metrics['recon_loss'].append(metrics['recon_loss'])
        epoch_metrics['kl_latent'].append(metrics['kl_latent'])
        epoch_metrics['si_loss'].append(metrics['si_loss'])
        epoch_metrics['elbo'].append(metrics['elbo'])
        
        epoch_pbar.set_postfix({
            'Loss': f"{metrics['loss']:.4f}",
            'Recon': f"{metrics['recon_loss']:.4f}",
            'KL_z': f"{metrics['kl_latent']:.4f}",
            'SI': f"{metrics['si_loss']:.4f}"
        })
        
        if record_metric_fn:
            record_metric_fn(task_idx, epoch, 'loss', metrics['loss'])
            record_metric_fn(task_idx, epoch, 'recon_loss', metrics['recon_loss'])
            record_metric_fn(task_idx, epoch, 'kl_latent', metrics['kl_latent'])
            record_metric_fn(task_idx, epoch, 'si_loss', metrics['si_loss'])
            record_metric_fn(task_idx, epoch, 'elbo', metrics['elbo'])
    
    model.complete_task()
    
    clean_memory(device)
    
    print(f"\nFinished training generative model with SI on Task {task_idx+1}")
    print(f"  Final metrics for Task {task_idx+1}:")
    print(f"  Loss: {metrics['loss']:.4f} | Recon: {metrics['recon_loss']:.4f} | "
          f"KL_z: {metrics['kl_latent']:.4f} | SI: {metrics['si_loss']:.4f}")
    
    return model, metrics

def print_dgm_eval_results(task_idx, results):
    print(f"\n=== Evaluation Results after Task {task_idx+1} ===")
    
    for t, task_result in enumerate(results[:-1]):
        print(f"  Task {t+1}:")
        print(f"    Reconstruction Error: {task_result.get('recon_error', 'N/A'):.4f}")
        print(f"    Log-Likelihood: {task_result.get('log_likelihood', 'N/A'):.4f}")
        
        if 'classifier_uncertainty' in task_result:
            print(f"    Classifier Uncertainty: {task_result['classifier_uncertainty']:.4f}")
    
    if len(results) > 1:
        avg_recon_error = sum(r.get('recon_error', 0) for r in results[:-1]) / len(results[:-1])
        avg_ll = sum(r.get('log_likelihood', 0) for r in results[:-1]) / len(results[:-1])
        avg_cu = sum(r.get('classifier_uncertainty', 0) for r in results[:-1] 
                    if 'classifier_uncertainty' in r) / len(results[:-1])
        
        print(f"\n  Average Log-Likelihood: {avg_ll:.4f}")
        print(f"  Average Reconstruction Error: {avg_recon_error:.4f}")
        print(f"  Average Classifier Uncertainty: {avg_cu:.4f}")
    
    print("  " + "-" * 60)

def create_task_montage(model, output_dir, task_idx, montage_idx=1):
    # create montage with one sample from each task
    model.eval()
    
    samples = []
    
    for t in range(task_idx + 1):
        model.set_task(t)
        
        with torch.no_grad():
            sample = model.sample(1, task_idx=t)[0].cpu()
        
        samples.append(sample)
    
    num_tasks = len(samples)
    fig, axes = plt.subplots(1, num_tasks, figsize=(num_tasks * 2, 2))
    
    if num_tasks == 1:
        axes = [axes]
    
    for i, sample in enumerate(samples):
        ax = axes[i]
        ax.imshow(sample.view(28, 28), cmap='gray')
        ax.set_title(f"Task {i+1}")
        ax.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"task{task_idx+1}_montage{montage_idx}.png")
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    
    return output_path

def visualize_reconstructions(model, dataloader, output_dir, task_idx, num_samples=5):
    model.eval()
    
    for x_batch, _ in dataloader:
        x_batch = x_batch[:num_samples].to(next(model.parameters()).device)
        break
    
    reconstructions = model.reconstruct(x_batch, task_idx=task_idx)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    
    for i in range(num_samples):
        axes[0, i].imshow(x_batch[i].cpu().view(28, 28), cmap='gray')
        axes[0, i].set_title(f"Original")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(reconstructions[i].cpu().view(28, 28), cmap='gray')
        axes[1, i].set_title(f"Reconstruction")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"task{task_idx+1}_reconstructions.png")
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    
    return output_path

def train_si_vae(
    train_loader_factories, 
    test_loader_factories, 
    num_tasks=10,
    hidden_size=500,
    latent_size=50,
    input_size=784,
    epochs=20,
    learning_rate=1e-3,
    optimizer_type="sgd",
    momentum=0.9,
    si_lambda=0.1,
    si_epsilon=1e-3,
    omega_decay=0.9,
    device='cuda',
    output_dir='experiments/si_vae',
    classifier=None,
    record_metric_fn=None,
    early_stopping_threshold=None
):

    start_time = time.time()
    
    exp_dir = Path(output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "samples").mkdir(exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    
    print("Initializing SI-VAE model...")
    model = SI_DGM_VAE(
        input_size=input_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        num_tasks=num_tasks
    ).to(device)
    
    model.set_si(
        lambda_reg=si_lambda,
        epsilon=si_epsilon,
        omega_decay=omega_decay,
        device=device
    )
    
    metrics = []
    if record_metric_fn is None:
        def default_record_metric(task, epoch, name, value):
            metrics.append({
                'task': task,
                'epoch': epoch,
                'metric': name,
                'value': value
            })
        record_metric_fn = default_record_metric
    
    test_loaders = []
    task_metrics = []
    
    progress_bar = tqdm(range(num_tasks), desc="Tasks", leave=False)
    for task_idx in progress_bar:
        print(f"\n====== Task {task_idx+1}/{num_tasks} ======")
        
        train_loader = train_loader_factories[task_idx]()
        
        model, task_result = train_dgm_model_for_task_with_si(
            model=model,
            loader=train_loader,
            task_idx=task_idx,
            epochs=epochs,
            lr=learning_rate,
            device=device,
            optimizer_type=optimizer_type,
            momentum=momentum,
            record_metric_fn=record_metric_fn,
            exp_dir=exp_dir
        )
        
        torch.save(model.state_dict(), exp_dir / f"checkpoints/model_task{task_idx+1}.pt")
        
        clean_loader(train_loader)
        
        while len(test_loaders) <= task_idx:
            test_loaders.append(test_loader_factories[len(test_loaders)]())
        
        montage_paths = []
        for i in range(5):
            montage_path = create_task_montage(model, exp_dir / "samples", task_idx, i+1)
            montage_paths.append(montage_path)
        print(f"Generated task montages saved to {exp_dir / 'samples'}")
        
        try:
            import wandb
            if wandb.run is not None:
                global_step = task_idx * epochs + epochs
                
                wandb.log({
                    f"samples/task{task_idx+1}_montage": wandb.Image(str(montage_paths[0]))
                }, step=global_step)
                
                for t in range(task_idx + 1):
                    wandb.log({
                        f"samples/task{t+1}_after_task{task_idx+1}": wandb.Image(str(montage_paths[0]))
                    }, step=global_step)
        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: Failed to log montages to wandb: {str(e)}")
        
        recon_path = visualize_reconstructions(model, test_loaders[task_idx], exp_dir / "samples", task_idx)
        print(f"Reconstructions saved to {recon_path}")
        
        try:
            import wandb
            if wandb.run is not None:
                global_step = task_idx * epochs + epochs
                
                wandb.log({
                    f"reconstructions/task{task_idx+1}": wandb.Image(str(recon_path))
                }, step=global_step)
        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: Failed to log reconstructions to wandb: {str(e)}")
        
        eval_results = evaluate_model_dgm(
            model, test_loaders[:task_idx+1], device,
            num_tasks_seen=task_idx+1, classifier=classifier
        )
        
        print_dgm_eval_results(task_idx, eval_results)
        
        if record_metric_fn:
            for t, result in enumerate(eval_results[:-1]):
                if 'log_likelihood' in result:
                    record_metric_fn(task_idx, -1, f'log_likelihood_task_{t+1}', result['log_likelihood'])
                if 'recon_error' in result:
                    record_metric_fn(task_idx, -1, f'recon_error_task_{t+1}', result['recon_error'])
                if 'classifier_uncertainty' in result:
                    record_metric_fn(task_idx, -1, f'cls_uncertainty_task_{t+1}', result['classifier_uncertainty'])
            
            avg_metrics = eval_results[-1]
            for key, value in avg_metrics.items():
                record_metric_fn(task_idx, -1, key, value)
        
        task_metrics.append({
            'train': task_result,
            'eval': eval_results
        })
        
        if early_stopping_threshold is not None:
            avg_recon_error = eval_results[-1].get('average_recon_error', float('inf'))
            
            if avg_recon_error > early_stopping_threshold:
                print(f"\n⚠️ EARLY STOPPING: Average reconstruction error ({avg_recon_error:.4f}) exceeded threshold ({early_stopping_threshold:.4f})")
                print("Terminating training early to save computation resources.")
                
                if record_metric_fn:
                    record_metric_fn(task_idx, -1, 'early_stopped', 1)
                    record_metric_fn(task_idx, -1, 'early_stopping_task', task_idx + 1)
                
                for loader in test_loaders:
                    clean_loader(loader)
                
                total_time = time.time() - start_time
                print(f"\nTotal training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
                
                if record_metric_fn:
                    record_metric_fn(task_idx, -1, 'total_duration_seconds', total_time)
                
                return model, task_metrics
    
    for loader in test_loaders:
        clean_loader(loader)
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    if record_metric_fn:
        record_metric_fn(num_tasks-1, -1, 'total_duration_seconds', total_time)
    
    print(f"\nTraining completed. All outputs saved to {exp_dir}")
    
    return model, task_metrics