"""
VCL stuff for deep generative models. Algorithm 1 from the paper but for 
generative.
"""

import torch
import time
from tqdm import tqdm
import torch.optim as optim
import os
from pathlib import Path
import wandb
import sys

from z_utils.utils import clean_memory, clean_loader
from z_utils.utils_dgm import evaluate_model_dgm, compute_decoder_stats, create_task_samples_montage
from z_utils.utils_dgm import (
    debug_parameter_changes, print_parameter_change_summary,
    compute_kl_breakdown, analyze_parameter_statistics,
    print_parameter_statistics, create_task_samples_montage,
    create_multiple_task_samples_montages, create_single_sample_montages,
    create_multiple_samples_averaged_montages, create_reconstruction_visualizations
)


def train_dgm_epoch(model, loader, optimizer, device, n_samples=1):
    model.train()
    model = model.to(device)

    dataset_size=len(loader.dataset)
    
    total_elbo = 0.0
    total_recon_loss = 0.0  
    total_kl_latent = 0.0
    total_kl_params = 0.0
    total_samples = 0
    batch_pbar = tqdm(loader, desc="Batches", leave=False)
    
    for x_batch, _ in batch_pbar:
        x_batch = x_batch.to(device)
        batch_size = x_batch.size(0)
        
        # flatten if needed (img data)
        if x_batch.dim() > 2:
            x_batch = x_batch.view(batch_size, -1)
        
        optimizer.zero_grad()
        loss, metrics = model.compute_elbo(x_batch, dataset_size, n_samples=n_samples)
        
        loss.backward()
        optimizer.step()
        
        # track metrics
        total_elbo += metrics['elbo'] * batch_size
        total_recon_loss += metrics['recon_loss'] * batch_size
        total_kl_latent+=metrics['kl_latent'] * batch_size
        total_kl_params += metrics['kl_params'] * batch_size
        total_samples += batch_size
        
        batch_pbar.set_postfix({
            'elbo': f"{metrics['elbo']:.4f}",
            'recon': f"{metrics['recon_loss']:.4f}",
            'kl_z': f"{metrics['kl_latent']:.4f}",
            'kl_p': f"{metrics['kl_params']:.4f}"
        })
    
    # calcs
    avg_elbo = total_elbo / total_samples if total_samples > 0 else 0
    avg_recon_loss = total_recon_loss / total_samples if total_samples > 0 else 0
    avg_kl_latent = total_kl_latent / total_samples if total_samples > 0 else 0
    avg_kl_params = total_kl_params / total_samples if total_samples > 0 else 0
    
    return {
        'elbo': avg_elbo,
        'recon_loss': avg_recon_loss,
        'kl_latent': avg_kl_latent,
        'kl_params': avg_kl_params,
        'n_samples': total_samples
    }


def train_dgm_model_for_task(model, loader, task_idx, epochs, lr, device, 
                           n_samples=1, record_metric_fn=None, exp_dir=None):
    """
    trains a generative VCL model for a specific task
    
    Args:
        model: the model to train
        loader: dataloader for current task
        task_idx: index of current task
        epochs: num epochs to train for
        lr: learning rate
        device: device to train on
        n_samples: num samples for elbo estimation
        record_metric_fn: optional fn to record metrics
        exp_dir: optional dir to save stuff in
    """

    model.set_task(task_idx)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epoch_metrics = {
        'elbo': [], 'recon_loss': [], 'kl_latent': [], 'kl_params': []
    }
    
    print(f"\nTraining generative model on Task {task_idx+1}...")
    
    epoch_pbar = tqdm(range(1, epochs + 1), desc="Epochs", leave=False)
    
    # initial decoder stats
    if record_metric_fn:
        decoder_stats = compute_decoder_stats(model)
        record_metric_fn(task_idx, 0, 'decoder_weight_std', decoder_stats['decoder_weight_std'])
        record_metric_fn(task_idx, 0, 'decoder_bias_std', decoder_stats['decoder_bias_std'])
    

    for epoch in epoch_pbar:
        metrics = train_dgm_epoch(model, loader, optimizer, device, n_samples)

        epoch_metrics['elbo'].append(metrics['elbo'])
        epoch_metrics['recon_loss'].append(metrics['recon_loss'])
        epoch_metrics['kl_latent'].append(metrics['kl_latent'])
        epoch_metrics['kl_params'].append(metrics['kl_params'])
        
        epoch_pbar.set_postfix({
            'ELBO': f"{metrics['elbo']:.4f}",
            'Recon': f"{metrics['recon_loss']:.4f}",
            'KL_z': f"{metrics['kl_latent']:.4f}",
            'KL_p': f"{metrics['kl_params']:.4f}"
        })
        
        if record_metric_fn:
            record_metric_fn(task_idx, epoch, 'elbo', metrics['elbo'])
            record_metric_fn(task_idx, epoch, 'recon_loss', metrics['recon_loss'])
            record_metric_fn(task_idx, epoch, 'kl_latent', metrics['kl_latent'])
            record_metric_fn(task_idx, epoch, 'kl_params', metrics['kl_params'])
            
            decoder_stats = compute_decoder_stats(model)
            record_metric_fn(task_idx, epoch, 'decoder_weight_std', decoder_stats['decoder_weight_std'])
            record_metric_fn(task_idx, epoch, 'decoder_bias_std', decoder_stats['decoder_bias_std'])
    
    clean_memory(device)
    
    print(f"\nFinished training generative model on Task {task_idx+1}")
    
    print(f"  Final metrics for Task {task_idx+1}:")
    print(f"  ELBO: {metrics['elbo']:.4f} | Recon: {metrics['recon_loss']:.4f} | "
          f"KL_z: {metrics['kl_latent']:.4f} | KL_p: {metrics['kl_params']:.4f}")
    
    return model, metrics


def print_dgm_eval_results(task_idx, results):
    print(f"\nEvaluation Results after Task {task_idx+1}")
    
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


def setup_directories(exp_dir):
    if not exp_dir:
        return None, None
        
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    sample_dir = exp_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    
    return exp_dir, checkpoint_dir, sample_dir


def save_generated_samples(model, task_idx, sample_dir, n_samples=100):
    # TODO: Fix this later
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not sample_dir:
        return
        
    model.set_task(task_idx)
    model.eval()
    
    with torch.no_grad():
        samples = model.sample(n_samples, task_idx=task_idx).cpu()
    
    samples_np = samples.view(-1, 28, 28).numpy()
    
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        if i < n_samples:
            ax.imshow(samples_np[i], cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(sample_dir / f"task{task_idx+1}_samples.png", dpi=200)
    plt.close(fig)


def train_vcl_dgm(
    model_class,
    train_loader_factories,
    test_loader_factories,
    epochs_per_task=200,
    lr=1e-4,
    device='cuda',
    record_metric_fn=None,
    exp_dir=None,
    n_train_samples=1,
    n_eval_samples=100,
    classifier=None,
    init_std=0.001,
    early_stopping_threshold=None
):
    """
    trains vcl for generative models
    
    Args:
        model_class: class to instantiate model from
        train_loader_factories: list of funcs that create train dataloaders
        test_loader_factories: list of funcs that create test dataloaders
        epochs_per_task: num epochs to train each task
        lr: learning rate
        device: device to train on
        record_metric_fn: optional func to record metrics
        exp_dir: optional dir to save stuff in
        n_train_samples: num samples for training elbo estimation
        n_eval_samples: num samples for evaluation
        classifier: optional classifier for uncertainty estimation
        init_std: initial std for params
        early_stopping_threshold: optional threshold for early stopping
    """
    start_time = time.time()
    
    if exp_dir:
        exp_dir, checkpoint_dir, sample_dir = setup_directories(exp_dir)
    
    model = model_class().to(device)
    model.set_init_std(init_std)
    
    num_tasks = len(train_loader_factories)
    test_loaders = [None] * num_tasks
    
    task_metrics = []
    
    try:
        import wandb as wandb_module
        wandb_available = wandb_module is not None
    except ImportError:
        wandb_available = False
    
    task_pbar = tqdm(range(num_tasks), desc="Tasks", leave=False)
    
    for task_idx in task_pbar:
        task_pbar.set_description(f"Task {task_idx+1}")
        
        train_loader = train_loader_factories[task_idx]()
        
        model.set_task(task_idx)
        
        print(f"\n===== STARTING TRAINING FOR TASK {task_idx+1} =====")
        stats_before = analyze_parameter_statistics(model)
        print_parameter_statistics(stats_before, task_idx)
        
        model._debug_prev_state = debug_parameter_changes(model, task_idx, stage="before")
        
        kl_before = compute_kl_breakdown(model, task_idx)
        print(f"\nKL Divergence before training task {task_idx+1}:")
        print(f"  Shared layers KL: {kl_before['shared_kl']:.4f}")
        print(f"  Task-specific layers KL: {kl_before['task_kl']:.4f}")
        print(f"  Total KL: {kl_before['total_kl']:.4f}")
        print(f"  Shared layers ratio: {kl_before['shared_ratio']:.4f}")
        
        model, task_metric = train_dgm_model_for_task(
            model, train_loader, task_idx, epochs_per_task, 
            lr, device, n_train_samples, record_metric_fn, exp_dir
        )
        
        param_differences = debug_parameter_changes(model, task_idx, stage="after")
        print_parameter_change_summary(param_differences, task_idx)
        
        kl_after = compute_kl_breakdown(model, task_idx)
        print(f"\nKL Divergence after training task {task_idx+1}:")
        print(f"  Shared layers KL: {kl_after['shared_kl']:.4f}")
        print(f"  Task-specific layers KL: {kl_after['task_kl']:.4f}")
        print(f"  Total KL: {kl_after['total_kl']:.4f}")
        print(f"  Shared layers ratio: {kl_after['shared_ratio']:.4f}")
        
        model.store_params_as_old()
        
        print("\n=== Parameter Statistics After Storing Parameters ===")
        stats_after_reset = analyze_parameter_statistics(model)
        print_parameter_statistics(stats_after_reset, task_idx)
        
        clean_loader(train_loader)
        
        if checkpoint_dir:
            torch.save(model.state_dict(), checkpoint_dir / f"model_task{task_idx+1}.pt")
        
        test_loaders[task_idx] = test_loader_factories[task_idx]()
        
        if sample_dir:
            print("Generating and saving samples...")
            for t_idx in range(task_idx + 1):
                montage_paths = create_multiple_task_samples_montages(
                    model, device, exp_dir, task_idx, num_montages=5
                )
                
                single_sample_paths = create_single_sample_montages(
                    model, device, exp_dir, task_idx, num_montages=5
                )
                
                avg_sample_paths = create_multiple_samples_averaged_montages(
                    model, device, exp_dir, task_idx, 
                    num_montages=5, samples_per_image=100
                )
                
                if wandb_available:
                    log_dict = {
                        f'samples_means_task{t_idx+1}_after_task{task_idx+1}': 
                            wandb_module.Image(str(montage_paths[0])),
                        f'samples_single_task{t_idx+1}_after_task{task_idx+1}': 
                            wandb_module.Image(str(single_sample_paths[0])),
                        f'samples_averaged_task{t_idx+1}_after_task{task_idx+1}': 
                            wandb_module.Image(str(avg_sample_paths[0]))
                    }
                    
                    wandb_module.log(log_dict, step=(task_idx+1)*epochs_per_task)
            
            recon_paths = create_reconstruction_visualizations(
                model, test_loaders[:task_idx+1], device, exp_dir, task_idx
            )
            
            print(f"Saved visualizations to {exp_dir}")
        
        eval_results = evaluate_model_dgm(
            model, test_loaders[:task_idx+1], device, num_tasks_seen=task_idx+1,
            classifier=classifier, n_ll_samples=n_eval_samples
        )
        
        print_dgm_eval_results(task_idx, eval_results)
        
        task_metrics.append(task_metric)
        
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
                
        if early_stopping_threshold is not None:
            avg_recon_error = eval_results[-1].get('average_recon_error', float('inf'))
            
            if avg_recon_error > early_stopping_threshold:
                print(f"\nEARLY STOPPING: Average reconstruction error ({avg_recon_error:.4f}) exceeded threshold ({early_stopping_threshold:.4f})")
                print("Terminating training early to save computation resources.")
                
                for loader in test_loaders:
                    if loader is not None:
                        clean_loader(loader)
                
                if wandb_available:
                    wandb_module.run.summary["early_stopped"] = True
                    wandb_module.run.summary["early_stopping_reason"] = f"Recon error {avg_recon_error:.4f} above threshold {early_stopping_threshold:.4f}"
                    wandb_module.run.summary["early_stopping_task"] = task_idx + 1
                    
                total_time = time.time() - start_time
                print(f"\nTotal training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
                
                if record_metric_fn:
                    record_metric_fn(task_idx, -1, 'early_stopped', 1)
                    record_metric_fn(task_idx, -1, 'early_stopping_task', task_idx + 1)
                    record_metric_fn(task_idx, -1, 'total_duration_seconds', total_time)
                
                return model
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    if record_metric_fn:
        record_metric_fn(num_tasks-1, -1, 'total_duration_seconds', total_time)
    
    for loader in test_loaders:
        if loader is not None:
            clean_loader(loader)
    
    return model