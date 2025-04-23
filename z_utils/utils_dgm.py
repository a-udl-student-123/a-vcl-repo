"""
Utility functions for DGM Models.
"""

import torch
import torch.nn.functional as F
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Import from existing utils module instead of duplicating
from z_utils.utils import clean_memory


def debug_parameter_changes(model, task_idx, stage="before"):
    """
    Track parameter changes before and after training for a task.
    """
    if stage == "before":
        state = {
            "shared_layers": [],
            "task_layers": []
        }
        

        for i, layer in enumerate(model.decoder.shared_layers):
            state["shared_layers"].append({
                "layer_idx": i,
                "weight_mu": layer.weight_mu.detach().cpu(),
                "weight_sigma": layer._get_sigma(layer.weight_rho).detach().cpu(),
                "bias_mu": layer.bias_mu.detach().cpu(),
                "bias_sigma": layer._get_sigma(layer.bias_rho).detach().cpu()
            })
        
        if task_idx < len(model.decoder.task_layers):
            task_network = model.decoder.task_layers[task_idx]
            for i, layer in enumerate(task_network):
                state["task_layers"].append({
                    "layer_idx": i,
                    "weight_mu": layer.weight_mu.detach().cpu(),
                    "weight_sigma": layer._get_sigma(layer.weight_rho).detach().cpu(),
                    "bias_mu": layer.bias_mu.detach().cpu(),
                    "bias_sigma": layer._get_sigma(layer.bias_rho).detach().cpu()
                })
        
        return state
    
    elif stage == "after":
        # compute diff from previous state
        prev_state = model._debug_prev_state
        differences = {
            "shared_layers": [],
            "task_layers": []
        }
        
        # compute differences for shared 
        for i, layer in enumerate(model.decoder.shared_layers):
            prev_layer = prev_state["shared_layers"][i]
            current_weight_mu = layer.weight_mu.detach().cpu()
            current_weight_sigma = layer._get_sigma(layer.weight_rho).detach().cpu()
            current_bias_mu = layer.bias_mu.detach().cpu()
            current_bias_sigma = layer._get_sigma(layer.bias_rho).detach().cpu()
            
            #  absolute differences
            weight_mu_diff = (current_weight_mu - prev_layer["weight_mu"]).abs().mean().item()
            weight_sigma_diff = (current_weight_sigma - prev_layer["weight_sigma"]).abs().mean().item()
            bias_mu_diff = (current_bias_mu - prev_layer["bias_mu"]).abs().mean().item()
            bias_sigma_diff = (current_bias_sigma - prev_layer["bias_sigma"]).abs().mean().item()
            
            differences["shared_layers"].append({
                "layer_idx": i,
                "weight_mu_diff": weight_mu_diff,
                "weight_sigma_diff": weight_sigma_diff,
                "bias_mu_diff": bias_mu_diff,
                "bias_sigma_diff": bias_sigma_diff
            })
        
        # diff task-specific layers
        if task_idx < len(model.decoder.task_layers):
            task_network = model.decoder.task_layers[task_idx]
            for i, layer in enumerate(task_network):
                prev_layer = prev_state["task_layers"][i]
                current_weight_mu = layer.weight_mu.detach().cpu()
                current_weight_sigma = layer._get_sigma(layer.weight_rho).detach().cpu()
                current_bias_mu = layer.bias_mu.detach().cpu()
                current_bias_sigma = layer._get_sigma(layer.bias_rho).detach().cpu()
                
                # calc absolute differences
                weight_mu_diff = (current_weight_mu - prev_layer["weight_mu"]).abs().mean().item()
                weight_sigma_diff = (current_weight_sigma - prev_layer["weight_sigma"]).abs().mean().item()
                bias_mu_diff = (current_bias_mu - prev_layer["bias_mu"]).abs().mean().item()
                bias_sigma_diff = (current_bias_sigma - prev_layer["bias_sigma"]).abs().mean().item()
                
                differences["task_layers"].append({
                    "layer_idx": i,
                    "weight_mu_diff": weight_mu_diff,
                    "weight_sigma_diff": weight_sigma_diff,
                    "bias_mu_diff": bias_mu_diff,
                    "bias_sigma_diff": bias_sigma_diff
                })
        
        return differences


def print_parameter_change_summary(differences, task_idx):
    """Print a summary of parameter changes after training on a task."""
    print(f"\n=== Parameter Change Summary for Task {task_idx+1} ===")
    

    print("\nShared Layers:")
    for i, layer_diff in enumerate(differences["shared_layers"]):
        print(f"  Layer {i+1}:")
        print(f"    Mean Weight μ Δ: {layer_diff['weight_mu_diff']:.6f}")
        print(f"    Mean Weight σ Δ: {layer_diff['weight_sigma_diff']:.6f}")
        print(f"    Mean Bias μ Δ: {layer_diff['bias_mu_diff']:.6f}")
        print(f"    Mean Bias σ Δ: {layer_diff['bias_sigma_diff']:.6f}")

    print(f"\nTask {task_idx+1} Specific Layers:")
    for i, layer_diff in enumerate(differences["task_layers"]):
        print(f"  Layer {i+1}:")
        print(f"    Mean Weight μ Δ: {layer_diff['weight_mu_diff']:.6f}")
        print(f"    Mean Weight σ Δ: {layer_diff['weight_sigma_diff']:.6f}")
        print(f"    Mean Bias μ Δ: {layer_diff['bias_mu_diff']:.6f}")
        print(f"    Mean Bias σ Δ: {layer_diff['bias_sigma_diff']:.6f}")


def compute_kl_breakdown(model, task_idx=None):
    if task_idx is None:
        task_idx = model.current_task
    
    shared_kl = 0
    for layer in model.decoder.shared_layers:
        shared_kl += layer.kl_to_old_posterior().item()
    
    task_kl = 0
    if task_idx < len(model.decoder.task_layers):
        task_network = model.decoder.task_layers[task_idx]
        for layer in task_network:
            task_kl += layer.kl_to_old_posterior().item()
    
    return {
        "shared_kl": shared_kl,
        "task_kl": task_kl,
        "total_kl": shared_kl + task_kl,
        "shared_ratio": shared_kl / (shared_kl + task_kl) if (shared_kl + task_kl) > 0 else 0
    }


def analyze_parameter_statistics(model):
    """
    Analyze parameter statistics for debugging.
    """
    stats = {
        "shared_layers": [],
        "task_layers": []
    }
    
    for i, layer in enumerate(model.decoder.shared_layers):
        weight_mu = layer.weight_mu.detach()
        weight_sigma = layer._get_sigma(layer.weight_rho).detach()
        bias_mu = layer.bias_mu.detach()
        bias_sigma = layer._get_sigma(layer.bias_rho).detach()
        

        stats["shared_layers"].append({
            "layer_idx": i,
            "weight_mu_mean": weight_mu.mean().item(),
            "weight_mu_std": weight_mu.std().item(),
            "weight_mu_min": weight_mu.min().item(),
            "weight_mu_max": weight_mu.max().item(),
            "weight_sigma_mean": weight_sigma.mean().item(),
            "weight_sigma_min": weight_sigma.min().item(),
            "weight_sigma_max": weight_sigma.max().item(),
            "bias_mu_mean": bias_mu.mean().item(), 
            "bias_sigma_mean": bias_sigma.mean().item()
        })
    

    task_idx = model.current_task
    if task_idx < len(model.decoder.task_layers):
        task_network = model.decoder.task_layers[task_idx]
        for i, layer in enumerate(task_network):
            weight_mu = layer.weight_mu.detach()
            weight_sigma = layer._get_sigma(layer.weight_rho).detach()
            bias_mu = layer.bias_mu.detach()
            bias_sigma = layer._get_sigma(layer.bias_rho).detach()
            
  
            stats["task_layers"].append({
                "layer_idx": i,
                "weight_mu_mean": weight_mu.mean().item(),
                "weight_mu_std": weight_mu.std().item(),
                "weight_mu_min": weight_mu.min().item(),
                "weight_mu_max": weight_mu.max().item(),
                "weight_sigma_mean": weight_sigma.mean().item(),
                "weight_sigma_min": weight_sigma.min().item(),
                "weight_sigma_max": weight_sigma.max().item(),
                "bias_mu_mean": bias_mu.mean().item(),
                "bias_sigma_mean": bias_sigma.mean().item()
            })
    
    return stats


def print_parameter_statistics(stats, task_idx):
    print(f"\n=== Parameter Statistics for Task {task_idx+1} ===")
    
    # shared 
    print("\nShared Layers:")
    for i, layer_stats in enumerate(stats["shared_layers"]):
        print(f"  Layer {i+1}:")
        print(f"    Weight μ: mean={layer_stats['weight_mu_mean']:.4f}, std={layer_stats['weight_mu_std']:.4f}")
        print(f"    Weight μ range: [{layer_stats['weight_mu_min']:.4f}, {layer_stats['weight_mu_max']:.4f}]")
        print(f"    Weight σ: mean={layer_stats['weight_sigma_mean']:.4f}, range=[{layer_stats['weight_sigma_min']:.4f}, {layer_stats['weight_sigma_max']:.4f}]")
        print(f"    Bias μ mean: {layer_stats['bias_mu_mean']:.4f}, σ mean: {layer_stats['bias_sigma_mean']:.4f}")
    
    # task
    print(f"\nTask {task_idx+1} Specific Layers:")
    for i, layer_stats in enumerate(stats["task_layers"]):
        print(f"  Layer {i+1}:")
        print(f"    Weight μ: mean={layer_stats['weight_mu_mean']:.4f}, std={layer_stats['weight_mu_std']:.4f}")
        print(f"    Weight μ range: [{layer_stats['weight_mu_min']:.4f}, {layer_stats['weight_mu_max']:.4f}]")
        print(f"    Weight σ: mean={layer_stats['weight_sigma_mean']:.4f}, range=[{layer_stats['weight_sigma_min']:.4f}, {layer_stats['weight_sigma_max']:.4f}]")
        print(f"    Bias μ mean: {layer_stats['bias_mu_mean']:.4f}, σ mean: {layer_stats['bias_sigma_mean']:.4f}")


def compute_decoder_stats(model):
    # Todo
    total_weight_std = 0.0
    total_bias_std = 0.0
    weight_count = 0
    bias_count = 0
    
    task_network = model.decoder.task_layers[model.current_task]
    for layer in task_network:
        weight_std = layer._get_sigma(layer.weight_rho)
        bias_std = layer._get_sigma(layer.bias_rho)
        
        total_weight_std += weight_std.sum().item()
        total_bias_std += bias_std.sum().item()
        weight_count += weight_std.numel()
        bias_count += bias_std.numel()
    
    for layer in model.decoder.shared_layers:
        weight_std = layer._get_sigma(layer.weight_rho)
        bias_std = layer._get_sigma(layer.bias_rho)
        
        total_weight_std += weight_std.sum().item()
        total_bias_std += bias_std.sum().item()
        weight_count += weight_std.numel()
        bias_count += bias_std.numel()
    

    avg_weight_std = total_weight_std / weight_count if weight_count > 0 else 0.0
    avg_bias_std = total_bias_std / bias_count if bias_count > 0 else 0.0
    
    return {
        'decoder_weight_std': avg_weight_std,
        'decoder_bias_std': avg_bias_std,
        'decoder_param_count': weight_count + bias_count
    }


def create_task_samples_montage(model, device, output_dir, current_task):
    """Generate a row of samples from each task seen so far"""
    output_dir = Path(output_dir) / "samples"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(1, current_task + 1, figsize=((current_task + 1) * 2, 2))
    

    if current_task == 0:
        axes = [axes]
    
    for task_idx in range(current_task + 1):
        model.set_task(task_idx)
        

        with torch.no_grad():
            sample = model.sample(1, task_idx=task_idx)[0].cpu()
        

        ax = axes[task_idx]
        ax.imshow(sample.view(28, 28), cmap='gray')
        ax.set_title(f"Task {task_idx}")
        ax.axis('off')
    
    montage_path = output_dir / f"task{current_task+1}_samples.png"
    plt.tight_layout()
    plt.savefig(montage_path, dpi=200)
    plt.close(fig)
    
    return montage_path


def create_multiple_task_samples_montages(model, device, output_dir, current_task, num_montages=5):
    """
    Generate multiple montages (rows) of samples from each task seen so far.
    """
    output_dir = Path(output_dir) / "samples" / "multiple_means"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    montage_paths = []
    
    for montage_idx in range(num_montages):
        fig, axes = plt.subplots(1, current_task + 1, figsize=((current_task + 1) * 2, 2))
        
        if current_task == 0:
            axes = [axes]
        
        for task_idx in range(current_task + 1):
            model.set_task(task_idx)
            
            with torch.no_grad():
                sample = model.sample(1, task_idx=task_idx)[0].cpu()
            
            ax = axes[task_idx]
            ax.imshow(sample.view(28, 28), cmap='gray')
            ax.set_title(f"Task {task_idx}")
            ax.axis('off')
        
        montage_path = output_dir / f"task{current_task+1}_samples_montage{montage_idx+1}.png"
        plt.tight_layout()
        plt.savefig(montage_path, dpi=200)
        plt.close(fig)
        
        montage_paths.append(montage_path)
    
    return montage_paths


def create_single_sample_montages(model, device, output_dir, current_task, num_montages=5):
    """Generate montages using single parameter samples (not means)"""
    output_dir = Path(output_dir) / "samples" / "single_samples"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    montage_paths = []
    
    def sample_with_weight_sampling(model, n_samples, task_idx):
        z = torch.randn(n_samples, model.latent_size, device=device)
        with torch.no_grad():
            logits = model.decoder(z, task_idx)
            return torch.sigmoid(logits)
    
    for montage_idx in range(num_montages):
        fig, axes = plt.subplots(1, current_task + 1, figsize=((current_task + 1) * 2, 2))
        
        if current_task == 0:
            axes = [axes]
        
        for task_idx in range(current_task + 1):
            model.set_task(task_idx)
            
            with torch.no_grad():
                sample = sample_with_weight_sampling(model, 1, task_idx)[0].cpu()
            
            ax = axes[task_idx]
            ax.imshow(sample.view(28, 28), cmap='gray')
            ax.set_title(f"Task {task_idx}")
            ax.axis('off')
        
        montage_path = output_dir / f"task{current_task+1}_single_sample_montage{montage_idx+1}.png"
        plt.tight_layout()
        plt.savefig(montage_path, dpi=200)
        plt.close(fig)
        
        montage_paths.append(montage_path)
    
    return montage_paths


def create_multiple_samples_averaged_montages(model, device, output_dir, current_task, num_montages=5, samples_per_image=100):
    #Generate montages using multiple averaged samples for each image
    output_dir = Path(output_dir) / "samples" / "averaged_samples"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    montage_paths = []
    
    def sample_with_averaged_weights(model, task_idx, n_samples_per_image):
        z = torch.randn(1, model.latent_size, device=device)
        z = z.expand(n_samples_per_image, -1)  # Same latent vector for all samples
        
        with torch.no_grad():
            accumulated_probs = torch.zeros(1, model.input_size, device=device)
            
            # avg over multiple decoder passes
            for _ in range(n_samples_per_image):
                logits = model.decoder(z[0:1], task_idx)
                probs = torch.sigmoid(logits)
                accumulated_probs += probs
                
            final_image = accumulated_probs / n_samples_per_image
            return final_image
    
    for montage_idx in range(num_montages):
        fig, axes = plt.subplots(1, current_task + 1, figsize=((current_task + 1) * 2, 2))
        
        if current_task == 0:
            axes = [axes]
        
        for task_idx in range(current_task + 1):
            model.set_task(task_idx)
            
            with torch.no_grad():
                sample = sample_with_averaged_weights(model, task_idx, samples_per_image)[0].cpu()
            
            ax = axes[task_idx]
            ax.imshow(sample.view(28, 28), cmap='gray')
            ax.set_title(f"Task {task_idx}")
            ax.axis('off')
        
        montage_path = output_dir / f"task{current_task+1}_avg{samples_per_image}_montage{montage_idx+1}.png"
        plt.tight_layout()
        plt.savefig(montage_path, dpi=200)
        plt.close(fig)
        
        montage_paths.append(montage_path)
    
    return montage_paths


def create_reconstruction_visualizations(model, test_loaders, device, output_dir, current_task):
    """Create side-by-side visualizations of original images and reconstructions"""
    recon_dir = Path(output_dir) / f"recons_after_task{current_task+1}"
    recon_dir.mkdir(exist_ok=True, parents=True)
    
    saved_paths = {}
    
    # For each task seen so far
    for task_idx in range(current_task + 1):
        model.set_task(task_idx)
        model.eval()
        
        loader = test_loaders[task_idx]
        if loader is None:
            continue
            
        try:
            x_batch, _ = next(iter(loader))
            x_batch = x_batch.to(device)
            
            n_images = min(5, x_batch.size(0))
            x_batch = x_batch[:n_images]
            
            batch_size = x_batch.size(0)
            x_flat = x_batch.view(batch_size, -1)
            
            with torch.no_grad():
                reconstructions = model.reconstruct(x_flat, task_idx=task_idx)
            
            fig, axes = plt.subplots(n_images, 2, figsize=(4, n_images * 2))
            
            if n_images == 1:
                axes = [axes]
                
            for i in range(n_images):
                # Original 
                axes[i][0].imshow(x_batch[i].cpu().view(28, 28), cmap='gray')
                axes[i][0].set_title("Original")
                axes[i][0].axis('off')
                
                # recon
                axes[i][1].imshow(reconstructions[i].cpu().view(28, 28), cmap='gray')
                axes[i][1].set_title("Reconstruction")
                axes[i][1].axis('off')
            
            recon_path = recon_dir / f"reconstructions_task_{task_idx+1}.png"
            plt.tight_layout()
            plt.savefig(recon_path, dpi=200)
            plt.close(fig)
            
            saved_paths[task_idx] = recon_path
            
        except StopIteration:
            continue 
    
    return saved_paths


def importance_sampling_log_likelihood(model, x_batch, task_idx=0, n_samples=100, n_runs=50):
    """Estimate test log-likelihood using importance sampling"""
    model.eval()
    with torch.no_grad():
        batch_size = x_batch.size(0)
        all_ll_runs = []  
        
        for run in range(n_runs):
            
            x_expanded = x_batch.unsqueeze(1)                  
            x_tiled = x_expanded.expand(-1, n_samples, -1, -1, -1)  
            x_reshaped = x_tiled.contiguous().view(-1, *x_batch.shape[1:])  # [B*n_samples, C, H, W]
            
            # Get encoder outputs in one pass
            z_mu, z_logvar = model.encode(x_reshaped, task_idx)
            
   
            std = torch.exp(0.5 * z_logvar)
            eps = torch.randn_like(std)
            z = z_mu + eps * std
            
          
            log_q = (-0.5 * z_logvar - 0.5 * ((z - z_mu) / std).pow(2) - 0.5 * math.log(2 * math.pi)).sum(dim=1)
            
            # Compute log p(z
            log_p_z = (-0.5 * z.pow(2) - 0.5 * math.log(2 * math.pi)).sum(dim=1)
            
            #  decoder distribution
            x_mu = model.decode(z, task_idx)
            x_original = x_batch.unsqueeze(1).expand(batch_size, n_samples, *x_batch.shape[1:]).contiguous().view(-1, *x_batch.shape[1:])
            
  
            x_original_flat = x_original.view(x_original.size(0), -1)
            
  
            log_p_x_given_z = -F.binary_cross_entropy_with_logits(
                x_mu, x_original_flat, reduction='none'
            ).sum(dim=1)
            

            log_w = log_p_x_given_z + log_p_z - log_q  # [B*n_samples]
            

            log_w = log_w.view(batch_size, n_samples)
            max_log_w, _ = torch.max(log_w, dim=1, keepdim=True)  # For num stab
            log_w_shifted = log_w - max_log_w
            ll_estimate = max_log_w.squeeze(1) + torch.log(torch.mean(torch.exp(log_w_shifted), dim=1))
            
            all_ll_runs.append(ll_estimate)
        
        all_ll = torch.stack(all_ll_runs, dim=0)  # [n_runs, B]
        example_ll = torch.mean(all_ll, dim=0)    # [B]
        return torch.mean(example_ll).item()


def compute_classifier_uncertainty(model, classifier, task_idx=None, n_samples=1000, 
                                   batch_size=100, device='cuda'):
    """
    Compute classifier uncertainty metric on generated samples.
    """
    if task_idx is None:
        task_idx = model.current_task

    model.eval()
    classifier.eval()

    total_kl = 0.0
    samples_processed = 0
    remaining_samples = n_samples

    n_batches = math.ceil(n_samples / batch_size)

    with torch.no_grad():
        for _ in range(n_batches):
            current_batch_size = min(batch_size, remaining_samples)
            remaining_samples -= current_batch_size
            samples_processed += current_batch_size

            samples = model.sample(current_batch_size, task_idx=task_idx).to(device)
            
            logits = classifier(samples)
            probs = F.softmax(logits, dim=1)
            
            one_hot = torch.zeros_like(probs)
            one_hot[:, task_idx] = 1.0

            kl_batch = -torch.log(probs[:, task_idx] + 1e-10).mean()

            total_kl += kl_batch.item() * current_batch_size

    average_kl = total_kl / samples_processed if samples_processed > 0 else float('inf')
    return average_kl


def compute_reconstruction_error(model, data_loader, task_idx=None, device='cuda'):
    """
    Compute average reconstruction error on a dataset.
    """
    if task_idx is None:
        task_idx = model.current_task
        
    model.eval()
    
    total_error = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for x_batch, _ in data_loader:
            x_batch = x_batch.to(device)
            batch_size = x_batch.size(0)
            
            x_flat = x_batch.view(batch_size, -1)
            
            reconstructions = model.reconstruct(x_flat, task_idx=task_idx)
            
            mse = F.mse_loss(reconstructions, x_flat, reduction='sum')
            
            total_error += mse.item()
            total_samples += batch_size
            
    return total_error / total_samples


def evaluate_model_dgm(model, test_loaders, device, num_tasks_seen=None, 
                     classifier=None, n_ll_samples=5000, n_class_samples=1000):

    if num_tasks_seen is None:
        num_tasks_seen = len(test_loaders)
    else:
        num_tasks_seen = min(num_tasks_seen, len(test_loaders))
    
    results = []
    
    sum_ll = 0.0
    sum_recon_error = 0.0
    sum_cls_uncertainty = 0.0
    cls_count = 0
    
    for task_idx in tqdm(range(num_tasks_seen), desc="Evaluating tasks"):
        model.set_task(task_idx)
        
        loader = test_loaders[task_idx]
        
        task_results = {}
        
        all_batches = list(loader)
        x_sample, _ = all_batches[0]
        x_sample = x_sample.to(device)
        recon_error = compute_reconstruction_error(model, loader, task_idx=task_idx, device=device)
        task_results['recon_error'] = recon_error
        sum_recon_error += recon_error
        
        samples_per_eval = 100
        num_evals = max(1, n_ll_samples // samples_per_eval)
        
        total_ll = 0.0
        total_examples = 0
        
        # find  batch for ll estimation
        eval_batch = None
        for batch in all_batches:
            if len(batch[0]) >= 10:
                eval_batch = batch
                break
        
        if eval_batch is None:
            eval_batch = all_batches[-1]
        
        x_batch, _ = eval_batch
        x_batch = x_batch[:10].to(device)  # limit to 10 examples
        
        for _ in range(num_evals):
            batch_ll = importance_sampling_log_likelihood(
                model, x_batch, task_idx=task_idx, n_samples=samples_per_eval
            )
            total_ll += batch_ll * len(x_batch)
            total_examples += len(x_batch)
                
        avg_ll = total_ll / total_examples if total_examples > 0 else 0
        task_results['log_likelihood'] = avg_ll
        sum_ll += avg_ll
        
        if classifier is not None:
            cls_uncertainty = compute_classifier_uncertainty(
                model, classifier, task_idx=task_idx, 
                n_samples=n_class_samples, device=device
            )
            task_results['classifier_uncertainty'] = cls_uncertainty
            sum_cls_uncertainty += cls_uncertainty
            cls_count += 1
            
        results.append(task_results)
        
        clean_memory(device)
    
    avg_metrics = {
        'average_log_likelihood': sum_ll / num_tasks_seen if num_tasks_seen > 0 else 0,
        'average_recon_error': sum_recon_error / num_tasks_seen if num_tasks_seen > 0 else 0
    }
    
    if cls_count > 0:
        avg_metrics['average_classifier_uncertainty'] = sum_cls_uncertainty / cls_count
        
    results.append(avg_metrics)
    
    return results 