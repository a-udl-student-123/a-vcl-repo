# ml initialization for gaussian vcl - trains standard net and transfers params

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import gc

from z_gaussian.utils_gaussian import gaussian_nll_loss, convert_to_onehot, compute_rmse, compute_gradient_norm

class HeteroscedasticStandardMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=100, output_size=10):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, output_size)
        self.logvar_head = nn.Linear(hidden_size, output_size)
        
        # kaiming init for hidden layers
        nn.init.kaiming_normal_(self.lin1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.lin1.bias)
        nn.init.kaiming_normal_(self.lin2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.lin2.bias)
        nn.init.xavier_normal_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        
        # small weights for logvar head
        nn.init.xavier_normal_(self.logvar_head.weight, gain=0.01)
        nn.init.constant_(self.logvar_head.bias, -4.6) # log(0.1^2)
        
    def forward(self, x):
        h1 = torch.relu(self.lin1(x))
        h2 = torch.relu(self.lin2(h1))
        mean = self.mean_head(h2)
        logvar = self.logvar_head(h2)
        return mean, logvar

def clean_loader(loader):
    if loader is None:
        return
        
    if hasattr(loader, '_iterator') and loader._iterator is not None:
        try:
            loader._iterator._shutdown_workers()
        except Exception as e:
            print(f"Warning: Error shutting down dataloader workers: {str(e)}")
    
    try:
        if hasattr(loader, '_iterator'):
            loader._iterator = None
    except Exception as e:
        print(f"Warning: Error clearing dataloader refs: {str(e)}")
    
    gc.collect()

def clean_memory(device='cuda', sleep_time=0):
    gc.collect()
    
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if sleep_time > 0:
        import time
        time.sleep(sleep_time)
        gc.collect()

def compute_model_stats(model):
    weight_abs = 0.0
    weight_count = 0
    bias_abs = 0.0
    bias_count = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_abs += torch.abs(param).sum().item()
            weight_count += param.numel()
        elif 'bias' in name:
            bias_abs += torch.abs(param).sum().item()
            bias_count += param.numel()
    
    avg_weight = weight_abs / weight_count if weight_count > 0 else 0
    avg_bias = bias_abs / bias_count if bias_count > 0 else 0
    
    return avg_weight, avg_bias

def train_batch_gaussian(model, inputs, targets, optimizer, device, compute_grad_norm=False):
    model.train()
    batch_size = inputs.size(0)
    
    inputs = inputs.to(device, non_blocking=True)
    
    if targets.dim() == 1:
        targets = convert_to_onehot(targets, num_classes=model.mean_head.out_features)
    targets = targets.to(device, non_blocking=True)
    
    optimizer.zero_grad(set_to_none=True)
    mean, logvar = model(inputs)
    
    loss = gaussian_nll_loss(mean, logvar, targets)
    loss.backward()
    
    grad_norm = compute_gradient_norm(model) if compute_grad_norm else 0.0
    
    optimizer.step()
    
    rmse = compute_rmse(mean, targets)
    
    return {
        'loss': loss.item() * batch_size,
        'rmse': rmse,
        'total': batch_size,
        'grad_norm': grad_norm
    }

def evaluate_model_gaussian(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            
            if targets.dim() == 1:
                targets = convert_to_onehot(targets, num_classes=model.mean_head.out_features)
            targets = targets.to(device, non_blocking=True)
            
            batch_size = inputs.size(0)
            
            mean, logvar = model(inputs)
            loss = gaussian_nll_loss(mean, logvar, targets)
            
            mse = ((mean - targets) ** 2).mean().item()
            
            total_loss += loss.item() * batch_size
            total_mse += mse * batch_size
            total += batch_size
    
    avg_loss = total_loss / total if total > 0 else 0
    avg_mse = total_mse / total if total > 0 else 0
    rmse = avg_mse ** 0.5
    
    return avg_loss, rmse

def train_gaussian_standard_mlp(train_loader, test_loader=None, max_epochs=100, lr=5e-3, 
                            weight_decay=1e-5, patience=30, device='cuda', 
                            input_size=784, hidden_size=100, output_size=10, exp_dir=None):

    model = HeteroscedasticStandardMLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=patience//4, 
        min_lr=1e-6, threshold=1e-4
    )
    
    best_loss = float('inf')
    best_state_dict = None
    best_train_rmse = 0.0
    epochs_no_improve = 0
    
    ml_metrics = {'loss': [], 'rmse': [], 'grad_norm': [], 'weight_abs': [], 'bias_abs': []}
    
    epoch_pbar = tqdm(range(1, max_epochs + 1), desc="ML Training", leave=False)
    for epoch in epoch_pbar:
        weight_abs, bias_abs = compute_model_stats(model)
        ml_metrics['weight_abs'].append(weight_abs)
        ml_metrics['bias_abs'].append(bias_abs)
        
        model.train()
        epoch_loss = 0.0
        total_rmse = 0.0
        total = 0
        grad_norms = []
        
        total_batches = len(train_loader)
        sampling_interval = max(1, int(total_batches / 10))
        
        batch_pbar = tqdm(enumerate(train_loader), total=total_batches, desc="Batches", leave=False)
        for batch_idx, (inputs, targets) in batch_pbar:
            compute_grad = (batch_idx % sampling_interval == 0)
            
            batch_metrics = train_batch_gaussian(model, inputs, targets, optimizer, device, compute_grad_norm=compute_grad)
            epoch_loss += batch_metrics['loss']
            total_rmse += batch_metrics['rmse'] * batch_metrics['total']
            total += batch_metrics['total']

            batch_rmse = batch_metrics['rmse']
            if compute_grad:
                grad_norms.append(batch_metrics['grad_norm'])
                batch_pbar.set_postfix({
                    "loss": f"{batch_metrics['loss']/batch_metrics['total']:.4f}",
                    "rmse": f"{batch_rmse:.4f}",
                    "grad": f"{batch_metrics['grad_norm']:.4f}"
                })
        
        train_loss = epoch_loss / total if total > 0 else 0
        train_rmse = total_rmse / total if total > 0 else 0
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
        
        ml_metrics['loss'].append(train_loss)
        ml_metrics['rmse'].append(train_rmse)
        ml_metrics['grad_norm'].append(avg_grad_norm)
        
        if test_loader:
            val_loss, val_rmse = evaluate_model_gaussian(model, test_loader, device)
            scheduler.step(val_loss)
            epoch_pbar.set_postfix({
                "val_rmse": f"{val_rmse:.4f}",
                "grad_norm": f"{avg_grad_norm:.4f}",
                "w_abs": f"{weight_abs:.4f}"
            })
            
            improved = val_loss < best_loss * 0.9999
            current_loss = val_loss
        else:
            epoch_pbar.set_postfix({
                "train_rmse": f"{train_rmse:.4f}",
                "grad_norm": f"{avg_grad_norm:.4f}",
                "w_abs": f"{weight_abs:.4f}"
            })
            
            improved = train_loss < best_loss * 0.9999
            current_loss = train_loss
        
        if improved:
            best_loss = current_loss
            best_state_dict = model.state_dict().copy()
            best_train_rmse = train_rmse
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            break
    
    if best_state_dict:
        model.load_state_dict(best_state_dict)
    
    avg_weight_abs, avg_bias_abs = compute_model_stats(model)

    print(f"avg: weight={avg_weight_abs:.6f}, bias={avg_bias_abs:.6f}")
    
    if test_loader:
        final_loss, final_rmse = evaluate_model_gaussian(model, test_loader, device)
        print(f"Training: rmse={best_train_rmse:.4f} | Test: rmse={final_rmse:.4f}, NLL={final_loss:.4f}")
    else:
        print(f"Training rmse: {best_train_rmse:.4f}")
    
    try:
        from z_utils.plotting_utils import create_ml_training_plots
        if exp_dir:
            renamed_metrics = {
                'loss': ml_metrics['loss'],
                'accuracy': ml_metrics['rmse'],
                'grad_norm': ml_metrics['grad_norm'],
                'weight_abs': ml_metrics['weight_abs'],
                'bias_abs': ml_metrics['bias_abs']
            }
            create_ml_training_plots(renamed_metrics, exp_dir)
    except Exception as e:
        print(f"Error creating ML training plots: {str(e)}")
    
    return model

def initialize_vcl_from_ml_gaussian(vcl_model, ml_model, init_std=0.001, adaptive_std=False, adaptive_std_epsilon=0.01, device='cuda'):
    vcl_model = vcl_model.to(device)
    
    with torch.no_grad():
        # copy weights from ml model
        vcl_model.lin1.weight_mu.copy_(ml_model.lin1.weight.to(device))
        vcl_model.lin1.bias_mu.copy_(ml_model.lin1.bias.to(device))
        vcl_model.lin2.weight_mu.copy_(ml_model.lin2.weight.to(device))
        vcl_model.lin2.bias_mu.copy_(ml_model.lin2.bias.to(device))
        
        vcl_model.mean_head.weight_mu.copy_(ml_model.mean_head.weight.to(device))
        vcl_model.mean_head.bias_mu.copy_(ml_model.mean_head.bias.to(device))
        vcl_model.logvar_head.weight_mu.copy_(ml_model.logvar_head.weight.to(device))
        vcl_model.logvar_head.bias_mu.copy_(ml_model.logvar_head.bias.to(device))
    
    vcl_model.set_init_std(init_std, adaptive_std, adaptive_std_epsilon)
    
    if not adaptive_std:
        print(f"VCL initialization - fixed std: {init_std:.6f}")
    else:
        actual_scale = getattr(vcl_model, '_last_global_scale', None)
        if actual_scale is not None:
            print(f"VCL initialization - adaptive std: target={init_std:.6f}, global_scale={actual_scale:.6f}, epsilon={adaptive_std_epsilon:.6f}")
        else:
            print(f"VCL initialization - adaptive std: target={init_std:.6f}, epsilon={adaptive_std_epsilon:.6f}")
    
    return vcl_model

def get_ml_initialized_gaussian_vcl_model(train_loader, vcl_model, test_loader=None, ml_epochs=100, lr=5e-3, 
                                       init_std=0.001, adaptive_std=False, adaptive_std_epsilon=0.01,
                                       device='cuda', exp_dir=None, different_perm=False):
    input_size = vcl_model.lin1.in_features
    hidden_size = vcl_model.lin1.out_features
    output_size = vcl_model.mean_head.out_features
    
    init_train_loader = train_loader
    if different_perm:
        try:
            from z_data.datasets import generate_permutations, PermutedMNISTDataset
            from torch.utils.data import DataLoader
            
            print("\nUsing different permutation for ML initialization")
            
            init_perm = generate_permutations(num_tasks=1, device='cpu')[0]
            
            original_batch_size = train_loader.batch_size
            original_workers = train_loader.num_workers
            original_shuffle = train_loader.sampler is not None and hasattr(train_loader.sampler, 'shuffle') and train_loader.sampler.shuffle
            
            persistent_workers = getattr(train_loader, 'persistent_workers', True)
            prefetch_factor = getattr(train_loader, 'prefetch_factor', 2)
            
            dataset = PermutedMNISTDataset(
                root='data/',
                permutation=init_perm,
                train=True, 
                download=True
            )
            
            loader_params = {
                'batch_size': original_batch_size,
                'shuffle': original_shuffle,
                'num_workers': original_workers,
                'pin_memory': True,
                'drop_last': False,
                'persistent_workers': persistent_workers,
                'prefetch_factor': prefetch_factor
            }
            
            init_train_loader = DataLoader(dataset, **loader_params)
            
            print(f"Created initialization loader with different permutation (batch size {original_batch_size}, workers {original_workers})")
        except Exception as e:
            print(f"\nWarning: Failed to create different permutation for initialization: {str(e)}")
            print("Using standard initialization instead")
    
    print(f"\nTraining Gaussian MLP for init...")
    ml_model = train_gaussian_standard_mlp(
        init_train_loader, 
        test_loader=test_loader, 
        max_epochs=ml_epochs, 
        lr=lr,
        weight_decay=1e-5,
        patience=30,
        device=device,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        exp_dir=exp_dir
    )
    
    if different_perm and init_train_loader != train_loader:
        clean_loader(init_train_loader)
    
    initialized_vcl_model = initialize_vcl_from_ml_gaussian(
        vcl_model, 
        ml_model, 
        init_std=init_std,
        adaptive_std=adaptive_std,
        adaptive_std_epsilon=adaptive_std_epsilon,
        device=device
    )
    
    del ml_model
    clean_memory(device)
    
    initialized_vcl_model.train()
    
    return initialized_vcl_model 