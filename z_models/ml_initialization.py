

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from z_utils.utils import clean_memory, clean_loader
from torch.utils.data import DataLoader
from z_data.datasets import PermutedMNISTDataset, get_optimal_workers

class StandardMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=100, output_size=10):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, output_size)
        
        # init weights
        nn.init.kaiming_normal_(self.lin1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.lin1.bias)
        nn.init.kaiming_normal_(self.lin2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.lin2.bias)
        nn.init.xavier_normal_(self.lin3.weight)
        nn.init.zeros_(self.lin3.bias)
        
    def forward(self, x):
        h1 = torch.relu(self.lin1(x))
        h2 = torch.relu(self.lin2(h1))
        return self.lin3(h2)

class FlexibleStandardMLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=None, output_size=10):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [100, 100]  # default 2 hidden layers
            
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(layer.bias)
            else:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
                
        self.num_layers = len(self.layers)
        
    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < self.num_layers - 1:
                h = torch.relu(h)
        return h

def compute_gradient_norm(model):
    total_norm = 0.0
    total_params = 0
    
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm(2).item() ** 2
            total_params += 1
            
    return (total_norm / total_params) ** 0.5 if total_params > 0 else 0

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

def train_batch(model, inputs, targets, optimizer, criterion, device, compute_grad_norm=False):
    model.train()
    batch_size = inputs.size(0)
    
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    
    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    
    grad_norm = compute_gradient_norm(model) if compute_grad_norm else 0.0
    
    optimizer.step()
    
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == targets).sum().item()
    
    return {
        'loss': loss.item() * batch_size,
        'correct': correct,
        'total': batch_size,
        'grad_norm': grad_norm
    }

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = inputs.size(0)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == targets).sum().item()
            total += batch_size
    
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy

def train_standard_mlp(train_loader, test_loader=None, max_epochs=100, lr=5e-3, 
                       weight_decay=1e-5, patience=30, device='cuda', 
                       input_size=784, hidden_size=100, output_size=10, exp_dir=None):
    model = StandardMLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=patience//4, 
        min_lr=1e-6, threshold=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    best_state_dict = None
    best_train_acc = 0.0
    epochs_no_improve = 0
    
    ml_metrics = {'loss': [], 'accuracy': [], 'grad_norm': [], 'weight_abs': [], 'bias_abs': []}
    
    epoch_pbar = tqdm(range(1, max_epochs + 1), desc="ML Training", leave=False)
    for epoch in epoch_pbar:
        weight_abs, bias_abs = compute_model_stats(model)
        ml_metrics['weight_abs'].append(weight_abs)
        ml_metrics['bias_abs'].append(bias_abs)
        
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        grad_norms = []
        
        total_batches = len(train_loader)
        sampling_interval = max(1, int(total_batches / 10))
        
        batch_pbar = tqdm(enumerate(train_loader), total=total_batches, desc="Batches", leave=False)
        for batch_idx, (inputs, targets) in batch_pbar:
            compute_grad = (batch_idx % sampling_interval == 0)
            
            batch_metrics = train_batch(model, inputs, targets, optimizer, criterion, device, compute_grad_norm=compute_grad)
            epoch_loss += batch_metrics['loss']
            correct += batch_metrics['correct']
            total += batch_metrics['total']

            batch_acc = batch_metrics['correct'] / batch_metrics['total'] if batch_metrics['total'] > 0 else 0
            if compute_grad:
                grad_norms.append(batch_metrics['grad_norm'])
                batch_pbar.set_postfix({
                    "loss": f"{batch_metrics['loss']/batch_metrics['total']:.4f}",
                    "acc": f"{batch_acc:.4f}",
                    "grad": f"{batch_metrics['grad_norm']:.4f}"
                })
        
        train_loss = epoch_loss / total if total > 0 else 0
        train_acc = correct / total if total > 0 else 0
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
        
        ml_metrics['loss'].append(train_loss)
        ml_metrics['accuracy'].append(train_acc)
        ml_metrics['grad_norm'].append(avg_grad_norm)
        
        if test_loader:
            val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
            scheduler.step(val_loss)
            epoch_pbar.set_postfix({
                "val_acc": f"{val_acc:.4f}",
                "grad_norm": f"{avg_grad_norm:.4f}",
                "w_abs": f"{weight_abs:.4f}"
            })
            
            improved = val_loss < best_loss * 0.9999
            current_loss = val_loss
        else:
            epoch_pbar.set_postfix({
                "train_acc": f"{train_acc:.4f}",
                "grad_norm": f"{avg_grad_norm:.4f}",
                "w_abs": f"{weight_abs:.4f}"
            })
            
            improved = train_loss < best_loss * 0.9999
            current_loss = train_loss
        
        if improved:
            best_loss = current_loss
            best_state_dict = model.state_dict().copy()
            best_train_acc = train_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            break
    
    if best_state_dict:
        model.load_state_dict(best_state_dict)
    
    avg_weight_abs, avg_bias_abs = compute_model_stats(model)
    print("\n=== ML Initialization Model Stats ===")
    print(f"avg: weight={avg_weight_abs:.6f}, bias={avg_bias_abs:.6f}")
    
    if test_loader:
        final_loss, final_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"Training: acc={best_train_acc:.4f} | Test: acc={final_acc:.4f}, CE={final_loss:.4f}")
    else:
        print(f"Training acc: {best_train_acc:.4f}")
    
    try:
        from z_utils.plotting_utils import create_ml_training_plots
        create_ml_training_plots(ml_metrics, exp_dir)
    except Exception as e:
        print(f"Error creating ML training plots: {str(e)}")
    
    return model

def train_flexible_standard_mlp(train_loader, test_loader=None, max_epochs=100, lr=5e-3, 
                              weight_decay=1e-5, patience=30, device='cuda', 
                              input_size=784, hidden_sizes=None, output_size=10, exp_dir=None):
    """Train flexible (variable # of hidden layers) standard MLP with configurable hidden layers for ML initialization."""
    model = FlexibleStandardMLP(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=patience//4, 
        min_lr=1e-6, threshold=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    best_state_dict = None
    best_train_acc = 0.0
    epochs_no_improve = 0
    
    ml_metrics = {'loss': [], 'accuracy': [], 'grad_norm': [], 'weight_abs': [], 'bias_abs': []}
    
    epoch_pbar = tqdm(range(1, max_epochs + 1), desc="ML Training", leave=False)
    for epoch in epoch_pbar:
        weight_abs, bias_abs = compute_model_stats(model)
        ml_metrics['weight_abs'].append(weight_abs)
        ml_metrics['bias_abs'].append(bias_abs)
        
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        grad_norms = []
        
        total_batches = len(train_loader)
        sampling_interval = max(1, int(total_batches / 10))
        
        batch_pbar = tqdm(enumerate(train_loader), total=total_batches, desc="Batches", leave=False)
        for batch_idx, (inputs, targets) in batch_pbar:
            compute_grad = (batch_idx % sampling_interval == 0)
            
            batch_metrics = train_batch(model, inputs, targets, optimizer, criterion, device, compute_grad_norm=compute_grad)
            epoch_loss += batch_metrics['loss']
            correct += batch_metrics['correct']
            total += batch_metrics['total']

            batch_acc = batch_metrics['correct'] / batch_metrics['total'] if batch_metrics['total'] > 0 else 0
            if compute_grad:
                grad_norms.append(batch_metrics['grad_norm'])
                batch_pbar.set_postfix({
                    "loss": f"{batch_metrics['loss']/batch_metrics['total']:.4f}",
                    "acc": f"{batch_acc:.4f}",
                    "grad": f"{batch_metrics['grad_norm']:.4f}"
                })
        
        train_loss = epoch_loss / total if total > 0 else 0
        train_acc = correct / total if total > 0 else 0
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
        
        ml_metrics['loss'].append(train_loss)
        ml_metrics['accuracy'].append(train_acc)
        ml_metrics['grad_norm'].append(avg_grad_norm)
        
        if test_loader:
            val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
            scheduler.step(val_loss)
            epoch_pbar.set_postfix({
                "val_acc": f"{val_acc:.4f}",
                "grad_norm": f"{avg_grad_norm:.4f}",
                "w_abs": f"{weight_abs:.4f}"
            })
            
            improved = val_loss < best_loss * 0.9999
            current_loss = val_loss
        else:
            epoch_pbar.set_postfix({
                "train_acc": f"{train_acc:.4f}",
                "grad_norm": f"{avg_grad_norm:.4f}",
                "w_abs": f"{weight_abs:.4f}"
            })
            
            improved = train_loss < best_loss * 0.9999
            current_loss = train_loss
        
        if improved:
            best_loss = current_loss
            best_state_dict = model.state_dict().copy()
            best_train_acc = train_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            break
    
    if best_state_dict:
        model.load_state_dict(best_state_dict)
    
    avg_weight_abs, avg_bias_abs = compute_model_stats(model)
    print("\n=== ML Initialization Model Stats ===")
    print(f"avg: weight={avg_weight_abs:.6f}, bias={avg_bias_abs:.6f}")
    print(f"Number of layers: 1 input + {len(model.layers) - 1} (hidden+output)")
    
    if test_loader:
        final_loss, final_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"Training: acc={best_train_acc:.4f} | Test: acc={final_acc:.4f}, CE={final_loss:.4f}")
    else:
        print(f"Training acc: {best_train_acc:.4f}")
    
    try:
        from z_utils.plotting_utils import create_ml_training_plots
        create_ml_training_plots(ml_metrics, exp_dir)
    except Exception as e:
        print(f"Error creating ML training plots: {str(e)}")
    
    return model

def initialize_vcl_from_ml(vcl_model, ml_model, init_std=0.001, adaptive_std=False, adaptive_std_epsilon=0.01, device='cuda'):
    """Initialize VCL model with parameters from ML-trained model."""
    vcl_model = vcl_model.to(device)
    
    with torch.no_grad():
        for i in range(1, 4):
            vcl_layer = getattr(vcl_model, f"lin{i}")
            ml_layer = getattr(ml_model, f"lin{i}")
            
            vcl_layer.weight_mu.copy_(ml_layer.weight.to(device))
            vcl_layer.bias_mu.copy_(ml_layer.bias.to(device))
    
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

def initialize_flexible_vcl_from_ml(vcl_model, ml_model, init_std=0.001, adaptive_std=False, adaptive_std_epsilon=0.01, device='cuda'):
    """Initialize flexible VCL model with parameters from flexible MLtrained model."""
    vcl_model = vcl_model.to(device)
    
    if hasattr(vcl_model, 'layers') and hasattr(ml_model, 'layers'):
        assert len(vcl_model.layers) == len(ml_model.layers), "Layer count mismatch between VCL and ML models"
        
        with torch.no_grad():
            for i, (vcl_layer, ml_layer) in enumerate(zip(vcl_model.layers, ml_model.layers)):
                vcl_layer.weight_mu.copy_(ml_layer.weight.to(device))
                vcl_layer.bias_mu.copy_(ml_layer.bias.to(device))
                
        print(f"Copied weights from ML model to VCL model ({len(vcl_model.layers)} layers)")
        
    elif hasattr(vcl_model, 'shared_layers') and hasattr(ml_model, 'layers'):
        assert len(vcl_model.shared_layers) == len(ml_model.layers) - 1, "Layer count mismatch between VCL shared layers and ML model"
        
        with torch.no_grad():
            for i, (vcl_layer, ml_layer) in enumerate(zip(vcl_model.shared_layers, ml_model.layers[:-1])):
                vcl_layer.weight_mu.copy_(ml_layer.weight.to(device))
                vcl_layer.bias_mu.copy_(ml_layer.bias.to(device))
            
            vcl_model.heads[0].weight_mu.copy_(ml_model.layers[-1].weight.to(device))
            vcl_model.heads[0].bias_mu.copy_(ml_layer.bias.to(device))
            
        print(f"Copied weights from ML model to VCL multi-head model ({len(vcl_model.shared_layers)} shared layers + first head)")
        
    else:
        raise ValueError("Models are not compatible for flexible initialization")
    
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

def create_different_perm_loaders(train_loader, test_loader, device='cpu'):
    init_permutation = torch.randperm(784, device=device)
    print(f"Generated special permutation for ML initialization")
    
    init_dataset = PermutedMNISTDataset(
        root='data/',
        permutation=init_permutation,
        train=True,
        download=True
    )
    
    batch_size = train_loader.batch_size
    original_num_workers = getattr(train_loader, 'num_workers', 4)
    num_workers, persistent_workers, prefetch_factor = get_optimal_workers(original_num_workers, batch_size)
    
    init_train_loader = DataLoader(
        init_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if persistent_workers else None,
        drop_last=False
    )
    
    init_test_loader = None
    if test_loader is not None:
        init_test_dataset = PermutedMNISTDataset(
            root='data/',
            permutation=init_permutation,
            train=False,
            download=True
        )
        
        init_test_loader = DataLoader(
            init_test_dataset, 
            batch_size=test_loader.batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if persistent_workers else None,
            drop_last=False
        )
    
    return init_train_loader, init_test_loader

def get_ml_initialized_vcl_model(train_loader, vcl_model, test_loader=None, ml_epochs=100, lr=5e-3, 
                                init_std=0.001, adaptive_std=False, adaptive_std_epsilon=0.01,
                                device='cuda', exp_dir=None, different_perm=False):
    # check if flexible model
    is_flexible = hasattr(vcl_model, 'layers') or hasattr(vcl_model, 'shared_layers')
    
    # get model dims
    if hasattr(vcl_model, 'layers'):
        input_size = vcl_model.layers[0].in_features
        output_size = vcl_model.layers[-1].out_features
        hidden_sizes = [layer.out_features for layer in vcl_model.layers[:-1]]
    elif hasattr(vcl_model, 'shared_layers'):
        input_size = vcl_model.shared_layers[0].in_features
        output_size = vcl_model.heads[0].out_features
        hidden_sizes = [layer.out_features for layer in vcl_model.shared_layers]
    else:
        input_size = vcl_model.lin1.in_features
        hidden_size = vcl_model.lin1.out_features
        output_size = vcl_model.lin3.out_features if hasattr(vcl_model, 'lin3') else vcl_model.heads[0].out_features
    
    if different_perm:
        print("\nUsing ML initialization with a DIFFERENT permutation than Task #1...")
        init_train_loader, init_test_loader = create_different_perm_loaders(train_loader, test_loader, device='cpu')
    else:
        init_train_loader, init_test_loader = train_loader, test_loader
    
    if is_flexible:
        print(f"\nTraining flexible MLP ({len(hidden_sizes)} hidden layers) for initialization...")
        ml_model = train_flexible_standard_mlp(
            init_train_loader, 
            test_loader=init_test_loader, 
            max_epochs=ml_epochs, 
            lr=lr,
            weight_decay=1e-5,
            patience=30,
            device=device,
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            exp_dir=exp_dir
        )
        
        initialized_vcl_model = initialize_flexible_vcl_from_ml(
            vcl_model, 
            ml_model, 
            init_std=init_std,
            adaptive_std=adaptive_std,
            adaptive_std_epsilon=adaptive_std_epsilon,
            device=device
        )
    else:
        hidden_size = vcl_model.lin1.out_features
        ml_model = train_standard_mlp(
            init_train_loader, 
            test_loader=init_test_loader, 
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
        
        initialized_vcl_model = initialize_vcl_from_ml(
            vcl_model, 
            ml_model, 
            init_std=init_std,
            adaptive_std=adaptive_std,
            adaptive_std_epsilon=adaptive_std_epsilon,
            device=device
        )
    
    del ml_model
    
    if different_perm:
        clean_loader(init_train_loader)
        if init_test_loader:
            clean_loader(init_test_loader)
        del init_train_loader, init_test_loader
    
    clean_memory(device)
    
    initialized_vcl_model.train()
    
    return initialized_vcl_model