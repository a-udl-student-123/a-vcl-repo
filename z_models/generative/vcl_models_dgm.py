# VCL models for generative case

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

from z_models.vcl_models import MeanFieldLinear


class EncoderNetwork(nn.Module):
    """Task-specific encoder network, outputs parameters for q(z|x)."""
    def __init__(self, input_size=784, hidden_size=500, latent_size=50):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # 3 hidden layers + mean/var outputs
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, latent_size)
        self.fc_rho = nn.Linear(hidden_size, latent_size)
        
        self._init_weights()
        
    def _init_weights(self):
        # kaiming for relu layers, xavier for outputs
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_normal_(self.fc_mean.weight)
        nn.init.zeros_(self.fc_mean.bias)
        nn.init.xavier_normal_(self.fc_rho.weight)
        nn.init.zeros_(self.fc_rho.bias)
        
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(-1, self.input_size)
            
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        
        z_mean = self.fc_mean(h)
        z_rho = self.fc_rho(h)
        
        return z_mean, z_rho
        
    def sample_z(self, x):
        # sample latent z with reparam trick
        z_mean, z_rho = self.forward(x)
        eps = torch.randn_like(z_mean)
        z = z_mean + torch.exp(0.5 * z_rho) * eps
        return z, z_mean, z_rho


class DecoderNetwork(nn.Module):
    # decoder with task-specific first layers and shared output head
    def __init__(self, latent_size=50, hidden_size=500, output_size=784, init_std=0.001):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_std = init_std
        
        # shared output layers
        self.shared_layers = nn.ModuleList([
            MeanFieldLinear(hidden_size, hidden_size),
            MeanFieldLinear(hidden_size, output_size)
        ])
        
        self.task_layers = nn.ModuleList()
        self.current_task = 0
        
    def add_task_layer(self):
        # add task-specific layers (50->500->500)
        task_specific_layers = nn.ModuleList([
            MeanFieldLinear(self.latent_size, self.hidden_size),
            MeanFieldLinear(self.hidden_size, self.hidden_size)
        ])
        
        if hasattr(self, 'init_std') and self.init_std is not None:
            for layer in task_specific_layers:
                layer.set_init_std(self.init_std)
                
        self.task_layers.append(task_specific_layers)
        return len(self.task_layers) - 1
        
    def set_task(self, task_idx):
        if task_idx >= len(self.task_layers):
            raise ValueError(f"Task index {task_idx} out of range (0-{len(self.task_layers)-1})")
        self.current_task = task_idx
        
    def forward(self, z, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task
            
        if task_idx >= len(self.task_layers):
            raise ValueError(f"Task index {task_idx} out of range (0-{len(self.task_layers)-1})")
            
        task_network = self.task_layers[task_idx]
        h = F.relu(task_network[0](z))
        h = F.relu(task_network[1](h))
        
        h = F.relu(self.shared_layers[0](h))
        x_logits = self.shared_layers[1](h)
        
        return x_logits
        
    def forward_mean(self, z, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task
            
        if task_idx >= len(self.task_layers):
            raise ValueError(f"Task index {task_idx} out of range (0-{len(self.task_layers)-1})")
            
        task_network = self.task_layers[task_idx]
        h = F.relu(task_network[0].forward_mean(z))
        h = F.relu(task_network[1].forward_mean(h))
        
        h = F.relu(self.shared_layers[0].forward_mean(h))
        x_logits = self.shared_layers[1].forward_mean(h)
        
        return x_logits
        
    def store_params_as_old(self, task_idx=None):
        # store current params as old posterior / prior
        for layer in self.shared_layers:
            layer.store_params_as_old()
        
        if task_idx is None:
            task_idx = self.current_task
            
        if task_idx < len(self.task_layers):
            for layer in self.task_layers[task_idx]:
                layer.store_params_as_old()
        
    def kl_loss(self, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task
            
        kl = 0
        for layer in self.shared_layers:
            kl += layer.kl_to_old_posterior()
            
        if task_idx < len(self.task_layers):
            task_network = self.task_layers[task_idx]
            for layer in task_network:
                kl += layer.kl_to_old_posterior()
            
        return kl
        
    def reset_shared_variances(self, reset_std=0.0025):
        for layer in self.shared_layers:
            with torch.no_grad():
                fixed_rho = layer._get_rho(torch.tensor(reset_std))
                layer.weight_rho.fill_(fixed_rho)
                layer.bias_rho.fill_(fixed_rho)
        
    def set_init_std(self, init_std):
        self.init_std = init_std
        for layer in self.shared_layers:
            layer.set_init_std(init_std)
        for task_layers in self.task_layers:
            for layer in task_layers:
                layer.set_init_std(init_std)


class VCL_VAE(nn.Module):
    # vcl vae with task-specific components
    def __init__(self, input_size=784, hidden_size=500, latent_size=50, num_tasks=10, init_std=0.001):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_tasks = num_tasks
        self.init_std = init_std
        
        self.decoder = DecoderNetwork(latent_size, hidden_size, input_size, init_std=init_std)
        
        # one encoder per task
        self.encoders = nn.ModuleList([
            EncoderNetwork(input_size, hidden_size, latent_size)
            for _ in range(num_tasks)
        ])
        
        self.current_task = 0
        
    def set_task(self, task_idx):
        if task_idx >= self.num_tasks:
            raise ValueError(f"Task index {task_idx} out of range (0-{self.num_tasks-1})")
            
        self.current_task = task_idx
        
        if task_idx >= len(self.decoder.task_layers):
            self.decoder.add_task_layer()
            
        self.decoder.set_task(task_idx)
    
    def encode(self, x, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task
            
        if task_idx >= len(self.encoders):
            raise ValueError(f"Task index {task_idx} out of range (0-{len(self.encoders)-1})")
            
        return self.encoders[task_idx](x)
    
    def decode(self, z, task_idx=None):
        return self.decoder(z, task_idx)
    
    def forward(self, x, task_idx=None, n_samples=1):
        if task_idx is None:
            task_idx = self.current_task
        
        z_mean, z_rho = self.encode(x, task_idx)
        
        batch_size = x.size(0)
        accumulated_logits = torch.zeros(batch_size, self.input_size, device=x.device)
        
        # monte carlo estimate
        for _ in range(n_samples):
            eps = torch.randn_like(z_mean)
            z = z_mean + torch.exp(0.5 * z_rho) * eps
            logits = self.decode(z, task_idx)
            accumulated_logits += logits
            
        return accumulated_logits / n_samples, z_mean, z_rho
    
    def sample(self, num_samples=1, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task
            
        z = torch.randn(num_samples, self.latent_size, device=next(self.parameters()).device)
        
        with torch.no_grad():
            logits = self.decoder.forward_mean(z, task_idx)
            
        return torch.sigmoid(logits)
    
    def reconstruct(self, x, task_idx=None):
        # reconstruct x -> z -> x
        if task_idx is None:
            task_idx = self.current_task
            
        with torch.no_grad():
            z_mean, _ = self.encode(x, task_idx)
            logits = self.decoder.forward_mean(z_mean, task_idx)
            
        return torch.sigmoid(logits)
    
    def compute_elbo(self, x, dataset_size, task_idx=None, n_samples=1):
        if task_idx is None:
            task_idx = self.current_task
            
        logits, z_mean, z_rho = self.forward(x, task_idx, n_samples)
        
        x_flat = x.view(-1, self.input_size)
        
        # recon loss (binary cross entropy)
        bce_elementwise = F.binary_cross_entropy_with_logits(
            logits, x_flat, reduction='none'
        )
        bce_sum = bce_elementwise.sum(dim=1)
        recon_loss = -bce_sum.mean()
        
        # kl for latent vars
        kl_terms = 1 + z_rho - z_mean.pow(2) - torch.exp(z_rho)
        kl_sum = torch.sum(kl_terms, dim=1)
        kl_latent = -0.5 * kl_sum.mean()
        
        # kl for model params
        kl_params = self.decoder.kl_loss(task_idx)
        kl_params_scaled = kl_params / dataset_size
        
        elbo = recon_loss - kl_latent - kl_params_scaled
        
        return -elbo, {
            'recon_loss': recon_loss.item(),
            'kl_latent': kl_latent.item(),
            'kl_params': kl_params_scaled.item(),
            'elbo': elbo.item()
        }
    
    def store_params_as_old(self):
        self.decoder.store_params_as_old(self.current_task)
    
    def kl_loss(self):
        return self.decoder.kl_loss(self.current_task)
        
    def reset_shared_variances(self, reset_std=0.0025):
        self.decoder.reset_shared_variances(reset_std)
        
    def set_init_std(self, init_std):
        self.decoder.set_init_std(init_std)