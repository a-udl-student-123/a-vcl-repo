# vae implementation with syn intelligence 

import torch
import torch.nn as nn
import torch.nn.functional as F

from z_synaptic.si import SynapticIntelligence
from z_models.generative.vcl_models_dgm import EncoderNetwork


class DeterministicDecoderNetwork(nn.Module):
    def __init__(self, latent_size=50, hidden_size=500, output_size=784, num_tasks=10):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_tasks = num_tasks
        
        # shared output  (500->500->784)
        self.shared_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, output_size)
        ])
        
        # track shared  for SI
        self.shared_params = []
        for i, layer in enumerate(self.shared_layers):
            for name, param in layer.named_parameters():
                prefixed_name = f"shared_layers.{i}.{name}"
                self.shared_params.append((prefixed_name, param))
        
        # create task layers directly
        self.task_layers = nn.ModuleList()
        for _ in range(num_tasks):
            self._create_task_layer()
        
        self.current_task = 0
        self._init_weights()
        
    def _init_weights(self):
        for layer in self.shared_layers:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)
    
    def _create_task_layer(self):
        task_specific_layers = nn.ModuleList([
            nn.Linear(self.latent_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size)
        ])
        
        for layer in task_specific_layers:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(layer.bias)
                
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


class SI_DGM_VAE(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, latent_size=50, num_tasks=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_tasks = num_tasks
        

        self.decoder = DeterministicDecoderNetwork(
            latent_size, hidden_size, input_size, num_tasks=num_tasks
        )
        
        # task-specific enc
        self.encoders = nn.ModuleList([
            EncoderNetwork(input_size, hidden_size, latent_size)
            for _ in range(num_tasks)
        ])
        
        self.current_task = 0
        self.si = None
    
    def set_si(self, lambda_reg=1.0, epsilon=1e-3, omega_decay=0.9, device='cuda'):
        self.si = SynapticIntelligence(
            self.decoder,
            lambda_reg=lambda_reg,
            epsilon=epsilon,
            omega_decay=omega_decay,
            device=device,
            shared_only=True
        )
        return self.si
    
    def set_task(self, task_idx):
        if task_idx >= self.num_tasks:
            raise ValueError(f"Task index {task_idx} out of range (0-{self.num_tasks-1})")
            
        self.current_task = task_idx
        self.decoder.set_task(task_idx)
    
    def encode(self, x, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task
            
        if task_idx >= len(self.encoders):
            raise ValueError(f"Task index {task_idx} out of range (0-{len(self.encoders)-1})")
            
        return self.encoders[task_idx](x)
    
    def decode(self, z, task_idx=None):
        return self.decoder(z, task_idx)
    
    def forward(self, x, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task
        
        
        z_mean, z_rho = self.encode(x, task_idx)
        
        # reparam trick
        eps = torch.randn_like(z_mean)
        z = z_mean + torch.exp(0.5 * z_rho) * eps
        
        logits = self.decode(z, task_idx)
        
        return logits, z_mean, z_rho
    
    def sample(self, num_samples=1, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task
            
        z = torch.randn(num_samples, self.latent_size, device=next(self.parameters()).device)
        
        with torch.no_grad():
            logits = self.decoder(z, task_idx)
            
        return torch.sigmoid(logits)
    
    def reconstruct(self, x, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task
            
        with torch.no_grad():
            z_mean, _ = self.encode(x, task_idx)
            logits = self.decoder(z_mean, task_idx)
            
        return torch.sigmoid(logits)
    
    def compute_loss(self, x, dataset_size, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task
            
        logits, z_mean, z_rho = self.forward(x, task_idx)
        
        x_flat = x.view(-1, self.input_size)
        bce_elementwise = F.binary_cross_entropy_with_logits(
            logits, x_flat, reduction='none'
        )
        bce_sum = bce_elementwise.sum(dim=1)
        recon_loss = -bce_sum.mean()
        
        # kl div
        kl_terms = 1 + z_rho - z_mean.pow(2) - torch.exp(z_rho)
        kl_sum = torch.sum(kl_terms, dim=1)
        kl_latent = -0.5 * kl_sum.mean()
        
        # si reg if not first task
        si_loss = self.si.compute_regularization_loss() if task_idx > 0 else torch.tensor(0.0, device=x.device)
        
        # total loss = -ELBO + SI reg
        loss = -recon_loss + kl_latent + si_loss
        
        return loss, {
            'recon_loss': recon_loss.item(),
            'kl_latent': kl_latent.item(),
            'si_loss': si_loss.item() if isinstance(si_loss, torch.Tensor) else si_loss,
            'total_loss': loss.item(),
            'elbo': recon_loss.item() - kl_latent.item()
        }
    
    def before_optimizer_step(self):
        if self.si is not None:
            self.si.before_step()
    
    def after_optimizer_step(self):
        if self.si is not None:
            self.si.accumulate_path_integral()
    
    def complete_task(self):
        if self.si is not None:
            self.si.update_omega() 