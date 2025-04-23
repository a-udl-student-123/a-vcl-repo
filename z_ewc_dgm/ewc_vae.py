# VAE with EWC regularization for continual learning
# reuses encoder/decoder from LP variant but with EWC instead of LP

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

from z_synaptic_dgm.si_vae import DeterministicDecoderNetwork
from z_models.generative.vcl_models_dgm import EncoderNetwork

class EWC_DGM_VAE(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, latent_size=50,
                 num_tasks=10):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.num_tasks = num_tasks

        # task-specific encodres
        self.encoders = nn.ModuleList([
            EncoderNetwork(input_size, hidden_size, latent_size)
            for _ in range(num_tasks)
        ])
        # shared decoder
        self.decoder = DeterministicDecoderNetwork(
            latent_size, hidden_size, input_size, num_tasks
        )
        self.current_task = 0

        # store fisher diag and params for prev tasks
        self.ewc_data = []

    def set_task(self, idx):
        if not 0 <= idx < self.num_tasks:
            raise ValueError(f"task {idx} out of range")
        self.current_task = idx
        self.decoder.set_task(idx)

    def forward(self, x, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task
        mu, rho = self.encoders[task_idx](x)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * rho)
        logits = self.decoder(z, task_idx)
        return logits, mu, rho

    def compute_loss(self, x, ewc_lambda, dataset_size, task_idx=None):
        """negative ELBO + EWC penalty"""
        if task_idx is None:
            task_idx = self.current_task
        logits, mu, rho = self.forward(x, task_idx)
        flat = x.view(-1, self.input_size)

        # reconstruction term
        recon = F.binary_cross_entropy_with_logits(
            logits, flat, reduction="sum"
        ) / x.size(0)

        # kl term
        kl = -0.5 * torch.sum(1 + rho - mu.pow(2) - rho.exp()) / x.size(0)

        # ewc penalty
        ewc_pen, _ = self._ewc_penalty(ewc_lambda)

        loss = recon + kl + ewc_pen
        return loss, {
            "recon_loss": recon.item(),
            "kl_latent": kl.item(),
            "ewc_penalty": ewc_pen.item(),
            "total_loss": loss.item(),
            "elbo": -(recon + kl).item()
        }

    def sample(self, k=1, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task
        z = torch.randn(k, self.latent_size, device=next(self.parameters()).device)
        with torch.no_grad():
            logits = self.decoder(z, task_idx)
        return torch.sigmoid(logits)

    def reconstruct(self, x, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task
        with torch.no_grad():
            mu, _ = self.encoders[task_idx](x)
            logits = self.decoder(mu, task_idx)
        return torch.sigmoid(logits)

    def decode(self, z, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task
        
        if task_idx >= self.num_tasks:
            raise ValueError(f"Task index {task_idx} out of range (0-{self.num_tasks-1})")
        
        return self.decoder(z, task_idx)

    def _target_params(self):
        # only regularize shared decoder params
        for n, p in self.decoder.shared_layers.named_parameters():
            yield f"decoder.shared_layers.{n}", p

    def _ewc_penalty(self, ewc_lambda):
        if not self.ewc_data:
            return torch.tensor(0., device=next(self.parameters()).device), 0.
        pen = torch.zeros(1, device=next(self.parameters()).device)
        for fisher, theta_star in self.ewc_data:
            for n, p in self._target_params():
                pen += (fisher[n] * (p - theta_star[n]).pow(2)).sum()
        n_prev = len(self.ewc_data)
        eff_lambda = ewc_lambda / n_prev
        ewc_term = 0.5 * eff_lambda * pen
        return ewc_term, pen

    def estimate_fisher(self, loader, device, n_samples=600, batch_size=64):
        # get diagonal fisher estimate from negative ELBO gradients
        idx = torch.randperm(len(loader.dataset))[:n_samples].tolist()
        fisher_loader = DataLoader(
            loader.dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(idx),
            num_workers=getattr(loader, "num_workers", 0),
            pin_memory=getattr(loader, "pin_memory", False),
        )
        fisher_diag = {n: torch.zeros_like(p, device=device)
                 for n, p in self._target_params()}
        total = 0

        for x, _ in fisher_loader:
            x = x.to(device)
            b = x.size(0)
            total += b

            with torch.enable_grad():
                logits, mu, rho = self.forward(x)
                recon = F.binary_cross_entropy_with_logits(
                    logits, x.view(b, -1), reduction="sum"
                )
                kl = -0.5 * torch.sum(1 + rho - mu.pow(2) - rho.exp())
                loss = (recon + kl) / b

                params = [p for _, p in self._target_params()]
                grads = torch.autograd.grad(loss, params,
                                           retain_graph=False,
                                           create_graph=False)
                for (n, _), g in zip(self._target_params(), grads):
                    fisher_diag[n] += g.detach().pow(2) * b

        for n in fisher_diag:
            fisher_diag[n] /= float(total)
        return fisher_diag

    def register_ewc_task(self, fisher):
        # store fisher and params for completed task
        theta_star = {}
        for n, p in self._target_params():
            theta_star[n] = p.clone().detach()
        
        # Make sure fisher uses the same parameter names!
        aligned_fisher = {}
        for n in theta_star:
            if n in fisher:
                aligned_fisher[n] = fisher[n].to(next(self.parameters()).device)
            else:
                print(f"Warning: Parameter {n} not found in Fisher matrix")
                aligned_fisher[n] = torch.ones_like(theta_star[n]) * 1e-8
        
        self.ewc_data.append((aligned_fisher, theta_star))

    def encode(self, x, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task
        
        if task_idx >= len(self.encoders):
            raise ValueError(f"Task index {task_idx} out of range (0-{len(self.encoders)-1})")
        
        return self.encoders[task_idx](x)
