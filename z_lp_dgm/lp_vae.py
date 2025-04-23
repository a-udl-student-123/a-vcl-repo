

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

from z_synaptic_dgm.si_vae import DeterministicDecoderNetwork          # noqa
from z_models.generative.vcl_models_dgm import EncoderNetwork  


class LP_DGM_VAE(nn.Module):
    # multi-head VAE with LP regularization
    # only shared decoder layers get regularized  
    def __init__(self, input_size=784, hidden_size=500, latent_size=50,
                 num_tasks=10):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.num_tasks = num_tasks

        self.encoders = nn.ModuleList([
            EncoderNetwork(input_size, hidden_size, latent_size)
            for _ in range(num_tasks)
        ])
        self.decoder = DeterministicDecoderNetwork(
            latent_size, hidden_size, input_size, num_tasks
        )
        self.current_task = 0

        # store prev task info
        self.lp_data = []  

    def set_task(self, idx: int):
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

    def compute_loss(self, x, lp_lambda, dataset_size, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task
        logits, mu, rho = self.forward(x, task_idx)
        flat = x.view(-1, self.input_size)

        recon = F.binary_cross_entropy_with_logits(
            logits, flat, reduction="sum"
        ) / x.size(0)

        # kl for latents
        kl = -0.5 * torch.sum(1 + rho - mu.pow(2) - rho.exp()) / x.size(0)

        lp_pen, _ = self._lp_penalty(lp_lambda)

        loss = recon + kl + lp_pen
        return loss, {
            "recon_loss": recon.item(),
            "kl_latent": kl.item(),
            "lp_penalty": lp_pen.item(),
            "total_loss": loss.item(),
            "elbo": -(recon + kl).item()
        }

    def encode(self, x, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task
        
        if task_idx >= len(self.encoders):
            raise ValueError(f"Task index {task_idx} out of range (0-{len(self.encoders)-1})")
            
        return self.encoders[task_idx](x)
        
    def decode(self, z, task_idx=None):
        if task_idx is None:
            task_idx = self.current_task
            
        if task_idx >= self.num_tasks:
            raise ValueError(f"Task index {task_idx} out of range (0-{self.num_tasks-1})")
            
        return self.decoder(z, task_idx)

    def _target_params(self):
        # get shared decoder params only
        for n, p in self.decoder.shared_layers.named_parameters():
            yield f"decoder.shared_layers.{n}", p

    def _lp_penalty(self, lp_lambda):
        if not self.lp_data:
            return torch.tensor(0., device=next(self.parameters()).device), 0.
        pen = torch.zeros(1, device=next(self.parameters()).device)
        for H, θstar in self.lp_data:
            for n, p in self._target_params():
                pen += (H[n] * (p - θstar[n]).pow(2)).sum()
        lp_term = 0.5 * lp_lambda / len(self.lp_data) * pen
        return lp_term, pen

    def estimate_hessian(self, loader, device, n_samples=600, batch_size=64):
        # diagonal gauss-newton approx
        idx = torch.randperm(len(loader.dataset))[:n_samples].tolist()
        h_loader = DataLoader(loader.dataset, batch_size=batch_size,
                              sampler=SubsetRandomSampler(idx),
                              num_workers=getattr(loader, "num_workers", 0),
                              pin_memory=getattr(loader, "pin_memory", False))
        H = {n: torch.zeros_like(p, device=device) for n, p in self._target_params()}
        total = 0
        
        for x, _ in h_loader:
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
                if all(p.requires_grad for p in params):
                    grads = torch.autograd.grad(loss, params,
                                              retain_graph=False, create_graph=False)
                    for (n, _), g in zip(self._target_params(), grads):
                        H[n] += g.pow(2).detach() * b

        with torch.no_grad():
            for n in H:
                H[n] /= float(total)
                
        return H

    def register_lp_task(self, hessian):
        θ = {n: p.clone().detach() for n, p in self._target_params()}
        self.lp_data.append((hessian, θ))

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
            z_mean, _ = self.encoders[task_idx](x)
            logits = self.decoder(z_mean, task_idx)
            
        return torch.sigmoid(logits)
