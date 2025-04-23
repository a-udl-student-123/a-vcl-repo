

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from z_models.vcl_models import MeanFieldLinear

class HeteroscedasticVCL_MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=100, output_size=10):
        super().__init__()
        self.lin1 = MeanFieldLinear(input_size, hidden_size)
        self.lin2 = MeanFieldLinear(hidden_size, hidden_size)
        
        # separate heads for mean and variance
        self.mean_head = MeanFieldLinear(hidden_size, output_size)
        self.logvar_head = MeanFieldLinear(hidden_size, output_size)

    def forward(self, x, n_samples=1):
        batch_size = x.size(0)
        
        accumulated_means = torch.zeros(batch_size, self.mean_head.out_features, device=x.device)
        accumulated_logvars = torch.zeros(batch_size, self.logvar_head.out_features, device=x.device)
        
        # store all samples for uncertainty estimation
        all_means = torch.zeros(n_samples, batch_size, self.mean_head.out_features, device=x.device)
        all_logvars = torch.zeros(n_samples, batch_size, self.logvar_head.out_features, device=x.device)
        
        for i in range(n_samples):
            h1 = F.relu(self.lin1(x))
            h2 = F.relu(self.lin2(h1))
            
            means = self.mean_head(h2)
            logvars = torch.clamp(self.logvar_head(h2), min=-10.0, max=10.0)
            
            accumulated_means += means
            accumulated_logvars += logvars
            
            all_means[i] = means
            all_logvars[i] = logvars
        
        mean_prediction = accumulated_means / n_samples
        logvar_prediction = accumulated_logvars / n_samples
        aleatoric_uncertainty = torch.exp(all_logvars).mean(dim=0)
        
        # variance across samples = epistemic uncertainty
        epistemic_uncertainty = torch.var(all_means, dim=0)
        
        return mean_prediction, logvar_prediction, aleatoric_uncertainty, epistemic_uncertainty
    
    def predict_mean_var(self, x, n_samples=100):
        means, logvars = self.forward(x, n_samples=n_samples)
        return means, torch.exp(logvars)
        
    def store_params_as_old(self):
        self.lin1.store_params_as_old()
        self.lin2.store_params_as_old()
        self.mean_head.store_params_as_old()
        self.logvar_head.store_params_as_old()

    def kl_loss(self):
        kl1 = self.lin1.kl_to_old_posterior()
        kl2 = self.lin2.kl_to_old_posterior()
        kl_mean = self.mean_head.kl_to_old_posterior()
        kl_logvar = self.logvar_head.kl_to_old_posterior()
        
        return kl1 + kl2 + kl_mean + kl_logvar
        
    def set_init_std(self, init_std, adaptive=False, adaptive_std_epsilon=0.01):
        if not adaptive:
            self.lin1.set_init_std(init_std)
            self.lin2.set_init_std(init_std)
            self.mean_head.set_init_std(init_std)
            self.logvar_head.set_init_std(init_std)
            return
            
        # adaptive case - scale based on param magnitudes
        param_magnitudes = []
        layers = [self.lin1, self.lin2, self.mean_head, self.logvar_head]
        
        for layer in layers:
            param_magnitudes.append(torch.abs(layer.weight_mu).flatten())
            param_magnitudes.append(torch.abs(layer.bias_mu).flatten())
            
        all_params = torch.cat(param_magnitudes)
        param_values = all_params + adaptive_std_epsilon
        param_values = torch.clamp(param_values, min=1e-6, max=2.0)
        
        global_avg = param_values.mean().item()
        global_scale = init_std / global_avg
        self._last_global_scale = global_scale
        
        for layer in layers:
            weight_std = (torch.abs(layer.weight_mu) + adaptive_std_epsilon) * global_scale
            bias_std = (torch.abs(layer.bias_mu) + adaptive_std_epsilon) * global_scale
            
            weight_std = torch.clamp(weight_std, min=1e-6, max=2.0)
            bias_std = torch.clamp(bias_std, min=1e-6, max=2.0)
            
            with torch.no_grad():
                layer.weight_rho.copy_(layer._get_rho(weight_std))
                layer.bias_rho.copy_(layer._get_rho(bias_std))
        
        print(f"\nAdaptive std init done (target={init_std:.6f}, eps={adaptive_std_epsilon:.6f})")
        self._print_std_summary()
    
    def _print_std_summary(self):
        print("  std stats by layer:")
        all_stds = []
        
        for i, layer in enumerate([self.lin1, self.lin2, self.mean_head, self.logvar_head]):
            weight_std = layer._get_sigma(layer.weight_rho)
            bias_std = layer._get_sigma(layer.bias_rho)
            
            all_std = torch.cat([weight_std.flatten(), bias_std.flatten()])
            all_stds.append(all_std)
            
            layer_min = all_std.min().item()
            layer_max = all_std.max().item()
            layer_mean = all_std.mean().item()
            
            if i < 2:
                layer_name = f"Layer {i+1}"
            elif i == 2:
                layer_name = "Mean Head"
            else:
                layer_name = "LogVar Head"
                
            print(f"    {layer_name}: min={layer_min:.6f}, mean={layer_mean:.6f}, max={layer_max:.6f}")
        
        combined_stds = torch.cat(all_stds)
        overall_min = combined_stds.min().item()
        overall_max = combined_stds.max().item()
        overall_mean = combined_stds.mean().item()
        
        print(f"    Overall: min={overall_min:.6f}, mean={overall_mean:.6f}, max={overall_max:.6f}")