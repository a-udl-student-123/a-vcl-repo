# Mean-Field-Layers and Bayesian NNs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

class MeanFieldLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # current posterior params
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.ones(out_features, in_features) * -6.0)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.ones(out_features) * -6.0)
        
        # init params
        init.normal_(self.weight_mu, mean=0.0, std=0.1)
        init.normal_(self.bias_mu, mean=0.0, std=0.1)

        # store old posterior (prior) for first task
        self.register_buffer('old_weight_mu', torch.zeros(out_features, in_features))
        self.register_buffer('old_weight_sigma', torch.ones(out_features, in_features))
        self.register_buffer('old_bias_mu', torch.zeros(out_features))
        self.register_buffer('old_bias_sigma', torch.ones(out_features))

    def forward(self, x):
        weight = self._sample(self.weight_mu, self.weight_rho)
        bias = self._sample(self.bias_mu, self.bias_rho)
        return F.linear(x, weight, bias)
        
    def forward_mean(self, x):
        return F.linear(x, self.weight_mu, self.bias_mu)

    @staticmethod
    def _get_sigma(rho):
        return torch.exp(0.5 * rho)

    @staticmethod
    def _get_rho(sigma):
        return 2.0 * torch.log(sigma)

    def _sample(self, mu, rho):
        eps = torch.randn_like(mu)
        sigma = self._get_sigma(rho)
        return mu + sigma * eps

    def store_params_as_old(self):
        with torch.no_grad():
            self.old_weight_mu.copy_(self.weight_mu)
            self.old_weight_sigma.copy_(self._get_sigma(self.weight_rho))
            self.old_bias_mu.copy_(self.bias_mu)
            self.old_bias_sigma.copy_(self._get_sigma(self.bias_rho))

    def kl_to_old_posterior(self):
        # compute KL for weights and biases
        kl_w = self._kl_normal_rho(
            mu_new=self.weight_mu,
            rho_new=self.weight_rho,
            mu_old=self.old_weight_mu,
            rho_old=self._get_rho(self.old_weight_sigma)
        )

        kl_b = self._kl_normal_rho(
            mu_new=self.bias_mu,
            rho_new=self.bias_rho,
            mu_old=self.old_bias_mu,
            rho_old=self._get_rho(self.old_bias_sigma)
        )
        
        return kl_w + kl_b

    @staticmethod
    def _kl_normal_rho(mu_new, rho_new, mu_old, rho_old):
        log_var_ratio = rho_old - rho_new
        variance_term = torch.exp(rho_new - rho_old)
        mu_diff_term = (mu_new - mu_old).pow(2) * torch.exp(-rho_old)
        kl = 0.5 * (log_var_ratio + variance_term + mu_diff_term - 1.0)
        return kl.sum()

    @staticmethod
    def _kl_normal(mu_new, sigma_new, mu_old, sigma_old):
        rho_new = MeanFieldLinear._get_rho(sigma_new)
        rho_old = MeanFieldLinear._get_rho(sigma_old)
        return MeanFieldLinear._kl_normal_rho(mu_new, rho_new, mu_old, rho_old)
    
    def set_init_std(self, init_std):
        with torch.no_grad():
            init_rho = self._get_rho(torch.tensor(init_std))
            self.weight_rho.fill_(init_rho)
            self.bias_rho.fill_(init_rho)


class VCL_MLP(nn.Module):
    """Multi-layer perceptron with mean-field layers for VCL."""
    def __init__(self, input_size=784, hidden_size=100, output_size=10):
        super().__init__()
        self.lin1 = MeanFieldLinear(input_size, hidden_size)
        self.lin2 = MeanFieldLinear(hidden_size, hidden_size)
        self.lin3 = MeanFieldLinear(hidden_size, output_size)

    def forward(self, x, n_samples=3):
        """Forward pass with weight sampling using multiple MC samples."""
        batch_size = x.size(0)
        accumulated_logits = torch.zeros(batch_size, self.lin3.out_features, device=x.device)
        
        for _ in range(n_samples):
            h1 = F.relu(self.lin1(x))
            h2 = F.relu(self.lin2(h1))
            logits = self.lin3(h2)
            accumulated_logits += logits
        
        return accumulated_logits / n_samples
        
    def predict_softmax_mean(self, x):
        """Forward pass using the mean weights (no sampling)."""
        h1 = F.relu(self.lin1.forward_mean(x))
        h2 = F.relu(self.lin2.forward_mean(h1))
        logits = self.lin3.forward_mean(h2)
        return F.softmax(logits, dim=1)
        
    def predict_softmax_samples(self, x, n_samples=100):
        """Average predictions over multiple forward passes."""
        logits = self.forward(x, n_samples=n_samples)
        return F.softmax(logits, dim=1)

    def store_params_as_old(self):
        self.lin1.store_params_as_old()
        self.lin2.store_params_as_old()
        self.lin3.store_params_as_old()

    def kl_loss(self):
        kl1 = self.lin1.kl_to_old_posterior()
        kl2 = self.lin2.kl_to_old_posterior()
        kl3 = self.lin3.kl_to_old_posterior()
        
        return kl1 + kl2 + kl3
        
    def set_init_std(self, init_std, adaptive=False, adaptive_std_epsilon=0.01):
        if not adaptive:
            self.lin1.set_init_std(init_std)
            self.lin2.set_init_std(init_std)
            self.lin3.set_init_std(init_std)
            return
            
        # adaptive case with global scaling factor
        param_magnitudes = []
        layers = [self.lin1, self.lin2, self.lin3]
        
        for layer in layers:
            param_magnitudes.append(torch.abs(layer.weight_mu).flatten())
            param_magnitudes.append(torch.abs(layer.bias_mu).flatten())
            
        all_params = torch.cat(param_magnitudes)
        param_values = all_params + adaptive_std_epsilon  # small offset
        param_values = torch.clamp(param_values, min=1e-6, max=2.0)  # clip
        
        # global calibration
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
        
        # Print initialization summary
        print(f"\nAdaptive std initialization complete (target avg={init_std:.6f}, epsilon={adaptive_std_epsilon:.6f})")
        self._print_std_summary()
    
    def _print_std_summary(self):
        print("  Standard deviation stats by layer:")
        all_stds = []
        
        # Process each layer
        for i, layer in enumerate([self.lin1, self.lin2, self.lin3]):
            weight_std = layer._get_sigma(layer.weight_rho)
            bias_std = layer._get_sigma(layer.bias_rho)
            
            all_std = torch.cat([weight_std.flatten(), bias_std.flatten()])
            all_stds.append(all_std)
            
            layer_min = all_std.min().item()
            layer_max = all_std.max().item()
            layer_mean = all_std.mean().item()
            
            layer_name = "Output Layer" if i == 2 else f"Layer {i+1}"
            print(f"    {layer_name}: min={layer_min:.6f}, mean={layer_mean:.6f}, max={layer_max:.6f}")
        
        # Calculate overall statistics
        combined_stds = torch.cat(all_stds)
        overall_min = combined_stds.min().item()
        overall_max = combined_stds.max().item()
        overall_mean = combined_stds.mean().item()
        
        print(f"    Overall: min={overall_min:.6f}, mean={overall_mean:.6f}, max={overall_max:.6f}")


class VCL_MultiHead_MLP(nn.Module):
    """Multi-head MLP for VCL with task-specific output heads."""
    def __init__(self, input_size=784, hidden_size=256, num_tasks=5, head_size=2, init_std=None):
        super().__init__()
        self.lin1 = MeanFieldLinear(input_size, hidden_size)
        self.lin2 = MeanFieldLinear(hidden_size, hidden_size)
        
        # Task-specific output heads
        self.heads = nn.ModuleList([
            MeanFieldLinear(hidden_size, head_size) for _ in range(num_tasks)
        ])
        
        self.num_tasks = num_tasks
        self.head_size = head_size
        self.current_task = 0  # track current task during training
        self.init_std = init_std
        
    def set_current_task(self, task_idx):
        assert 0 <= task_idx < self.num_tasks, f"Task index {task_idx} out of range (0-{self.num_tasks-1})"
        self.current_task = task_idx
        
    def forward(self, x, task_idx=None, n_samples=3):
        if task_idx is None:
            task_idx = self.current_task
            
        batch_size = x.size(0)
        accumulated_logits = torch.zeros(batch_size, self.head_size, device=x.device)
        
        for _ in range(n_samples):
            h1 = F.relu(self.lin1(x))
            h2 = F.relu(self.lin2(h1))
            logits = self.heads[task_idx](h2)
            accumulated_logits += logits
        
        return accumulated_logits / n_samples
        
    def predict_all_tasks(self, x, n_samples=3):
        """Forward pass for all tasks, returning list of predictions."""
        results = []
        for task_idx in range(self.num_tasks):
            logits = self.forward(x, task_idx=task_idx, n_samples=n_samples)
            results.append(logits)
        return results
        
    def predict_softmax_mean(self, x, task_idx=None):
        """Forward pass using mean weights without sampling."""
        if task_idx is None:
            task_idx = self.current_task
            
        h1 = F.relu(self.lin1.forward_mean(x))
        h2 = F.relu(self.lin2.forward_mean(h1))
        logits = self.heads[task_idx].forward_mean(h2)
        return F.softmax(logits, dim=1)
        
    def predict_softmax_samples(self, x, task_idx=None, n_samples=100):
        """Average predictions over multiple forward passes."""
        if task_idx is None:
            task_idx = self.current_task
            
        logits = self.forward(x, task_idx=task_idx, n_samples=n_samples)
        return F.softmax(logits, dim=1)

    def store_params_as_old(self):
        # store current params as old posterior for next task
        self.lin1.store_params_as_old()
        self.lin2.store_params_as_old()
        self.heads[self.current_task].store_params_as_old()

    def kl_loss(self):
        kl1 = self.lin1.kl_to_old_posterior()
        kl2 = self.lin2.kl_to_old_posterior()
        kl_head = self.heads[self.current_task].kl_to_old_posterior()
        return kl1 + kl2 + kl_head
    
    def initialize_new_head(self, task_idx, mean_std=0.1, posterior_std=0.05):
        head = self.heads[task_idx]
        
        # use init_std if provided
        if self.init_std is not None:
            mean_std = self.init_std
            posterior_std = self.init_std
        
        with torch.no_grad():
            init.normal_(head.weight_mu, mean=0.0, std=mean_std)
            init.normal_(head.bias_mu, mean=0.0, std=mean_std)
            
            fixed_rho = head._get_rho(torch.tensor(posterior_std))
            head.weight_rho.fill_(fixed_rho)
            head.bias_rho.fill_(fixed_rho)
            
        print(f"\nInitialized head {task_idx+1}: mean with std={mean_std:.4f}, posterior_std={posterior_std:.4f}")
        
    def set_init_std(self, init_std, adaptive=False, adaptive_std_epsilon=0.01):
        """
        Set initial posterior standard deviation for shared layers and first head.
        For later tasks, we use initialize_new_head instead. Optionally, we can use adaptive std
        (only sensbile if combined with ml initialization).
        """
        # For subsequent tasks, we use initialize_new_head instead
        if self.current_task > 0:
            print(f"\nSkipping set_init_std for task {self.current_task} (use initialize_new_head instead)")
            return
            
        if not adaptive:
            self.lin1.set_init_std(init_std)
            self.lin2.set_init_std(init_std)
            self.heads[0].set_init_std(init_std)
            print(f"\nInitialized shared layers and first head with fixed std={init_std:.6f}")
            return
            
        param_magnitudes = []
        layers = [self.lin1, self.lin2, self.heads[0]]
        
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
        
        print(f"\nAdaptive std initialization complete (target avg={init_std:.6f}, epsilon={adaptive_std_epsilon:.6f})")
        self._print_std_summary()
    
    def _print_std_summary(self):
        print("  Standard deviation statistics by layer:")
        all_stds = []
        
        # shared layers
        for i, layer in enumerate([self.lin1, self.lin2]):
            weight_std = layer._get_sigma(layer.weight_rho)
            bias_std = layer._get_sigma(layer.bias_rho)
            
            all_std = torch.cat([weight_std.flatten(), bias_std.flatten()])
            all_stds.append(all_std)
            
            layer_min = all_std.min().item()
            layer_max = all_std.max().item()
            layer_mean = all_std.mean().item()
            
            print(f"    Shared Layer {i+1}: min={layer_min:.6f}, mean={layer_mean:.6f}, max={layer_max:.6f}")
        
        # current head stats
        head = self.heads[self.current_task]
        weight_std = head._get_sigma(head.weight_rho)
        bias_std = head._get_sigma(head.bias_rho)
        
        head_std = torch.cat([weight_std.flatten(), bias_std.flatten()])
        all_stds.append(head_std)
        
        head_min = head_std.min().item()
        head_max = head_std.max().item()
        head_mean = head_std.mean().item()
        
        print(f"    Head {self.current_task+1}: min={head_min:.6f}, mean={head_mean:.6f}, max={head_max:.6f}")
        
        combined_stds = torch.cat(all_stds)
        overall_min = combined_stds.min().item()
        overall_max = combined_stds.max().item()
        overall_mean = combined_stds.mean().item()
        
        print(f"    Overall: min={overall_min:.6f}, mean={overall_mean:.6f}, max={overall_max:.6f}")


class VCL_FlexibleMultiHead_MLP(nn.Module):
    """Multi-head MLP for VCL with task-specific output heads and configurable hidden layers."""
    def __init__(self, input_size=784, hidden_sizes=None, num_tasks=5, head_size=2, init_std=None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 256]  # Default to two hidden layers
            
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.num_tasks = num_tasks
        self.head_size = head_size
        self.current_task = 0  
        self.init_std = init_std 
        
        self.shared_layers = nn.ModuleList()
        
        self.shared_layers.append(MeanFieldLinear(input_size, hidden_sizes[0]))
        
        for i in range(1, len(hidden_sizes)):
            self.shared_layers.append(MeanFieldLinear(hidden_sizes[i-1], hidden_sizes[i]))
        
        # task-specific output heads
        self.heads = nn.ModuleList([
            MeanFieldLinear(hidden_sizes[-1], head_size) for _ in range(num_tasks)
        ])
        
    def set_current_task(self, task_idx):
        assert 0 <= task_idx < self.num_tasks, f"Task index {task_idx} out of range (0-{self.num_tasks-1})"
        self.current_task = task_idx
        
    def forward(self, x, task_idx=None, n_samples=3):
        if task_idx is None:
            task_idx = self.current_task
            
        batch_size = x.size(0)
        accumulated_logits = torch.zeros(batch_size, self.head_size, device=x.device)
        
        for _ in range(n_samples):
            h = x
            for layer in self.shared_layers:
                h = F.relu(layer(h))
            
            # task head
            logits = self.heads[task_idx](h)
            accumulated_logits += logits
        
        return accumulated_logits / n_samples
        
    def predict_all_tasks(self, x, n_samples=3):
        """Forward pass for all tasks, returning list of predictions."""
        results = []
        for task_idx in range(self.num_tasks):
            logits = self.forward(x, task_idx=task_idx, n_samples=n_samples)
            results.append(logits)
        return results
        
    def predict_softmax_mean(self, x, task_idx=None):
        """Forward pass using mean weights without sampling."""
        if task_idx is None:
            task_idx = self.current_task
            
        h = x
        for layer in self.shared_layers:
            h = F.relu(layer.forward_mean(h))
        
        logits = self.heads[task_idx].forward_mean(h)
        return F.softmax(logits, dim=1)
        
    def predict_softmax_samples(self, x, task_idx=None, n_samples=100):
        """Average predictions over multiple forward passes."""
        if task_idx is None:
            task_idx = self.current_task
            
        logits = self.forward(x, task_idx=task_idx, n_samples=n_samples)
        return F.softmax(logits, dim=1)

    def store_params_as_old(self):
        for layer in self.shared_layers:
            layer.store_params_as_old()
        
        self.heads[self.current_task].store_params_as_old()

    def kl_loss(self):
        total_kl = 0
        for layer in self.shared_layers:
            total_kl += layer.kl_to_old_posterior()
        
        total_kl += self.heads[self.current_task].kl_to_old_posterior()
        
        return total_kl
    
    def initialize_new_head(self, task_idx, mean_std=0.1, posterior_std=0.05):
        head = self.heads[task_idx]
        
        if self.init_std is not None:
            mean_std = self.init_std
            posterior_std = self.init_std
        
        with torch.no_grad():
            init.normal_(head.weight_mu, mean=0.0, std=mean_std)
            init.normal_(head.bias_mu, mean=0.0, std=mean_std)
            
            fixed_rho = head._get_rho(torch.tensor(posterior_std))
            head.weight_rho.fill_(fixed_rho)
            head.bias_rho.fill_(fixed_rho)
            
        print(f"\nInitialized head {task_idx+1}: mean with std={mean_std:.4f}, posterior_std={posterior_std:.4f}")
        
    def set_init_std(self, init_std, adaptive=False, adaptive_std_epsilon=0.01):
        """
        Set initial posterior standard deviation for shared layers and first head.
        """
        # For later tasks, we use initialize_new_head instead
        if self.current_task > 0:
            print(f"\nSkipping set_init_std for task {self.current_task} (use initialize_new_head instead)")
            return
            
        if not adaptive:
            for layer in self.shared_layers:
                layer.set_init_std(init_std)
            self.heads[0].set_init_std(init_std)
            print(f"\nInitialized shared layers and first head with fixed std={init_std:.6f}")
            return
            
        param_magnitudes = []
        layers = list(self.shared_layers) + [self.heads[0]]
        
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
        
        # Print initialization summary
        print(f"\nAdaptive std initialization complete (target avg={init_std:.6f}, epsilon={adaptive_std_epsilon:.6f})")
        self._print_std_summary()
    
    def _print_std_summary(self):
        print("  Standard deviation stats by layer:")
        all_stds = []
        
        for i, layer in enumerate(self.shared_layers):
            weight_std = layer._get_sigma(layer.weight_rho)
            bias_std = layer._get_sigma(layer.bias_rho)
            
            all_std = torch.cat([weight_std.flatten(), bias_std.flatten()])
            all_stds.append(all_std)
            
            layer_min = all_std.min().item()
            layer_max = all_std.max().item()
            layer_mean = all_std.mean().item()
            
            print(f"    Shared Layer {i+1}: min={layer_min:.6f}, mean={layer_mean:.6f}, max={layer_max:.6f}")
        
        head = self.heads[self.current_task]
        weight_std = head._get_sigma(head.weight_rho)
        bias_std = head._get_sigma(head.bias_rho)
        
        head_std = torch.cat([weight_std.flatten(), bias_std.flatten()])
        all_stds.append(head_std)
        
        head_min = head_std.min().item()
        head_max = head_std.max().item()
        head_mean = head_std.mean().item()
        
        print(f"    Head {self.current_task+1}: min={head_min:.6f}, mean={head_mean:.6f}, max={head_max:.6f}")
        
        combined_stds = torch.cat(all_stds)
        overall_min = combined_stds.min().item()
        overall_max = combined_stds.max().item()
        overall_mean = combined_stds.mean().item()
        
        print(f"    Overall: min={overall_min:.6f}, mean={overall_mean:.6f}, max={overall_max:.6f}")
