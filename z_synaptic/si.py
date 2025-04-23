# core SI implementation
import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=100, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MultiHeadMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_tasks=5, head_size=2, num_hidden_layers=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_tasks = num_tasks
        self.head_size = head_size
        self.num_hidden_layers = num_hidden_layers
        

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # task-spec
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, head_size) for _ in range(num_tasks)
        ])
        
        self.current_task = 0
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        for layer in self.layers:
            x = F.relu(layer(x))
        
        x = self.heads[self.current_task](x)
        return x
    
    def set_current_task(self, task_idx):
        if 0 <= task_idx < self.num_tasks:
            self.current_task = task_idx
        else:
            raise ValueError(f"Task index {task_idx} out of range (0-{self.num_tasks-1})")
    
    def forward_all_tasks(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        for layer in self.layers:
            x = F.relu(layer(x))
        
        return [head(x) for head in self.heads]

class SynapticIntelligence:
    def __init__(self, model, lambda_reg=1.0, epsilon=1e-4, omega_decay=0.9, device='cuda', shared_only=False):
        # SI regularizer - tracks param importance
        self.model = model
        self.lambda_reg = lambda_reg
        self.base_epsilon = epsilon
        self.omega_decay = omega_decay
        self.device = device
        self.shared_only = shared_only
        
        # importance tracking
        self.omega = {}  # across tasks
        self.path_integral = {}  # for current task
        self.old_params = {}  #  after finishing a task
        self.pre_step_params = {}  #before optimizer step
        self.avg_delta2 = {}  # moving avg of squred changes for adaptive eps
        self.delta_decay_rate = 0.99
        
        self._debug_prev_state = {}
        
        self._initialize()
    
    def _initialize(self):
        if self.shared_only:
            params = self.model.shared_params
        else:
            params = self.model.named_parameters()
            
        for name, param in params:
            self.omega[name] = torch.zeros_like(param, device=self.device)
            self.path_integral[name] = torch.zeros_like(param, device=self.device)
            self.old_params[name] = param.clone().detach()
            self.pre_step_params[name] = param.clone().detach()
            self.avg_delta2[name] = torch.ones_like(param, device=self.device) * self.base_epsilon
            self._debug_prev_state[name] = param.clone().detach()
            
    def compute_regularization_loss(self):
        reg_loss = 0.0
        try:
            for name, param in self.model.named_parameters():
                if name not in self.omega:
                    continue
                    
                if name in self.omega and name in self.old_params:
                    # handle nans/infs
                    safe_omega = torch.nan_to_num(self.omega[name], nan=0.0, posinf=0.0, neginf=0.0)
                    
                    param_diff = param - self.old_params[name]
                    param_diff = torch.clamp(param_diff, min=-1.0, max=1.0)
                    
                    local_reg = (safe_omega * param_diff.pow(2)).sum()
                    
                    if not torch.isnan(local_reg) and not torch.isinf(local_reg):
                        reg_loss += local_reg
            
            reg_loss = 0.5 * self.lambda_reg * reg_loss
            
            if torch.isnan(reg_loss) or torch.isinf(reg_loss):
                print("WARNING: SI reg loss is NaN/Inf, returning 0")
                return torch.tensor(0.0, device=self.device)
                
            return reg_loss
        except Exception as e:
            print(f"WARNING: SI reg error: {str(e)}. returning 0")
            return torch.tensor(0.0, device=self.device)
    
    def update_omega(self):
        max_importance = 1000.0  
        
        for name, param in self.model.named_parameters():
            if name not in self.omega:
                continue
                
            try:
                delta = param.detach() - self.old_params[name]
                
                if name in self.path_integral:
                    # adaptive eps based on moving avg
                    adaptive_epsilon = torch.clamp(self.avg_delta2[name], min=self.base_epsilon)
                    denominator = torch.clamp(delta.pow(2) + adaptive_epsilon, min=1e-5)
                    
                    importance_update = torch.abs(self.path_integral[name]) / denominator
                    importance_update = torch.nan_to_num(importance_update, nan=0.0, posinf=0.0, neginf=0.0)
                    importance_update = torch.clamp(importance_update, min=-max_importance, max=max_importance)
                    
                    self.omega[name] += importance_update
                    self.omega[name] *= self.omega_decay
                    self.omega[name] = torch.clamp(self.omega[name], min=-max_importance, max=max_importance)
                    
                self.old_params[name] = param.detach().clone()
                self.pre_step_params[name] = param.detach().clone()
                
                # reset 
                self.path_integral[name] = torch.zeros_like(param, device=self.device)
                self.avg_delta2[name] = torch.ones_like(param, device=self.device) * self.base_epsilon
                
                self._debug_prev_state[name] = param.clone().detach()
                
            except Exception as e:
                print(f"WARNING: omega update failed for {name}: {str(e)}")
                self.omega[name] = torch.zeros_like(param, device=self.device)
                self.path_integral[name] = torch.zeros_like(param, device=self.device)
                self.old_params[name] = param.detach().clone()
                self.pre_step_params[name] = param.detach().clone()
                self.avg_delta2[name] = torch.ones_like(param, device=self.device) * self.base_epsilon
    
    def before_step(self):
        # store params before optimizer step
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in self.pre_step_params:
                self.pre_step_params[name] = param.detach().clone()
    
    def accumulate_path_integral(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in self.path_integral:
                delta = param.detach() - self.pre_step_params[name]
                
                safe_grad = torch.clamp(param.grad.detach(), min=-1.0, max=1.0)
                safe_grad = torch.nan_to_num(safe_grad, nan=0.0, posinf=0.0, neginf=0.0)
                
   
                current_delta2 = delta.pow(2)
                current_delta2 = torch.nan_to_num(current_delta2, nan=self.base_epsilon)
                self.avg_delta2[name] = self.delta_decay_rate * self.avg_delta2[name] + \
                                       (1 - self.delta_decay_rate) * current_delta2
                
                path_integral_update = -safe_grad * delta
                path_integral_update = torch.nan_to_num(path_integral_update, nan=0.0)
                self.path_integral[name] += path_integral_update 