# gaussian synaptic intelligence models

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HeteroscedasticSI_MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=100, output_size=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # shared
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # mean and var heads
        self.mean_head = nn.Linear(hidden_size, output_size)
        self.logvar_head = nn.Linear(hidden_size, output_size)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)
        
        nn.init.xavier_normal_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        
        # small init 
        nn.init.xavier_normal_(self.logvar_head.weight, gain=0.01)
        nn.init.constant_(self.logvar_head.bias, -4.6)

    def forward(self, x):

        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        
        mean = self.mean_head(h2)
        # clamp logvar for num stabili
        logvar = torch.clamp(self.logvar_head(h2), min=-15.0, max=15.0)
        
        return mean, logvar
    
    def predict_mean_var(self, x):
        mean, logvar = self.forward(x)
        return mean, torch.exp(logvar)
