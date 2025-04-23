# models with hessian estimation and task registration for LP trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler


class LP_MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=100, output_size=10):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, output_size)

        # store (hessian, params) for each task
        self.lp_data = []

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h1 = F.relu(self.lin1(x))
        h2 = F.relu(self.lin2(h1))
        return self.lin3(h2)

    def predict_softmax(self, x):
        with torch.no_grad():
            return F.softmax(self(x), dim=1)

    def estimate_hessian(self, loader, device, n_samples=600, batch_size=64):
        # diagonal hessian approx using squared grads on true labels
        self.eval()

        idxs = torch.randperm(len(loader.dataset))[:n_samples].tolist()
        sampler = SubsetRandomSampler(idxs)
        h_loader = DataLoader(loader.dataset, batch_size=batch_size,
                            sampler=sampler,
                            num_workers=getattr(loader, "num_workers", 0),
                            pin_memory=getattr(loader, "pin_memory", False))

        criterion = nn.CrossEntropyLoss(reduction='sum')
        hessian = {n: torch.zeros_like(p, device=device)
                   for n, p in self.named_parameters()}
        total = 0

        for x, y in h_loader:
            x, y = x.to(device), y.to(device)
            b = x.size(0)
            total += b

            for n, p in self.named_parameters():
                if p.requires_grad:
                    logits = self(x)
                    loss = criterion(logits, y)
                    # torch.autograd instead of backward() to avoid warnings
                    grads = torch.autograd.grad(loss, p, create_graph=False,retain_graph=False, allow_unused=True)
                    if grads[0] is not None:
                        hessian[n] += grads[0].pow(2).detach()
                    self.zero_grad()

        for n in hessian:
            hessian[n] = (hessian[n] / float(total)).clone().detach()
        return hessian

    def register_lp_task(self, hessian):
        # snapshot params and hessian after task finishes
        theta_star = {n: p.clone().detach() for n, p in self.named_parameters()}
        hessian = {n: h.to(self.get_parameter(n).device) for n, h in hessian.items()}
        self.lp_data.append((hessian, theta_star))


class MultiHeadLP_MLP(nn.Module):
    def __init__(self, input_size=784, hidden_layers=2, hidden_size=256,
                 num_tasks=5, head_size=2):
        super().__init__()
        layers, prev = [], input_size
        for _ in range(hidden_layers):
            layers += [nn.Linear(prev, hidden_size), nn.ReLU(inplace=True)]
            prev = hidden_size
        self.shared_layers = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, head_size)
                                  for _ in range(num_tasks)])
        self.current_task = 0
        self.lp_data = []

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = self.shared_layers(x)
        return self.heads[self.current_task](h)

    def set_current_task(self, idx):
        if 0 <= idx < len(self.heads):
            self.current_task = idx
        else:
            raise ValueError(f"task {idx} out of range")

    predict_softmax = LP_MLP.predict_softmax
    estimate_hessian = LP_MLP.estimate_hessian
    register_lp_task = LP_MLP.register_lp_task
