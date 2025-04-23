import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler


class EWC_MLP(nn.Module):
    # basic 3-layer MLP with EWC functionality for storing fisher matrices
    # and params from past tasks to compute penalty during training
    def __init__(self, input_size=784, hidden_size=100, output_size=10):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, output_size)

        self.ewc_data = [] # fisher mats + params from prev tasks

    def forward(self, x):
        h1 = F.relu(self.lin1(x))
        h2=F.relu(self.lin2(h1))  
        return self.lin3(h2)

    def predict_softmax(self, x):
        with torch.no_grad():
            return F.softmax(self(x), dim=1)

    def estimate_fisher(self, data_loader, device, n_samples=600, batch_size=64):
        # compute diagonal fisher info matrix by sampling n examples and getting
        # grads of log likelihood wrt params
        self.eval()

        dataset = data_loader.dataset
        idxs = torch.randperm(len(dataset))[:n_samples].tolist()
        sampler = SubsetRandomSampler(idxs)
        fisher_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=getattr(data_loader, "num_workers", 0),
            pin_memory=getattr(data_loader, "pin_memory", False),
        )

        criterion = nn.CrossEntropyLoss(reduction="sum")
        fisher = {n: torch.zeros_like(p, device=device) for n, p in self.named_parameters()}
        total = 0


        for inputs, _ in fisher_loader:
            inputs = inputs.to(device)
            b = inputs.size(0)
            total += b

            logits = self(inputs)

            probs = F.softmax(logits, dim=1)
            sampled = torch.multinomial(probs, 1).squeeze(1)

            loss = criterion(logits, sampled)

            self.zero_grad()
            loss.backward()

            for n, p in self.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)

        for n in fisher:
            fisher[n] = (fisher[n]/float(total)).clone().detach()

        return fisher

    def register_ewc_task(self, fisher):
        # store fisher + params for completed task
        theta_star = {n: p.clone().detach() for n, p in self.named_parameters()}
        fisher = {n: f.to(p.device) for (n, f), (_, p) in zip(fisher.items(),
                                                              self.named_parameters())}
        self.ewc_data.append((fisher, theta_star))


class MultiHeadEWC_MLP(nn.Module):
    """shared mlp backbone + one linear head per task. keeps same estimate_fisher / 
    register_ewc_task interface as EWC_MLP so train_ewc() works unchanged"""
    def __init__(self, input_size=784, hidden_layers=2, hidden_size=100,
                 num_tasks=5, head_size=2):  # binary heads for split mnist
        super().__init__()
        layers = []
        prev_size = input_size
        for _ in range(hidden_layers):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(inplace=True)
            ])
            prev_size = hidden_size
        self.shared_layers = nn.Sequential(*layers)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, head_size) for _ in range(num_tasks)]
        )
        self.current_task = 0
        self.ewc_data = [] # fisher + params for done tasks

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = self.shared_layers(x)
        return self.heads[self.current_task](h)
    
    def predict_softmax(self, x):
        with torch.no_grad():
            return F.softmax(self(x), dim=1)

    def set_current_task(self, task_idx):
        if 0 <= task_idx < len(self.heads):
            self.current_task = task_idx
        else:
            raise ValueError(f"task {task_idx} out of range")

    # reuse these from parent class since theyre identical
    estimate_fisher = EWC_MLP.estimate_fisher
    register_ewc_task = EWC_MLP.register_ewc_task