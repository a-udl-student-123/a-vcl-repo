

import torch
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
from z_utils.training_utils import clean_memory
from z_utils.utils import clean_loader
import math


class CoresetDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data_list = []
        self.label_list = []
        self.task_indices = []  # track which task each batch came from
        self._flat_data = None
        self._flat_labels = None
        self._flat_tasks = None
        self._needs_rebuild = True

    def add_examples(self, examples, labels, task_idx=None):
        self.data_list.append(examples.detach().cpu())
        self.label_list.append(labels.detach().cpu())
        self.task_indices.append(torch.full((examples.size(0),), task_idx if task_idx is not None else -1, 
                                         dtype=torch.long, device='cpu'))
        self._needs_rebuild = True

    def __len__(self):
        return sum(x.size(0) for x in self.data_list)

    def __getitem__(self, idx):
        if self._needs_rebuild or self._flat_data is None:
            self._build_flat_cache()
        return self._flat_data[idx], self._flat_labels[idx], self._flat_tasks[idx]
    
    def get_task_dataset(self, task_idx):
        if self._needs_rebuild or self._flat_data is None:
            self._build_flat_cache()
            
        task_mask = (self._flat_tasks == task_idx)
        
        if not torch.any(task_mask):
            return None
            
        task_data = self._flat_data[task_mask]
        task_labels = self._flat_labels[task_mask]
        
        return SimpleTaskDataset(task_data, task_labels)

    def _build_flat_cache(self):
        if len(self.data_list) > 0:
            self._flat_data = torch.cat(self.data_list, dim=0)
            self._flat_labels = torch.cat(self.label_list, dim=0)
            self._flat_tasks=torch.cat(self.task_indices, dim=0)
            self._needs_rebuild = False


class SimpleTaskDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class FilteredDataset(Dataset):
    def __init__(self, dataset, exclude_indices):
        self.dataset = dataset
        self.exclude_indices = set(exclude_indices)
        self.indices_map = [i for i in range(len(dataset)) if i not in self.exclude_indices]
        
    def __len__(self):
        return len(self.indices_map)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices_map[idx]]


class CoresetWrapper(Dataset):
    def __init__(self, coreset_dataset):
        self.dataset = coreset_dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, label, _ = self.dataset[idx]
        return data, label


def random_coreset_selection(loader, coreset_size):
    if coreset_size <= 0:
        return None, None, []
        
    coreset_x = []
    coreset_y = []
    indices = []
    samples_seen = 0
    batch_start_idx = 0
    
    for x_batch, y_batch in loader:
        batch_size = x_batch.size(0)
        
        for i in range(batch_size):
            global_idx = batch_start_idx + i
            samples_seen += 1
            
            if len(coreset_x) < coreset_size:
                coreset_x.append(x_batch[i].cpu().unsqueeze(0))
                coreset_y.append(y_batch[i].cpu().unsqueeze(0))
                indices.append(global_idx)
            else:
                j = random.randint(0, samples_seen - 1)
                if j < coreset_size:
                    coreset_x[j] = x_batch[i].cpu().unsqueeze(0)
                    coreset_y[j] = y_batch[i].cpu().unsqueeze(0)
                    indices[j] = global_idx
        
        batch_start_idx += batch_size

    if not coreset_x:
        return None, None, []
        
    return torch.cat(coreset_x, dim=0), torch.cat(coreset_y, dim=0), indices


def _process_batches(all_x, batch_idx, batch_size, center, compute_device, storage_device, dist2centers=None):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, all_x.size(0))
    
    batch_x = all_x[start_idx:end_idx].to(compute_device)
    
    new_dists = torch.sum((batch_x - center)**2, dim=1)
    
    if dist2centers is not None:
        curr_dists = dist2centers[start_idx:end_idx].to(compute_device)
        result = torch.min(curr_dists, new_dists).to(storage_device)
    else:
        result = new_dists.to(storage_device)
        
    return start_idx, end_idx, result


def kcenter_coreset_selection(loader, coreset_size, device='cpu', batch_size=1024):
    if coreset_size <= 0:
        return None, None, []
        
    compute_device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    storage_device = torch.device(device)
    
    all_x, all_y = [], []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(storage_device)
        y_batch = y_batch.to(storage_device)
        
        if x_batch.dim() > 2:
            x_batch = x_batch.view(x_batch.size(0), -1)

        all_x.append(x_batch)
        all_y.append(y_batch)

    if not all_x:
        return None, None, []
        
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)

    N = all_x.size(0)
    coreset_size = min(coreset_size, N)
    
    centers_idx = [torch.randint(0, N, size=(1,), device=storage_device).item()]
    first_center = all_x[centers_idx[0]].to(compute_device)
    
    dist2centers = torch.zeros(N, device=storage_device)
    num_batches = (N + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx, end_idx, batch_dist = _process_batches(
            all_x, i, batch_size, first_center, compute_device, storage_device
        )
        dist2centers[start_idx:end_idx] = batch_dist

    with tqdm(range(1, coreset_size), desc="Selecting K-centers", leave=False) as center_pbar:
        for _ in center_pbar:
            max_dist, max_idx = torch.max(dist2centers, dim=0)
            centers_idx.append(max_idx.item())
            center_pbar.set_postfix(max_dist=f"{max_dist.item():.4f}")

            new_center = all_x[max_idx].to(compute_device)
            for i in range(num_batches):
                start_idx, end_idx, updated_dists = _process_batches(
                    all_x, i, batch_size, new_center, compute_device, 
                    storage_device, dist2centers
                )
                dist2centers[start_idx:end_idx] = updated_dists

    final_indices = torch.tensor(centers_idx, dtype=torch.long, device=storage_device)
    return all_x[final_indices].clone(), all_y[final_indices].clone(), centers_idx


def create_filtered_dataloader(original_loader, exclude_indices, batch_size=None, num_workers=None):
    filtered_dataset = FilteredDataset(original_loader.dataset, exclude_indices)
    
    batch_size = batch_size or original_loader.batch_size
    num_workers = num_workers if num_workers is not None else getattr(original_loader, 'num_workers', 0)
    
    dataset_size = len(filtered_dataset)
    
    if dataset_size < 1000:
        workers = max(1, min(num_workers, 2))
    else:
        workers = num_workers
    
    prefetch = min(16, max(2, 4096 // batch_size))
    
    persistent_workers = workers > 0
    
    loader_params = {
        'batch_size': batch_size,
        'num_workers': workers,
        'shuffle': True,
        'pin_memory': True,
        'drop_last': False
    }
    
    if loader_params['num_workers'] > 0:
        loader_params['persistent_workers'] = persistent_workers
        loader_params['prefetch_factor'] = prefetch
    
    filtered_loader = DataLoader(filtered_dataset, **loader_params)
    
    clean_loader(original_loader)
    
    return filtered_loader


def select_coreset(train_loader, coreset_size, use_kcenter, device, kcenter_batch_size=1024):
    if coreset_size <= 0:
        return None, None, []
        
    if use_kcenter:
        return kcenter_coreset_selection(
            train_loader, coreset_size, device=device, batch_size=kcenter_batch_size
        )
    else:
        return random_coreset_selection(train_loader, coreset_size)


def update_coreset(coreset_ds, coreset_x, coreset_y, device, task_idx=None):
    if coreset_x is not None and coreset_y is not None:
        coreset_ds.add_examples(coreset_x, coreset_y, task_idx)
        
    if coreset_x is not None:
        del coreset_x, coreset_y
        clean_memory(device)
        
    return coreset_ds


def create_coreset_loader(coreset_ds, batch_size=128, num_workers=0, task_idx=None):
    if not coreset_ds or len(coreset_ds) == 0:
        return None
        
    if task_idx is not None:
        task_ds = coreset_ds.get_task_dataset(task_idx)
        if task_ds is None or len(task_ds) == 0:
            return None
        dataset = task_ds
    else:
        dataset = CoresetWrapper(coreset_ds)
    
    data_size = len(dataset)
    
    if data_size < 100:
        optimized_batch_size = min(batch_size, max(16, data_size // 4))
    else:
        optimized_batch_size=min(batch_size, max(32, data_size // 8))
    
    if data_size < 1000:
        workers = max(1, min(num_workers, 2))
        prefetch = 2
    else:
        workers = max(1, min(num_workers, 4))
        prefetch = min(4, max(2, data_size // (optimized_batch_size * 2)))
        
    persistent_workers = workers > 0
        
    loader_params = {
        'batch_size': optimized_batch_size,
        'shuffle': True,
        'num_workers': workers,
        'pin_memory': True,
        'drop_last': False
    }
    
    if workers > 0:
        loader_params['persistent_workers'] = persistent_workers
        loader_params['prefetch_factor'] = prefetch
        
    return DataLoader(dataset, **loader_params)


def initialize_coreset():
    return CoresetDataset()


def compute_uncertainty(model, data_loader, n_samples=100):
    """Gets predictive probs using model sampling and computes per-example entropy"""
    model.eval()
    uncertainties, indices = [], []
    offset = 0
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for xb, _ in data_loader:
            xb = xb.to(device)
            
            B = xb.size(0)
            logits = model(xb, n_samples=n_samples)
            p_mean = torch.softmax(logits, dim=1)

            ent = -torch.sum(p_mean * torch.log(p_mean + 1e-8), dim=1)
            for i in range(B):
                uncertainties.append(ent[i].item())
                indices.append(offset + i)
            offset += B

    return uncertainties, indices


def active_select_coreset(model, train_loader, coreset_size,
                         lambda_mix, use_kcenter,
                         device, kcenter_batch_size=1024,
                         num_samples=100):
    # picks m=ceil(lambda K) most uncertain examples + (K-m) random/k-center examples
    # TODO
    uncertainties, sample_indices = compute_uncertainty(model, train_loader, num_samples)
    dataset_size, target_size = len(uncertainties), coreset_size
    num_uncertain = math.ceil(lambda_mix * target_size)

    uncertainty_pairs = list(zip(uncertainties, sample_indices))
    uncertainty_pairs.sort(key=lambda x: -x[0])
    most_uncertain_indices = [i for _, i in uncertainty_pairs[:num_uncertain]]

    remaining_indices = list(set(range(dataset_size)) - set(most_uncertain_indices))
    remaining_needed = target_size - num_uncertain

    additional_indices = []
    if remaining_needed > 0:
        if use_kcenter:
            remaining_subset = torch.utils.data.Subset(train_loader.dataset, remaining_indices)
            remaining_loader = DataLoader(remaining_subset, batch_size=kcenter_batch_size, shuffle=False)
            kcenter_x, kcenter_y, kcenter_indices = kcenter_coreset_selection(
                remaining_loader, remaining_needed, device=device, batch_size=kcenter_batch_size
            )
            additional_indices = [remaining_indices[j] for j in kcenter_indices]
        else:
            additional_indices = random.sample(remaining_indices, remaining_needed)

    selected_indices = most_uncertain_indices + additional_indices

    coreset_inputs, coreset_labels = [], []
    for idx in selected_indices:
        input_tensor, label = train_loader.dataset[idx]
        coreset_inputs.append(input_tensor.unsqueeze(0))
        coreset_labels.append(torch.tensor([label], device=device))
    if coreset_inputs:
        coreset_x = torch.cat(coreset_inputs, dim=0).to(device)
        coreset_y = torch.cat(coreset_labels, dim=0).to(device)
    else:
        coreset_x = coreset_y = None

    return coreset_x, coreset_y, selected_indices