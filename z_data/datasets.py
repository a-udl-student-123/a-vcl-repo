import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import numpy as np
from pathlib import Path
import requests
import tarfile
import shutil
import PIL.Image
from tqdm import tqdm
import io
import random
import warnings

warnings.filterwarnings("ignore", message="A newer version of deeplake.*", category=UserWarning)

try:
    import deeplake
    DEEPLAKE_AVAILABLE = True
except ImportError:
    DEEPLAKE_AVAILABLE = False

def generate_permutations(num_tasks, image_size=28, device='cuda'):
    perms = []
    num_pixels = image_size * image_size
    for _ in range(num_tasks):
        perm = torch.randperm(num_pixels, device=device)
        perms.append(perm)
    return perms

class PermutedMNISTDataset(Dataset):
    def __init__(self, root, permutation, train=True, download=True):
        self.permutation = permutation.cpu()
        self.mnist_data = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        image, label = self.mnist_data[idx]
        image = image.view(-1)
        return image[self.permutation], label

class SplitMNISTDataset(Dataset):
    def __init__(self, root, digits, train=True, download=True):
        self.digits = set(digits)
        self.mnist_data = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor()
        )
        
        self.indices = [i for i, (_, label) in enumerate(self.mnist_data) 
                        if label in self.digits]
        
        self.label_map = {digit: i for i, digit in enumerate(sorted(self.digits))}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.mnist_data[self.indices[idx]]
        image = image.view(-1)
        
        if hasattr(label, 'item'):
            label_value = label.item()
        else:
            label_value = label
            
        binary_label = self.label_map[label_value]
        return image, binary_label

class notMNISTDataset(Dataset):
    def __init__(self, root, train=True, download=True, transform=None, max_samples_per_class=None):
        if not DEEPLAKE_AVAILABLE and download:
            raise ImportError("deeplake not installed. do pip install deeplake")
            
        self.root = Path(root) / "notMNIST"
        self.train = train
        self.transform = transform or transforms.ToTensor()
        self.max_samples_per_class=max_samples_per_class
        
        self.root.mkdir(parents=True, exist_ok=True)
        
        split_name = "train" if train else "test"
        max_samples_str = f"_max{max_samples_per_class}" if max_samples_per_class else ""
        self.cache_file = self.root / f"{split_name}{max_samples_str}.pt"
        
        if self.cache_file.exists():
            self.data, self.targets = torch.load(self.cache_file)
        elif download:
            self._download_and_cache_data()
        else:
            raise RuntimeError(f"Cache file {self.cache_file} not found and download=False")
        
    def _download_and_cache_data(self):
        hub_path = 'hub://activeloop/not-mnist-large' if self.train else 'hub://activeloop/not-mnist-small'
        
        try:
            import io
            import sys
            from contextlib import redirect_stdout
            with redirect_stdout(io.StringIO()):
                ds = deeplake.load(hub_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load notmnist dataset from deeplake: {e}")
        
        images = []
        labels = []
        
        class_counts = {i: 0 for i in range(10)}  # A=0, B=1 etc
        
        for i, sample in enumerate(ds):
            img = sample.images.numpy()
            label = sample.labels.numpy()[0]
            
            if self.max_samples_per_class and class_counts[label] >= self.max_samples_per_class:
                continue
                
            img_tensor = torch.from_numpy(img).float() / 255.0
            if len(img_tensor.shape) == 2:
                img_tensor = img_tensor.unsqueeze(0)
                
            if img_tensor.shape != (1, 28, 28):
                continue
                
            images.append(img_tensor)
            labels.append(label)
            class_counts[label] += 1
        
        self.data = torch.stack(images) if images else torch.empty(0, 1, 28, 28)
        self.targets = torch.tensor(labels, dtype=torch.long) if labels else torch.empty(0, dtype=torch.long)
        
        torch.save((self.data, self.targets), self.cache_file)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        if self.transform:
            img = self.transform(img)
            
        return img, target

class SplitnotMNISTDataset(Dataset):
    def __init__(self, root, letters, train=True, download=True):
        self.letters = set(letters)
        self.notmnist_data = notMNISTDataset(
            root=root,
            train=train,
            download=download
        )
        
        # get indices w/ requested letters
        self.indices = [i for i, label in enumerate(self.notmnist_data.targets) 
                        if label.item() in self.letters]
        
        # map orig labels to binary
        self.label_map = {letter: i for i, letter in enumerate(sorted(self.letters))}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image = self.notmnist_data.data[self.indices[idx]]
        label = self.notmnist_data.targets[self.indices[idx]]
        
        image = image.view(-1)  # flatten to 784 d
        
        label_value = label.item()
        binary_label = self.label_map[label_value]
        
        return image, binary_label

def get_optimal_workers(num_workers, batch_size, single_batch=False):
    if not single_batch:
        if num_workers <= 0:
            num_workers = max(2, min(8, (os.cpu_count() or 4) // 2))
        
        persistent_workers = num_workers > 0
        prefetch_factor = min(6, max(2, 2048 // batch_size))
        
        return num_workers, persistent_workers, prefetch_factor
    else:
        workers = 2 if num_workers >= 2 else num_workers
        return workers, True, 2

def get_test_loader_params(num_workers, batch_size):
    workers = max(1, min(num_workers // 2, 2))
    persistent_workers = False
    
    prefetch_factor = min(4, max(2, 2048//batch_size))
    
    return workers, persistent_workers, prefetch_factor

def create_permuted_mnist_loader_factories(root, permutations, batch_size=256, train=True, num_workers=0):
    """makes factory funcs that generate permuted mnist dataloaders"""
    loader_factories = []
    for perm in permutations:
        def make_loader(permutation=perm, force_persistent=False):
            if train:
                workers, persistent, prefetch = get_optimal_workers(num_workers, batch_size)
            else:
                workers, persistent, prefetch = get_test_loader_params(num_workers, batch_size)
            
            if force_persistent:
                persistent = True
                prefetch = min(16, max(8, 4096 // batch_size))
                
            dataset = PermutedMNISTDataset(
                root=root, 
                permutation=permutation, 
                train=train, 
                download=True
            )
            
            loader_params = {
                'batch_size': batch_size,
                'shuffle': train,
                'num_workers': workers,
                'pin_memory': True,
                'drop_last': False
            }
            
            if workers > 0:
                loader_params['persistent_workers'] = persistent
                loader_params['prefetch_factor'] = prefetch
            
            loader = DataLoader(dataset, **loader_params)
            return loader
            
        loader_factories.append(make_loader)
    
    return loader_factories

def create_split_mnist_loader_factories(root, batch_size=256, train=True, num_workers=0, single_batch=False):
    task_digits = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    
    loader_factories = []
    for digits in task_digits:
        def make_loader(digit_pair=digits, force_persistent=False):
            if train:
                workers, persistent, prefetch=get_optimal_workers(num_workers, batch_size, single_batch)
            else:
                workers, persistent, prefetch = get_test_loader_params(num_workers, batch_size)
                
            if force_persistent:
                persistent = True
                prefetch = min(16, max(4, 4096 // batch_size)) if not single_batch else 2
                
            dataset = SplitMNISTDataset(
                root=root, 
                digits=digit_pair, 
                train=train, 
                download=True
            )
            
            actual_batch_size = len(dataset) if single_batch else batch_size
            
            loader_params = {
                'batch_size': actual_batch_size,
                'shuffle': train,
                'num_workers': workers,
                'pin_memory': True,
                'drop_last': False
            }
            
            if workers > 0:
                loader_params['persistent_workers'] = persistent
                loader_params['prefetch_factor'] = prefetch
            
            loader = DataLoader(dataset, **loader_params)
            return loader
            
        loader_factories.append(make_loader)
    
    return loader_factories

def create_split_notmnist_loader_factories(root, batch_size=256, train=True, num_workers=0, single_batch=False):
    # A/F, B/G, C/H, D/I, E/J pairs
    task_letters = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]  # A=0, B=1 etc
    
    loader_factories = []
    for letters in task_letters:
        def make_loader(letter_pair=letters, force_persistent=False):
            if train:
                workers, persistent, prefetch=get_optimal_workers(num_workers, batch_size, single_batch)
            else:
                workers, persistent, prefetch = get_test_loader_params(num_workers, batch_size)
                
            if force_persistent:
                persistent = True
                prefetch = min(16, max(4, 4096//batch_size)) if not single_batch else 2
                
            dataset = SplitnotMNISTDataset(
                root=root, 
                letters=letter_pair, 
                train=train, 
                download=True
            )
            
            actual_batch_size = len(dataset) if single_batch else batch_size
            
            loader_params = {
                'batch_size': actual_batch_size,
                'shuffle': train,
                'num_workers': workers,
                'pin_memory': True,
                'drop_last': False
            }
            
            if workers > 0:
                loader_params['persistent_workers'] = persistent
                loader_params['prefetch_factor'] = prefetch
            
            loader = DataLoader(dataset, **loader_params)
            return loader
            
        loader_factories.append(make_loader)
    
    return loader_factories
