
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import os
import math
from tqdm import tqdm

from z_data.datasets import get_optimal_workers, get_test_loader_params


class SingleDigitMNIST(Dataset):
    def __init__(self, root, digit, train=True, download=True, transform=None):
        self.digit = digit
        transform = transform or transforms.ToTensor()
        
        self.mnist_data = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform
        )
        
        self.indices = [i for i, (_, label) in enumerate(self.mnist_data) 
                      if label == digit]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        image, _ = self.mnist_data[self.indices[idx]]
        return image, self.digit  # return digit as label


def create_digit_mnist_loader(root, digit, batch_size=128, train=True, 
                            shuffle=True, num_workers=2, force_persistent=False):
    dataset = SingleDigitMNIST(
        root=root,
        digit=digit,
        train=train,
        download=True
    )
    
    if train:
        workers, persistent, prefetch = get_optimal_workers(num_workers, batch_size)
    else:
        workers, persistent, prefetch=get_test_loader_params(num_workers, batch_size)
    
    # override persistence if needed
    if force_persistent:
        persistent = True
        prefetch = min(16, max(4, 4096//batch_size))
    
    loader_params = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': workers,
        'pin_memory': True,
        'drop_last': False
    }
    
    if workers > 0:
        loader_params['persistent_workers'] = persistent
        loader_params['prefetch_factor'] = prefetch
    
    return DataLoader(dataset, **loader_params)


def create_digit_mnist_loader_factory(root, digit, batch_size=128, train=True, 
                                     num_workers=2):
    def make_loader(force_persistent=False):
        return create_digit_mnist_loader(
            root=root,
            digit=digit,
            batch_size=batch_size,
            train=train,
            shuffle=train,
            num_workers=num_workers,
            force_persistent=force_persistent
        )
    
    return make_loader


def create_digit_mnist_loader_factories(root, batch_size=128, train=True, 
                                       num_workers=2, num_digits=10):
    return [
        create_digit_mnist_loader_factory(
            root, digit, batch_size, train, num_workers
        )
        for digit in range(num_digits)
    ]


class SingleLetterNotMNIST(Dataset):
    def __init__(self, root, letter, train=True, download=True, transform=None):
        self.letter = letter
        assert 0 <= letter < 10, "Letter idx must be 0-9 (A-J)"
        
        from z_data.datasets import notMNISTDataset
        
        transform = transform or transforms.ToTensor()
        
        try:
            self.notmnist_data = notMNISTDataset(
                root=root,
                train=train,
                download=download,
                transform=transform
            )
            
            # filter for just this letter
            self.indices = [i for i, label in enumerate(self.notmnist_data.targets) 
                           if label.item() == letter]
                           
        except (ImportError, FileNotFoundError) as e:
            print(f"Error loading notMNIST dataset: {e}")
            print("Using empty dataset as fallback")
            
            self.notmnist_data = None
            self.indices = []
    
    def __len__(self):
        return len(self.indices) if self.notmnist_data is not None else 0
    
    def __getitem__(self, idx):
        if self.notmnist_data is None:
            return torch.zeros(1, 28, 28), self.letter
            
        image = self.notmnist_data.data[self.indices[idx]]
        return image, self.letter


def create_letter_notmnist_loader(root, letter, batch_size=128, train=True, 
                                shuffle=True, num_workers=2, force_persistent=False):
    dataset = SingleLetterNotMNIST(
        root=root,
        letter=letter,
        train=train,
        download=True
    )
    
    if len(dataset)==0:
        print(f"Warning: Empty dataset for letter {letter}")

    if train:
        workers, persistent, prefetch = get_optimal_workers(num_workers, batch_size)
    else:
        workers, persistent, prefetch = get_test_loader_params(num_workers, batch_size)
    
    if force_persistent:
        persistent = True
        prefetch = min(16, max(4, 4096//batch_size))
    
    loader_params = {
        'batch_size': min(batch_size, max(1, len(dataset))),
        'shuffle': shuffle,
        'num_workers': workers,
        'pin_memory': True,
        'drop_last': False
    }
    
    if workers > 0:
        loader_params['persistent_workers'] = persistent
        loader_params['prefetch_factor'] = prefetch
    
    return DataLoader(dataset, **loader_params)


def create_letter_notmnist_loader_factory(root, letter, batch_size=128, train=True, 
                                         num_workers=2):
    def make_loader(force_persistent=False):
        return create_letter_notmnist_loader(
            root=root,
            letter=letter,
            batch_size=batch_size,
            train=train,
            shuffle=train,
            num_workers=num_workers,
            force_persistent=force_persistent
        )
    
    return make_loader


def create_letter_notmnist_loader_factories(root, batch_size=128, train=True, 
                                          num_workers=2, num_letters=10):
    return [
        create_letter_notmnist_loader_factory(
            root, letter, batch_size, train, num_workers
        )
        for letter in range(num_letters)
    ] 