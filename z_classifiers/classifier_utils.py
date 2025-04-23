# classifier utils - handles loading/training classifiers for for classifier uncertainty

from pathlib import Path
import torch

from .mnist_classifier import get_mnist_classifier
from .notmnist_classifier import get_notmnist_classifier


def get_classifier(experiment_type, device, force_train=False):
    """Loads or trains classifier for given experiment type"""
    if experiment_type == 'digit_mnist':
        return get_mnist_classifier(device, force_train)
    elif experiment_type == 'letter_notmnist':
        return get_notmnist_classifier(device, force_train)  
    else:
        print(f"No classifier available for experiment type: {experiment_type}")
        return None 