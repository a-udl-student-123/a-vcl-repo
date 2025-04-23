This repo contains implementations of continual learning methods, including VCL, EWC, Synaptic Intelligence (SI), and LP. The code supports DDM and DGM, and includes experiments on MNIST, permuted MNIST, split MNIST, and notMNIST datasets.

## Project Structure

### Naming Convention

All code folders start with a "z_" prefix so that they are grouped together at the bottom of the file tree.

### Core VCL Implementation

For al VCL experiments (with/without coreset, with/without adaptive standard deviation initialization) except for the active learning coreset extension and Gaussian likeilhood extension:
- Main experiment runners are in the parent directory: experiment_runner.py and experiment_runner_dgm.py
- Core code is organized in:
  - z_core/: Core algorithm implementations
  - z_models/: Model architectures
  - z_data/: Dataset loaders and utilities
- Hyperparameter searches:
  - z_sweeps/: WandB sweep runners
  - z_sweep_configs/: Configuration files for hyperparameter sweeps

### Method-Specific Implementations

Each other continual learning method has its own self-contained folders:

Elastic Weight Consolidation (EWC):
- z_ewc/: DDM experiments with EWC
- z_ewc_dgm/: DGM experiments with EWC

Synaptic Intelligence (SI):
- z_synaptic/: DDM experiments with SI (including Gaussian Likelihood)
- z_synaptic_dgm/: DGM experiments with SI
  
Laplace Propagation (LP):
- z_lp/: DDM experiments with LP
- z_lp_dgm/: DGM experiments with LP

Active Coreset Selection:
- z_active_coreset/: Self-contained implementation of VCL with active coreset selection

Gaussian Likelihood VCL:
- z_gaussian/: Self-contained implementation of VCL with Gaussian Likelihood

### Supporting Modules

- z_classifiers/: Includes classifiers for uncertainty estimation in DGM experiments.
- z_utils/: General utilities used across methods (primarily for VCL, but sometimes by other methods)

## Datasets

The experiments use several datasets:
- Permuted MNIST: MNIST with pixel permutations, creating different tasks
- Split MNIST: Binary classification tasks using MNIST digits.
- Split notMNIST: Binary classification tasks using notMNIST letters
- Single digit MNIST: for DGM
- Single letter notMNIST: for DGM

## Dependencies

Main dependencies include:
- PyTorch
- Weights & Biases (wandb) for experiment tracking
- NumPy
- deeplake
- Various standard Python libraries

PS: Sweeps are not all up-to-date and have been changed several times to gather data.