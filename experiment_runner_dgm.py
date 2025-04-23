# core  for running gen model experiments

import torch
import pandas as pd
from pathlib import Path
import gc
import wandb
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback
from datetime import datetime
from torchvision.utils import make_grid, save_image

from z_data.datasets_dgm import (
    create_digit_mnist_loader_factories,
    create_letter_notmnist_loader_factories
)
from z_models.generative.vcl_models_dgm import VCL_VAE
from z_core.vcl_dgm import train_vcl_dgm
from z_utils.wandb_utils import init_wandb, finish_wandb
from z_utils.utils import clean_loader


def setup_experiment_dirs(experiment_dir):
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "plots").mkdir(exist_ok=True)
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "samples").mkdir(exist_ok=True)
    return experiment_dir


def cleanup_resources(train_loader_factories=None, test_loader_factories=None, classifier=None):
    if test_loader_factories is not None:
        del test_loader_factories
    if classifier is not None:
        del classifier
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def create_model_factory(config, device):
    input_size = config.get("input_size", 784)
    hidden_size = config.get("hidden_size", 500)
    num_tasks = config.get("num_tasks", 10)
    
    def create_model():
        return VCL_VAE(
            input_size=input_size,
            hidden_size=hidden_size,
            latent_size=50,
            num_tasks=num_tasks
        ).to(device)
        
    return create_model


def create_data_loader_factories(config, num_workers):
    experiment_type = config.get("experiment_type", "digit_mnist")
    batch_size = config.get("batch_size", 128)
    num_tasks = config.get("num_tasks", 10)

    if experiment_type == "letter_notmnist":
        train_loader_factories = create_letter_notmnist_loader_factories(
            root='data/',
            batch_size=batch_size,
            train=True,
            num_letters=num_tasks
        )
        
        test_loader_factories = create_letter_notmnist_loader_factories(
            root='data/',
            batch_size=batch_size*2,
            train=False,
            num_workers=num_workers,
            num_letters=num_tasks
        )

        train_loader_factories = create_digit_mnist_loader_factories(
            root='data/',
            batch_size=batch_size,
            train=True,
            num_workers=num_workers,
        )
        
        test_loader_factories = create_digit_mnist_loader_factories(
            root='data/',
            batch_size=batch_size*2,
            train=False,
            num_workers=num_workers,
            num_digits=num_tasks
        )



def create_metric_recorder():
    metrics = []
    
    def record_metric(task, epoch, name, value):
        value_to_record = float(value) if isinstance(value, (int, float)) else value
        metrics.append({
            'task': task,
            'epoch': epoch,
            'metric': name,
            'value': value_to_record
        })
        
    return record_metric, metrics


def create_wandb_metric_recorder(config, metrics):
    epochs_per_task = config.get('epochs', 200)
    
    def record_metric_with_wandb(task, epoch, name, value):
        value_to_record = float(value) if isinstance(value, (int, float)) else value
        metrics.append({
            'task': task,
            'epoch': epoch,
            'metric': name,
            'value': value_to_record
        })
        
        if wandb.run is None:
            return
            
        if epoch >= 0:
            global_step = task * epochs_per_task + epoch
            
            if name in ['elbo', 'recon_loss', 'kl_latent', 'kl_params']:
                wandb.log({
                    f"training/elbo/{name}": value_to_record
                }, step=global_step)
            elif name.startswith('decoder_'):
                wandb.log({
                    f"parameters/{name}": value_to_record
                }, step=global_step)
            else:
                wandb.log({
                    f"training/{name}": value_to_record
                }, step=global_step)
        else:
            task_end_step = (task+1) * epochs_per_task
            
            if name.startswith('log_likelihood_task_'):
                task_num = name.split('_')[-1]
                wandb.log({
                    f"evaluation/tasks/{task_num}/log_likelihood": value_to_record
                }, step=task_end_step)
            elif name.startswith('recon_error_task_'):
                task_num = name.split('_')[-1]
                wandb.log({
                    f"evaluation/tasks/{task_num}/recon_error": value_to_record
                }, step=task_end_step)
            elif name.startswith('cls_uncertainty_task_'):
                task_num = name.split('_')[-1]
                wandb.log({
                    f"evaluation/tasks/{task_num}/cls_uncertainty": value_to_record
                }, step=task_end_step)
            elif name.startswith('average_'):
                wandb.log({
                    f"evaluation/{name}": value_to_record
                }, step=task_end_step)
            else:
                wandb.log({
                    f"metrics/{name}": value_to_record
                }, step=task_end_step)
                
    return record_metric_with_wandb


def create_classifier(config, device):
    # get pretrained classifier for eval metrics
    experiment_type = config.get("experiment_type", "digit_mnist")
    force_train = config.get("force_train_classifier", False)
    
    from z_classifiers.classifier_utils import get_classifier


def visualize_reconstructions(model, test_loaders, device, output_dir):
    model.eval()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    samples_dir = output_dir / "samples"
    model.eval()  
    
    results = []
    
    # get reconstructions for each task
    for task_id, loader in enumerate(test_loaders):
        if loader is None:
            continue
            
        try:
            batch = next(iter(loader))
            if isinstance(batch, (list, tuple)):
                x, _ = batch
            else:
                x = batch
            
            x_cpu = x.cpu()
            
            if x.dim() > 2:
                x_flat = x.reshape(x.size(0), -1).to(device)
            else:
                x_flat = x.to(device)
                
            with torch.no_grad():
                try:
                    recon_x = model.reconstruct(x_flat)
                    recon_x = recon_x.cpu()
                except RuntimeError as e:
                    if "Expected all tensors to be on the same device" in str(e):
                        actual_device = x_flat.device
                        print(f"Device mismatch, moving model to {actual_device}")
                        model = model.to(actual_device)
                        recon_x = model.reconstruct(x_flat)
                        recon_x = recon_x.cpu()
                    else:
                        raise
            
            if x_cpu.dim() > 2:
                original_shape = x_cpu.shape
                recon_x = recon_x.reshape(original_shape)
                
            # generate samples for first task
            if task_id == 0:
                with torch.no_grad():
                    try:
                        samples = model.sample(100).cpu()
                        samples_path = samples_dir / f"samples_task_{task_id}.png"
                        if samples.dim() > 2:
                            samples = samples.reshape(100, *original_shape[1:])
                            
                        grid = make_grid(samples, nrow=10, normalize=True)
                        save_image(grid, samples_path)
                    except Exception as e:
                        print(f"Error generating samples for task {task_id}: {str(e)}")
                        
            fig, axes = plt.subplots(2, 8, figsize=(16, 4))
            
            x_np = x_cpu.numpy()
            recon_x_np = recon_x.numpy()
            
            for i in range(8):
                if x_cpu.size(1) == 1:
                    axes[0, i].imshow(x_np[i, 0], cmap='gray')
                    axes[1, i].imshow(recon_x_np[i, 0], cmap='gray')
                else:
                    axes[0, i].imshow(np.transpose(x_np[i], (1, 2, 0)))
                    axes[1, i].imshow(np.transpose(recon_x_np[i], (1, 2, 0)))
                
                axes[0, i].axis('off')
                axes[1, i].axis('off')
                
            axes[0, 0].set_ylabel('Original')
            axes[1, 0].set_ylabel('Reconstruction')
            
            recon_path = output_dir / f"reconstructions_task_{task_id}.png"
            plt.tight_layout()
            plt.savefig(recon_path)
            plt.close()
            
            results.append((f"reconstructions_task_{task_id}", wandb.Image(str(recon_path))))
            
        except Exception as e:
            print(f"Error processing visualizations for task {task_id}: {str(e)}")
            traceback.print_exc()
    
    # combine examples from all tasks
    try:
        all_tasks_x = []
        all_tasks_recon = []
        
        for task_id, loader in enumerate(test_loaders):
            if loader is None:
                continue
                
            try:
                batch = next(iter(loader))
                if isinstance(batch, (list, tuple)):
                    x, _ = batch
                else:
                    x = batch
                    
                x = x[:2]
                
                x_cpu = x.cpu()
                
                if x.dim() > 2:
                    x_flat = x.reshape(x.size(0), -1).to(device)
                else:
                    x_flat = x.to(device)
                    
                with torch.no_grad():
                    try:
                        recon_x = model.reconstruct(x_flat)
                        recon_x = recon_x.cpu()
                    except RuntimeError as e:
                        if "Expected all tensors to be on the same device" in str(e):
                            actual_device = x_flat.device
                            print(f"Device mismatch, moving model to {actual_device}")
                            model = model.to(actual_device)
                            recon_x = model.reconstruct(x_flat)
                            recon_x = recon_x.cpu()
                        else:
                            raise
                
                if x_cpu.dim() > 2:
                    original_shape = x_cpu.shape
                    recon_x = recon_x.reshape(original_shape)
                    
                all_tasks_x.append(x_cpu)
                all_tasks_recon.append(recon_x)
                
            except Exception as e:
                print(f"Error creating multi-task vis for task {task_id}: {str(e)}")
                continue
        
        if all_tasks_x:
            max_tasks = min(5, len(all_tasks_x))
            
            fig, axes = plt.subplots(max_tasks * 2, 2, figsize=(4, max_tasks * 2))
            
            for i in range(max_tasks):
                if i >= len(all_tasks_x):
                    break
                
                x = all_tasks_x[i]
                recon_x = all_tasks_recon[i]
                
                x_np = x.numpy()
                recon_x_np = recon_x.numpy()
                
                for j in range(min(2, x.size(0))):
                    if x.dim() > 2:
                        if x.size(1) == 1:  # grayscale
                            axes[i*2, j].imshow(x_np[j, 0], cmap='gray')
                            axes[i*2+1, j].imshow(recon_x_np[j, 0], cmap='gray')
                        else:  # rgb
                            axes[i*2, j].imshow(np.transpose(x_np[j], (1, 2, 0)))
                            axes[i*2+1, j].imshow(np.transpose(recon_x_np[j], (1, 2, 0)))
                    else:
                        side_length = int(np.sqrt(x.size(1)))
                        axes[i*2, j].imshow(x_np[j].reshape(side_length, side_length), cmap='gray')
                        axes[i*2+1, j].imshow(recon_x_np[j].reshape(side_length, side_length), cmap='gray')
                        
                    axes[i*2, j].axis('off')
                    axes[i*2+1, j].axis('off')
                    
                axes[i*2, 0].set_ylabel(f'Task {i}\nOrig')
                axes[i*2+1, 0].set_ylabel('Recon')
                
            multi_task_path = output_dir / "multi_task_reconstructions.png"
            plt.tight_layout()
            plt.savefig(multi_task_path)
            plt.close()
            
            results.append(("multi_task_reconstructions", wandb.Image(str(multi_task_path))))
    except Exception as e:
        print(f"Error creating multi-task visualization: {str(e)}")
        traceback.print_exc()
        
    return results


def save_error_log(error, config, experiment_dir):
    error_file = experiment_dir / "error_log.txt"
    with open(error_file, "w") as f:
        f.write(f"Error occurred at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Error message: {str(error)}\n\n")
        f.write("Traceback:\n")
        f.write(traceback.format_exc())
        
        f.write("\nExperiment Configuration:\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")


def run_experiment(config, experiment_dir, device='cuda'):
    experiment_dir = setup_experiment_dirs(experiment_dir)
    
    record_metric, metrics = create_metric_recorder()
    
    train_loader_factories, test_loader_factories = create_data_loader_factories(
        config, config.get('num_workers', 4)
    )
    model_factory = create_model_factory(config, device)
    
    classifier = create_classifier(config, device) if config.get('use_classifier', True) else None
    
    record_metric(0, -1, 'init_std', config.get('init_std', 0.001))
    
    try:
        vcl_params = {
            'model_class': model_factory,
            'train_loader_factories': train_loader_factories,
            'test_loader_factories': test_loader_factories,
            'epochs_per_task': config.get('epochs', 200),
            'lr': config.get('lr', 1e-4),
            'device': device,
            'num_workers': config.get('num_workers', 4),
            'record_metric_fn': record_metric,
            'exp_dir': experiment_dir,
            'n_train_samples': config.get('n_train_samples', 1),
            'n_eval_samples': config.get('n_eval_samples', 5000),
            'classifier': classifier,
            'init_std': config.get('init_std', 0.001),
            'early_stopping_threshold': config.get('early_stopping_threshold', None),
        }
        
        if config.get('method') == 'coreset_vcl':
            vcl_params.update({
                'coreset_size': config.get('coreset_size', 200),
                'use_kcenter': config.get('use_kcenter', False),
                'kcenter_batch_size': config.get('kcenter_batch_size', 1024)
            })
        else:
            vcl_params['coreset_size'] = 0
        
        trained_model = train_vcl_dgm(**vcl_params)
        
        visualization_results = visualize_reconstructions(trained_model, 
                                [loader() for loader in test_loader_factories], 
                                device, 
                                experiment_dir)
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(experiment_dir / "metrics.csv", index=False)
        
        return metrics_df
    
    except Exception as e:
        save_error_log(e, config, experiment_dir)
        raise
    
    finally:
        cleanup_resources(train_loader_factories, test_loader_factories, classifier)


def run_experiment_with_wandb(config, experiment_dir, project_name, device='cuda'):
    experiment_dir = setup_experiment_dirs(experiment_dir)
    
    init_wandb(config, project_name=project_name, 
               tags=[config.get("method", "standard_vcl_dgm"), 
                     f"tasks={config.get('num_tasks', 10)}", 
                     config["experiment_type"]])
    
    metrics = []
    record_metric_with_wandb = create_wandb_metric_recorder(config, metrics)
    
    train_loader_factories, test_loader_factories = create_data_loader_factories(
        config, config.get('num_workers', 4)
    )
    model_factory = create_model_factory(config, device)
    
    classifier = create_classifier(config, device) if config.get('use_classifier', True) else None
    
    try:
        vcl_params = {
            'model_class': model_factory,
            'train_loader_factories': train_loader_factories,
            'test_loader_factories': test_loader_factories,
            'epochs_per_task': config.get('epochs', 200),
            'lr': config.get('lr', 1e-4),
            'device': device,
            'record_metric_fn': record_metric_with_wandb,
            'exp_dir': experiment_dir,
            'n_train_samples': config.get('n_train_samples', 1),
            'n_eval_samples': config.get('n_eval_samples', 5000),
            'classifier': classifier,
            'init_std': config.get('init_std', 0.001),
            'early_stopping_threshold': config.get('early_stopping_threshold', None),
        }
        
        trained_model = train_vcl_dgm(**vcl_params)
        
        try:
            vis_test_loaders = []
            for i, loader_factory in enumerate(test_loader_factories):
                if i < getattr(trained_model, "current_task", 1) + 1:
                    try:
                        vis_test_loaders.append(loader_factory())
                    except Exception as e:
                        print(f"Error creating test loader for task {i}: {str(e)}")
                        vis_test_loaders.append(None)
                else:
                    vis_test_loaders.append(None)
                    
            visualization_results = visualize_reconstructions(
                trained_model, 
                vis_test_loaders, 
                device, 
                experiment_dir
            )
            
            for loader in vis_test_loaders:
                if loader is not None:
                    clean_loader(loader)
        except Exception as e:
            print(f"Error during visualization generation: {str(e)}")
            visualization_results = []
        
        try:
            for plot_file in (experiment_dir / "plots").glob("*.png"):
                wandb.log({f"plots/{plot_file.stem}": wandb.Image(str(plot_file))})
        
            for caption, img in visualization_results:
                wandb.log({f"visualizations/{caption}": img})
                
            for sample_file in (experiment_dir / "samples").glob("*.png"):
                wandb.log({f"samples/{sample_file.stem}": wandb.Image(str(sample_file))})
        except Exception as e:
            print(f"Error logging images to wandb: {str(e)}")
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(experiment_dir / "metrics.csv", index=False)
        
        return metrics_df
    
    except Exception as e:
        save_error_log(e, config, experiment_dir)
        raise
    
    finally:
        finish_wandb()
        cleanup_resources(train_loader_factories, test_loader_factories, classifier)


if __name__ == "__main__":
    # example config for mnist digit generation
    config = {
        'experiment_type': 'digit_mnist',
        'method': 'standard_vcl',
        'num_tasks': 10,
        'epochs': 200,
        'lr': 1e-4,
        'batch_size': 128,
        'init_std': 0.001,
        'n_train_samples': 1,
        'n_eval_samples': 1000,
        'input_size': 784,
        'hidden_size': 500,
        'latent_size': 50,
        'use_classifier': True,
    }
    
    if config['method'] == 'coreset_vcl':
        config.update({
            'coreset_size': 200,
            'use_kcenter': False,
        })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_dir = Path('experiments/dgm/mnist_test')
    
    run_experiment(config, experiment_dir, device)