# core  for running vcl exps

import torch
import pandas as pd
from pathlib import Path
import gc
import wandb

from z_data.datasets import generate_permutations, create_permuted_mnist_loader_factories
from z_models.vcl_models import VCL_MLP
from z_core.vcl import train_vcl
from z_utils.wandb_utils import init_wandb, log_training_metrics, log_evaluation_metrics, finish_wandb
import warnings

warnings.filterwarnings("ignore", message="A newer version of deeplake.*", category=UserWarning, module="deeplake.util.check_latest_version")

def setup_experiment_dirs(experiment_dir):
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "plots").mkdir(exist_ok=True)
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    return experiment_dir

def cleanup_resources(train_loader_factories=None, test_loader_factories=None, permutations=None):
    if train_loader_factories is not None:
        del train_loader_factories
    if test_loader_factories is not None:
        del test_loader_factories
    if permutations is not None:
        del permutations
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    gc.collect()

def create_data_loader_factories(config, num_workers):
    experiment_type = config.get("experiment_type", "permuted_mnist")
    batch_size = config.get("batch_size", 256)
    num_tasks = config.get("num_tasks", 10)
    
    if experiment_type == "split_mnist":
        from z_data.datasets import create_split_mnist_loader_factories
        single_batch = config.get("single_batch", True)
        
        train_loader_factories = create_split_mnist_loader_factories(
            root='data/',
            batch_size=batch_size,
            train=True,
            num_workers=num_workers,
            single_batch=single_batch
        )
        
        test_loader_factories = create_split_mnist_loader_factories(
            root='data/',
            batch_size=512,
            train=False,
            num_workers=num_workers,
            single_batch=False
        )
        
        permutations = None
        
    elif experiment_type == "split_notmnist":
        from z_data.datasets import create_split_notmnist_loader_factories
        single_batch = config.get("single_batch", True)
        
        train_loader_factories = create_split_notmnist_loader_factories(
            root='data/',
            batch_size=batch_size,
            train=True,
            num_workers=num_workers,
            single_batch=single_batch
        )
        
        test_loader_factories = create_split_notmnist_loader_factories(
            root='data/',
            batch_size=512,
            train=False,
            num_workers=num_workers,
            single_batch=False
        )
        
        permutations = None
        
    else:  # p-mnist
        from z_data.datasets import generate_permutations, create_permuted_mnist_loader_factories
        permutations = generate_permutations(num_tasks=num_tasks, device='cpu')
        
        train_loader_factories = create_permuted_mnist_loader_factories(
            root='data/',
            permutations=permutations,
            batch_size=batch_size,
            train=True,
            num_workers=num_workers
        )
        
        test_loader_factories = create_permuted_mnist_loader_factories(
            root='data/',
            permutations=permutations,
            batch_size=batch_size*2,
            train=False,
            num_workers=num_workers
        )
    
    return train_loader_factories, test_loader_factories, permutations

def create_model_factory(config, device):
    experiment_type = config.get("experiment_type", "permuted_mnist")
    hidden_size = config.get("hidden_size", 100)
    init_std = config.get("init_std", None)
    
    if experiment_type == "split_mnist":
        from z_models.vcl_models import VCL_MultiHead_MLP
        num_tasks = 5  
        
        def create_model():
            return VCL_MultiHead_MLP(
                input_size=784, 
                hidden_size=hidden_size,
                num_tasks=num_tasks,
                head_size=2,
                init_std=init_std
            ).to(device)
            
        return create_model
    
    elif experiment_type == "split_notmnist":
        from z_models.vcl_models import VCL_FlexibleMultiHead_MLP
        num_tasks = 5
        hidden_sizes = [150, 150, 150, 150] 
        
        def create_model():
            return VCL_FlexibleMultiHead_MLP(
                input_size=784, 
                hidden_sizes=hidden_sizes,
                num_tasks=num_tasks,
                head_size=2,
                init_std=init_std
            ).to(device)
            
        return create_model
        
    else:
        from z_models.vcl_models import VCL_MLP
        
        def create_model():
            return VCL_MLP(
                input_size=784, 
                hidden_size=hidden_size, 
                output_size=10
            ).to(device)
            
        return create_model

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
    def record_metric_with_wandb(task, epoch, name, value):
        value_to_record = float(value) if isinstance(value, (int, float)) else value
        metrics.append({
            'task': task,
            'epoch': epoch,
            'metric': name,
            'value': value_to_record
        })
        
        if name.startswith('train_prediction'):
            return 
        
        if epoch >= 0 and task >= 0 and name.startswith('train_') and '_bias_mu_abs' in name:
            prefix = name.rsplit('_bias_mu_abs', 1)[0]
            
            epoch_metrics = {
                'avg_weight_std': value_to_record,
                'avg_bias_std': next((m['value'] for m in metrics if m['task'] == task and 
                                    m['epoch'] == epoch and m['metric'] == f"{prefix}_bias_std"), 0),
                'avg_weight_mu_abs': next((m['value'] for m in metrics if m['task'] == task and 
                                        m['epoch'] == epoch and m['metric'] == f"{prefix}_weight_mu_abs"), 0),
                'avg_bias_mu_abs': next((m['value'] for m in metrics if m['task'] == task and 
                                        m['epoch'] == epoch and m['metric'] == f"{prefix}_bias_mu_abs"), 0),
                'avg_loss': next((m['value'] for m in metrics if m['task'] == task and 
                                m['epoch'] == epoch and m['metric'] == f"{prefix}_loss"), 0),
                'avg_ce_loss': next((m['value'] for m in metrics if m['task'] == task and 
                                    m['epoch'] == epoch and m['metric'] == f"{prefix}_ce_loss"), 0),
                'avg_kl_loss': next((m['value'] for m in metrics if m['task'] == task and 
                                    m['epoch'] == epoch and m['metric'] == f"{prefix}_kl_loss"), 0),
                'accuracy': next((m['value'] for m in metrics if m['task'] == task and 
                                m['epoch'] == epoch and m['metric'] == f"{prefix}_accuracy"), 0),
                'avg_grad_norm': next((m['value'] for m in metrics if m['task'] == task and 
                                    m['epoch'] == epoch and m['metric'] == f"{prefix}_grad_norm"), 0),
            }
            log_training_metrics(epoch_metrics, task, epoch, config['epochs'])
        
        if epoch < 0 and name == 'average_accuracy':
            log_wandb_evaluation_metrics(metrics, task, config['epochs'])
            
    return record_metric_with_wandb

def log_wandb_evaluation_metrics(metrics, task, epochs_per_task):
    task_accuracies = []
    task_ce_losses = []
    task_forgetting = []
    max_task_idx = task + 1
    
    for t in range(max_task_idx):
        accuracy = next((m['value'] for m in metrics if m['task'] == task and 
                     m['metric'] == f'accuracy_on_task_{t+1}'), None)
        if accuracy is not None:
            task_accuracies.append(accuracy)
            
            ce_loss = next((m['value'] for m in metrics if m['task'] == task and 
                        m['metric'] == f'ce_loss_on_task_{t+1}'), 0)
            task_ce_losses.append(ce_loss)
            
            if t < task:
                max_acc = max([m['value'] for m in metrics if 
                            m['metric'] == f'accuracy_on_task_{t+1}' and 
                            m['task'] <= task] or [0])
                if max_acc > 0:
                    forgetting = max_acc - accuracy
                    task_forgetting.append(forgetting)
    
    avg_accuracy = next((m['value'] for m in metrics if m['task'] == task and 
                     m['metric'] == 'average_accuracy'), 0)
    avg_ce_loss = next((m['value'] for m in metrics if m['task'] == task and 
                     m['metric'] == 'average_ce_loss'), 0)
    
    log_evaluation_metrics({
        "task_accuracies": task_accuracies,
        "task_ce_losses": task_ce_losses,
        "avg_accuracy": avg_accuracy,
        "avg_ce_loss": avg_ce_loss,
        "task_forgetting": task_forgetting if task_forgetting else None
    }, task, epochs_per_task)

def save_error_log(error, config, experiment_dir):
    import traceback
    from datetime import datetime
    
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
    
    train_loader_factories, test_loader_factories, permutations = create_data_loader_factories(
        config, config['num_workers']
    )
    model_factory = create_model_factory(config, device)
    
    record_metric(0, -1, 'init_std', config.get('init_std', 0.001))
    record_metric(0, -1, 'adaptive_std', 1 if config.get('adaptive_std', False) else 0)
    record_metric(0, -1, 'adaptive_std_epsilon', config.get('adaptive_std_epsilon', 0.01))
    
    try:
        vcl_params = {
            'model_class': model_factory,
            'train_loader_factories': train_loader_factories,
            'test_loader_factories': test_loader_factories,
            'epochs_per_task': config['epochs'],
            'lr': config['lr'],
            'device': device,
            'num_workers': config['num_workers'],
            'record_metric_fn': record_metric,
            'use_ml_initialization': config['use_ml_initialization'],
            'ml_epochs': config['ml_epochs'],
            'exp_dir': experiment_dir,
            'n_eval_samples': config.get('n_eval_samples', 100),
            'n_train_samples': config.get('n_train_samples', 3),
            'init_std': config.get('init_std', 0.001),
            'adaptive_std': config.get('adaptive_std', False),
            'adaptive_std_epsilon': config.get('adaptive_std_epsilon', 0.01),
            'different_perm_init': config.get('different_perm_init', False),
            'early_stopping_threshold': config.get('early_stopping_threshold', None),
        }
        
        needs_multi_head = config.get("experiment_type") in ["split_mnist", "split_notmnist"]
        
        use_coreset = config.get('method') != 'standard_vcl'
        
        if use_coreset:
            vcl_params['coreset_size'] = config.get('coreset_size', 0)
            vcl_params['use_kcenter'] = config.get('use_kcenter', False)
            vcl_params['kcenter_batch_size'] = config.get('kcenter_batch_size', 1024)
        else:
            vcl_params['coreset_size'] = 0
        
        from z_core.vcl import train_vcl
        
        if needs_multi_head or config.get('method') == 'multi_head_vcl':
            from z_utils.multi_head_utils import modify_vcl_for_multi_head
            train_vcl_fn = modify_vcl_for_multi_head(train_vcl)
            
            vcl_params['use_task_specific_prediction'] = config.get('use_task_specific_prediction', True)
        else:
            train_vcl_fn = train_vcl
        
        trained_model = train_vcl_fn(**vcl_params)
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(experiment_dir / "metrics.csv", index=False)
        
        return metrics_df
    
    except Exception as e:
        save_error_log(e, config, experiment_dir)
        raise
    
    finally:
        cleanup_resources(train_loader_factories, test_loader_factories, permutations)

def run_experiment_with_wandb(config, experiment_dir, project_name, device='cuda'):
    experiment_dir = setup_experiment_dirs(experiment_dir)
    
    if "experiment_type" not in config:
        if config.get("method") == "multi_head_vcl":
            config["experiment_type"] = "split_mnist"
        else:
            config["experiment_type"] = "permuted_mnist"
    
    init_wandb(config, project_name=project_name, 
               tags=[config["method"], f"tasks={config['num_tasks']}", config["experiment_type"]])
    
    metrics = []
    record_metric_with_wandb = create_wandb_metric_recorder(config, metrics)
    
    train_loader_factories, test_loader_factories, permutations = create_data_loader_factories(
        config, config['num_workers']
    )
    model_factory = create_model_factory(config, device)
    
    record_metric_with_wandb(0, -1, 'init_std', config.get('init_std', 0.001))
    record_metric_with_wandb(0, -1, 'adaptive_std', 1 if config.get('adaptive_std', False) else 0)
    record_metric_with_wandb(0, -1, 'adaptive_std_epsilon', config.get('adaptive_std_epsilon', 0.01))
    
    try:
        vcl_params = {
            'model_class': model_factory,
            'train_loader_factories': train_loader_factories,
            'test_loader_factories': test_loader_factories,
            'epochs_per_task': config['epochs'],
            'lr': config['lr'],
            'device': device,
            'num_workers': config['num_workers'],
            'record_metric_fn': record_metric_with_wandb,
            'use_ml_initialization': config['use_ml_initialization'],
            'ml_epochs': config['ml_epochs'],
            'exp_dir': experiment_dir,
            'n_eval_samples': config.get('n_eval_samples', 100),
            'n_train_samples': config.get('n_train_samples', 3),
            'init_std': config.get('init_std', 0.001),
            'adaptive_std': config.get('adaptive_std', False),
            'adaptive_std_epsilon': config.get('adaptive_std_epsilon', 0.01),
            'different_perm_init': config.get('different_perm_init', False),
            'early_stopping_threshold': config.get('early_stopping_threshold', None),
        }
        
        if config['method'] == 'standard_vcl':
            vcl_params['coreset_size'] = 0
        else:
            vcl_params['coreset_size'] = config.get('coreset_size', 0)
            vcl_params['use_kcenter'] = config.get('use_kcenter', False)
            vcl_params['kcenter_batch_size'] = config.get('kcenter_batch_size', 1024)
        
        if config.get("experiment_type") in ["split_mnist", "split_notmnist"] or config['method'] == 'multi_head_vcl':
            from z_utils.multi_head_utils import modify_vcl_for_multi_head
            from z_core.vcl import train_vcl
            
            multi_head_train_vcl = modify_vcl_for_multi_head(train_vcl)
            
            vcl_params['use_task_specific_prediction'] = config.get('use_task_specific_prediction', True)
            
            trained_model = multi_head_train_vcl(**vcl_params)
        else:
            from z_core.vcl import train_vcl
            trained_model = train_vcl(**vcl_params)
        
        for plot_file in (experiment_dir / "plots").glob("*.png"):
            wandb.log({plot_file.stem: wandb.Image(str(plot_file))})
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(experiment_dir / "metrics.csv", index=False)
        
        return metrics_df
    
    except Exception as e:
        save_error_log(e, config, experiment_dir)
        raise
    
    finally:
        finish_wandb()
        cleanup_resources(train_loader_factories, test_loader_factories, permutations)