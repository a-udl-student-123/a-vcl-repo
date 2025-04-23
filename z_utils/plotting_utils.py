

import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import gc

def setup_plotting(prev_backend=None):
    if prev_backend is None:
        prev_backend = matplotlib.get_backend()
    matplotlib.use('Agg')
    plt.rcParams['figure.max_open_warning'] = 10
    return prev_backend

def cleanup_plotting(prev_backend=None):
    plt.close('all')
    plt.rcParams.update(plt.rcParamsDefault)
    
    if prev_backend:
        try:
            matplotlib.use(prev_backend)
        except:
            pass
    
    gc.collect()

def create_plots_dir(exp_dir, subdir=None):
    plots_dir = exp_dir / "plots"
    if subdir:
        plots_dir = plots_dir / subdir
    plots_dir.mkdir(exist_ok=True, parents=True)
    return plots_dir

def create_accuracy_plot(avg_accuracies, exp_dir):
    prev_backend = None
    try:
        prev_backend = setup_plotting()
        plots_dir = create_plots_dir(exp_dir)
        
        num_tasks = len(avg_accuracies)
        tasks = list(range(1, num_tasks + 1))
        
        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(tasks, avg_accuracies, '-', linewidth=1)
        
        plt.xlabel('Number of Tasks Trained')
        plt.ylabel('Average Accuracy')
        plt.title('Average Accuracy Across All Tasks vs. Number of Tasks Trained')
        plt.grid(True, alpha=0.3)
        plt.xticks(tasks)
        plt.ylim(0, 1.0)
        
        for i, acc in enumerate(avg_accuracies):
            plt.annotate(f'{acc:.4f}', (tasks[i], acc), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.savefig(plots_dir / "average_accuracy_progression.png", dpi=100, bbox_inches='tight')
        
    except Exception as e:
        print(f"Error creating accuracy plot: {str(e)}")
    finally:
        cleanup_plotting(prev_backend)

def create_task_specific_plots(task_accuracies, exp_dir):
    # plot per-task accuracy 
    prev_backend = None
    try:
        prev_backend = setup_plotting()
        plots_dir = create_plots_dir(exp_dir)
        
        num_tasks = len(task_accuracies)
        
        plt.figure(figsize=(12, 8), dpi=100)
        
        for task_idx in range(num_tasks):
            accuracies = [task_accuracies[train_idx][task_idx] 
                         for train_idx in range(task_idx, num_tasks)]
            train_steps = list(range(task_idx + 1, num_tasks + 1))
            
            plt.plot(train_steps, accuracies, '-', linewidth=1, 
                    label=f'Task {task_idx + 1}')
        
        plt.xlabel('Number of Tasks Trained')
        plt.ylabel('Accuracy')
        plt.title('Task-Specific Accuracy as Training Progresses')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, num_tasks + 1))
        plt.ylim(0, 1.0)
        plt.legend()
        plt.savefig(plots_dir / "all_tasks_accuracy.png", dpi=100, bbox_inches='tight')
        plt.close()
        
        for task_idx in range(num_tasks):
            accuracies = [task_accuracies[train_idx][task_idx] 
                         for train_idx in range(task_idx, num_tasks)]
            train_steps = list(range(task_idx + 1, num_tasks + 1))
            
            if not accuracies:
                continue
            
            plt.figure(figsize=(10, 6), dpi=100)
            plt.plot(train_steps, accuracies, '-', linewidth=1, color=f'C{task_idx % 10}')
            
            plt.axhline(y=accuracies[0], color=f'C{task_idx % 10}', 
                       linestyle='--', alpha=0.5,
                       label=f'Initial: {accuracies[0]:.4f}')
            
            for i, acc in enumerate(accuracies):
                plt.annotate(f'{acc:.4f}', (train_steps[i], acc), 
                            textcoords="offset points", xytext=(0,10), ha="center")
            
            plt.xlabel('Number of Tasks Trained')
            plt.ylabel(f'Accuracy on Task {task_idx + 1}')
            plt.title(f'Performance on Task {task_idx + 1} as Training Progresses')
            plt.grid(True, alpha=0.3)
            plt.xticks(range(1, num_tasks + 1))
            plt.ylim(0, 1.0)
            plt.legend()
            plt.savefig(plots_dir / f"task_{task_idx + 1}_accuracy.png", dpi=100, bbox_inches='tight')
            plt.close()
            
            if task_idx % 2 == 1:
                gc.collect()
    
    except Exception as e:
        print(f"Error creating task-specific plots: {str(e)}")
    finally:
        cleanup_plotting(prev_backend)

def plot_loss_components(epochs, loss_data, title_prefix, plots_dir, file_suffix):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_data['loss'], '-', label='Total Loss (CE + KL/N)', linewidth=1)
    plt.plot(epochs, loss_data['ce_loss'], '-', label='CE Loss', linewidth=1)
    plt.plot(epochs, loss_data['kl'], '-', label='KL/N Term', linewidth=1)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f'{title_prefix}: Loss Components', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / f"task_{file_suffix}_loss_components.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_accuracy(epochs, accuracy_data, title_prefix, plots_dir, file_suffix):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracy_data, '-', color='green', linewidth=1)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'{title_prefix}: Training Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(plots_dir / f"task_{file_suffix}_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_std_values(epochs, std_data, title_prefix, plots_dir, file_suffix):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, std_data['weight_std'], '-', label='Weight Std', linewidth=1)
    plt.plot(epochs, std_data['bias_std'], '-', label='Bias Std', linewidth=1)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Standard Deviation', fontsize=12)
    plt.title(f'{title_prefix}: Parameter Standard Deviations', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / f"task_{file_suffix}_std_values.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_param_magnitudes(epochs, magnitude_data, title_prefix, plots_dir, file_suffix, is_vcl=True):
    plt.figure(figsize=(10, 6))
    weight_label = '|Weight μ|' if is_vcl else '|Weight|'
    bias_label = '|Bias μ|' if is_vcl else '|Bias|'
    weight_key = 'weight_mu_abs' if is_vcl else 'weight_abs'
    bias_key = 'bias_mu_abs' if is_vcl else 'bias_abs'
    
    plt.plot(epochs, magnitude_data[weight_key], '-', label=weight_label, linewidth=1)
    plt.plot(epochs, magnitude_data[bias_key], '-', label=bias_label, linewidth=1)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Absolute Value', fontsize=12)
    plt.title(f'{title_prefix}: Parameter Magnitudes', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / f"task_{file_suffix}_param_magnitudes.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_gradient_norm(epochs, grad_norm_data, title_prefix, plots_dir, file_suffix):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, grad_norm_data, '-', color='purple', linewidth=1)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Gradient Norm', fontsize=12)
    plt.title(f'{title_prefix}: Gradient Norm', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"task_{file_suffix}_grad_norm.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_loss_accuracy_combined(epochs, metrics_data, plots_dir, file_name="ml_loss_accuracy.png"):
    # plot loss and acc togeter
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12, color=color)
    ax1.plot(epochs, metrics_data['loss'], '-', color=color, linewidth=1, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Accuracy', fontsize=12, color=color)
    ax2.plot(epochs, metrics_data['accuracy'], '-', color=color, linewidth=1, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.0)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.title('ML Initialization: Loss and Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / file_name, dpi=150, bbox_inches='tight')
    plt.close()

def has_data(metrics, *keys):
    return all(key in metrics and len(metrics[key]) > 0 for key in keys)

def create_task_training_plots(task_idx, epoch_metrics, dataset_size, exp_dir, model_type=None):
    prev_backend = None
    try:
        prev_backend = setup_plotting()
        plots_dir = create_plots_dir(exp_dir, "training_curves")
        
        epochs = list(range(1, len(epoch_metrics.get('loss', [])) + 1))
        if not epochs:
            return
            
        suffix = f"{task_idx}{('_' + model_type) if model_type else ''}"
        title_prefix = f"Task {task_idx}" + (f" ({model_type.capitalize()} Model)" if model_type else "")
        
        if has_data(epoch_metrics, 'loss', 'ce_loss', 'kl'):
            plot_loss_components(epochs, epoch_metrics, title_prefix, plots_dir, suffix)
        
        if has_data(epoch_metrics, 'accuracy'):
            plot_accuracy(epochs, epoch_metrics['accuracy'], title_prefix, plots_dir, suffix)
        
        if has_data(epoch_metrics, 'weight_std', 'bias_std'):
            plot_std_values(epochs, epoch_metrics, title_prefix, plots_dir, suffix)
        
        if has_data(epoch_metrics, 'weight_mu_abs', 'bias_mu_abs'):
            plot_param_magnitudes(epochs, epoch_metrics, title_prefix, plots_dir, suffix, is_vcl=True)
        
        if has_data(epoch_metrics, 'grad_norm') and any(v > 0 for v in epoch_metrics['grad_norm']):
            plot_gradient_norm(epochs, epoch_metrics['grad_norm'], title_prefix, plots_dir, suffix)
        
    except Exception as e:
        model_str = f" ({model_type})" if model_type else ""
        print(f"Error creating training plots for task {task_idx}{model_str}: {str(e)}")
    finally:
        cleanup_plotting(prev_backend)

def create_ml_training_plots(ml_metrics, exp_dir):
    prev_backend = None
    try:
        prev_backend = setup_plotting()
        plots_dir = create_plots_dir(exp_dir, "ml_initialization")
        
        epochs = list(range(1, len(ml_metrics.get('loss', [])) + 1))
        if not epochs:
            return
        
        if has_data(ml_metrics, 'loss', 'accuracy'):
            plot_loss_accuracy_combined(epochs, ml_metrics, plots_dir)
        
        if has_data(ml_metrics, 'weight_abs', 'bias_abs'):
            plot_param_magnitudes(epochs, ml_metrics, 'ML Initialization', plots_dir, 'ml', is_vcl=False)
        
        if has_data(ml_metrics, 'grad_norm') and any(v > 0 for v in ml_metrics['grad_norm']):
            plot_gradient_norm(epochs, ml_metrics['grad_norm'], 'ML Initialization', plots_dir, 'ml')
            
    except Exception as e:
        print(f"Error creating ML initialization plots: {str(e)}")
    finally:
        cleanup_plotting(prev_backend)