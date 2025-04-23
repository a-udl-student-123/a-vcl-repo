# gaussian vcl utils - loss funcs, metrics, eval helpers

import torch
import math

def convert_to_onehot(labels, num_classes=10):
    if labels.dim() > 1:
        return labels
        
    onehot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    return onehot.scatter_(1, labels.unsqueeze(1), 1.0)

def gaussian_nll_loss(mean, logvar, target, reduction='mean'):
    # add small eps to avoid numerical issues
    var = torch.exp(logvar) + 1e-10
    nll = 0.5 * (logvar + (target - mean).pow(2) / var)
    
    if reduction == 'none':
        return nll
    elif reduction == 'sum':
        return nll.sum()
    else:
        return nll.mean()

def compute_rmse(mean, target):
    mse = ((mean - target) ** 2).mean()
    return torch.sqrt(mse).item()

def compute_gradient_norm(model):
    total_norm = 0.0
    total_params = 0
    
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm(2).item() ** 2
            total_params += 1
            
    return (total_norm / total_params) ** 0.5 if total_params > 0 else 0

def compute_vcl_model_stats(model):
    # track weight/bias stats across model
    stats = {
        'weight_mu_abs': 0.0, 'weight_mu_count': 0,
        'bias_mu_abs': 0.0, 'bias_mu_count': 0,
        'weight_std_avg': 0.0, 'weight_std_count': 0,
        'bias_std_avg': 0.0, 'bias_std_count': 0
    }
    
    for layer_name in ['lin1', 'lin2', 'mean_head', 'logvar_head']:
        layer = getattr(model, layer_name)
        
        stats['weight_mu_abs'] += torch.abs(layer.weight_mu).sum().item()
        stats['weight_mu_count'] += layer.weight_mu.numel()
        stats['bias_mu_abs'] += torch.abs(layer.bias_mu).sum().item() 
        stats['bias_mu_count'] += layer.bias_mu.numel()
        
        weight_std = layer._get_sigma(layer.weight_rho)
        bias_std = layer._get_sigma(layer.bias_rho)
        
        stats['weight_std_avg'] += weight_std.sum().item()
        stats['weight_std_count'] += weight_std.numel()
        stats['bias_std_avg'] += bias_std.sum().item()
        stats['bias_std_count'] += bias_std.numel()
    
    results = {
        'avg_weight_mu_abs': stats['weight_mu_abs'] / stats['weight_mu_count'] if stats['weight_mu_count'] > 0 else 0,
        'avg_bias_mu_abs': stats['bias_mu_abs'] / stats['bias_mu_count'] if stats['bias_mu_count'] > 0 else 0,
        'avg_weight_std': stats['weight_std_avg'] / stats['weight_std_count'] if stats['weight_std_count'] > 0 else 0,
        'avg_bias_std': stats['bias_std_avg'] / stats['bias_std_count'] if stats['bias_std_count'] > 0 else 0
    }
    
    return results

def clean_memory(device='cuda', sleep_time=0):
    import gc
    gc.collect()
    
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if sleep_time > 0:
        import time
        time.sleep(sleep_time)
        gc.collect()

def evaluate_model_gaussian(model, dataloader, device='cuda', n_samples=100):
    model.eval()
    model = model.to(device)
    
    total_squared_error = 0.0
    total_correct = 0
    total_nll = 0.0
    total_variance = 0.0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            
            labels = convert_to_onehot(labels, num_classes=model.mean_head.out_features)
            labels = labels.to(device, non_blocking=True)
            
            means, logvars, aleatoric_uncertainty, epistemic_uncertainty = model(inputs, n_samples=n_samples)
            
            squared_errors = ((means - labels) ** 2).sum()
            batch_nll = gaussian_nll_loss(means, logvars, labels)
            batch_variance = aleatoric_uncertainty.mean().item() + epistemic_uncertainty.mean().item()
            
            pred_indices = torch.argmax(means, dim=1)
            target_indices = torch.argmax(labels, dim=1)
            batch_correct = (pred_indices == target_indices).float().sum()
            
            batch_size = inputs.size(0)
            total_squared_error += squared_errors.item()
            total_nll += batch_nll.item() * batch_size
            total_variance += batch_variance * batch_size
            total_correct += batch_correct.item()
            total += batch_size
            
            del inputs, labels, means, logvars
    
    kl_loss = model.kl_loss().item()
    
    dim = model.mean_head.out_features
    rmse = math.sqrt(total_squared_error / (total * dim)) if total>0 else 0
    accuracy = total_correct / total if total > 0 else 0
    nll_loss = total_nll / total if total > 0 else 0
    avg_variance = total_variance / total if total > 0 else 0
    
    total_loss = nll_loss + kl_loss / total if total > 0 else 0
    
    clean_memory(device)
    
    return rmse, accuracy, nll_loss, kl_loss, total_loss, avg_variance

def evaluate_all_tasks_gaussian(model, test_loaders, current_task, device, n_eval_samples, record_metric_fn=None):
    from tqdm import tqdm
    
    model.eval()
    rmse_values = []
    accuracy_values = []
    nll_values = []
    kl_values = []
    loss_values = []
    variance_values = []
    
    num_tasks_to_eval = min(current_task, len(test_loaders))
    
    eval_pbar = tqdm(range(num_tasks_to_eval), desc="Task Eval", leave=False)
    for task_idx in eval_pbar:
        rmse, accuracy, nll, kl, loss, variance = evaluate_model_gaussian(
            model, test_loaders[task_idx], device, n_eval_samples
        )
        
        if record_metric_fn:
            t_idx = current_task - 1
            record_metric_fn(t_idx, -1, f'rmse_on_task_{task_idx+1}', rmse)
            record_metric_fn(t_idx, -1, f'accuracy_on_task_{task_idx+1}', accuracy)
            record_metric_fn(t_idx, -1, f'nll_on_task_{task_idx+1}', nll)
            record_metric_fn(t_idx, -1, f'kl_on_task_{task_idx+1}', kl)
            record_metric_fn(t_idx, -1, f'loss_on_task_{task_idx+1}', loss)
            record_metric_fn(t_idx, -1, f'variance_on_task_{task_idx+1}', variance)
        
        rmse_values.append(rmse)
        accuracy_values.append(accuracy)
        nll_values.append(nll)
        kl_values.append(kl)
        loss_values.append(loss)
        variance_values.append(variance)
        
        eval_pbar.set_postfix(
            rmse=f"{rmse:.4f}", 
            acc=f"{accuracy:.4f}", 
            nll=f"{nll:.4f}",
            var=f"{variance:.4f}",
            loss=f"{loss:.4f}", 
            task=f"{task_idx+1}"
        )
    
    clean_memory(device)
    
    avg_rmse = sum(rmse_values) / len(rmse_values) if rmse_values else 0
    avg_accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0
    avg_nll = sum(nll_values) / len(nll_values) if nll_values else 0
    avg_kl = sum(kl_values) / len(kl_values) if kl_values else 0
    avg_loss = sum(loss_values) / len(loss_values) if loss_values else 0
    avg_variance = sum(variance_values) / len(variance_values) if variance_values else 0
    
    metrics = {
        'rmse_values': rmse_values,
        'accuracy_values': accuracy_values,
        'nll_values': nll_values,
        'kl_values': kl_values,
        'loss_values': loss_values,
        'variance_values': variance_values,
        'avg_rmse': avg_rmse,
        'avg_accuracy': avg_accuracy,
        'avg_nll': avg_nll,
        'avg_kl': avg_kl,
        'avg_loss': avg_loss,
        'avg_variance': avg_variance
    }
    
    return metrics

def print_eval_results_gaussian(task_idx, metrics):
    print(f"    Evaluation Results after Task {task_idx}:")
    
    rmse_values = metrics['rmse_values']
    accuracy_values = metrics['accuracy_values']
    nll_values = metrics['nll_values']
    kl_values = metrics['kl_values']
    loss_values = metrics['loss_values']
    variance_values = metrics['variance_values']
    
    for idx, (rmse, acc, nll, kl, loss, var) in enumerate(zip(
        rmse_values, accuracy_values, nll_values, kl_values, loss_values, variance_values), start=1):
        print(f"    Task {idx}: RMSE={rmse:.4f}, Acc={acc:.4f}, NLL={nll:.4f}, KL={kl:.4f}, Variance={var:.4f}, Loss={loss:.4f}")
    
    print(f"    Average: RMSE={metrics['avg_rmse']:.4f}, Acc={metrics['avg_accuracy']:.4f}, " +
          f"NLL={metrics['avg_nll']:.4f}, KL={metrics['avg_kl']:.4f}, Variance={metrics['avg_variance']:.4f}, Loss={metrics['avg_loss']:.4f}")
    print("    " + "-" * 100 + "\n")

def compute_accuracy(mean, target):
    pred_indices = torch.argmax(mean, dim=1)
    target_indices = torch.argmax(target, dim=1)
    
    correct = (pred_indices == target_indices).float().sum()
    total = target.size(0)
    
    return (correct / total).item()