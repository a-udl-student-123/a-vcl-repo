

import torch
import gc
from tqdm import tqdm

def clean_memory(device='cuda', sleep_time=0):
    gc.collect()
    
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if sleep_time > 0:
        import time
        time.sleep(sleep_time)
        gc.collect()

def clean_loader(loader):
    # cleanup dataloader workers and refs
    if loader is None:
        return
        
    if hasattr(loader, '_iterator') and loader._iterator is not None:
        try:
            loader._iterator._shutdown_workers()
        except Exception as e:
            print(f"Warning: worker shutdown failed: {str(e)}")
    
    try:
        if hasattr(loader, '_iterator'):
            loader._iterator = None
    except Exception as e:
        print(f"Warning: ref cleanup failed: {str(e)}")
    
    gc.collect()

def calculate_batch_metrics(probs, labels):
    # get acc and loss for a batch
    batch_size = labels.size(0)
    
    _, predicted = torch.max(probs, dim=1)
    correct = (predicted == labels).sum().item()
    
    true_class_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    batch_ce_loss = -torch.log(true_class_probs + 1e-10).mean().item()
    
    return {
        'correct': correct,
        'total': batch_size,
        'ce_loss': batch_ce_loss * batch_size
    }

def evaluate_model(model, dataloader, device='cuda', n_samples=100):
    model.eval()
    model = model.to(device)
    
    correct = 0
    total = 0  
    total_ce_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if hasattr(model, 'predict_softmax_samples'):
                probs = model.predict_softmax_samples(inputs, n_samples=n_samples)
            else:
                probs = model.predict_softmax(inputs)
            
            metrics = calculate_batch_metrics(probs, labels)
            correct += metrics['correct']
            total += metrics['total']
            total_ce_loss += metrics['ce_loss']
            
            del inputs, labels, probs
    
    clean_memory(device)
    
    accuracy = correct / total if total > 0 else 0
    avg_ce_loss = total_ce_loss / total if total > 0 else 0
    
    return accuracy, avg_ce_loss

def record_task_metrics(record_fn, current_task, task_idx, accuracy, ce_loss):
    if record_fn is None:
        return
        
    t_idx = current_task - 1  # 0-base
    record_fn(t_idx, -1, f'accuracy_on_task_{task_idx+1}', accuracy)
    record_fn(t_idx, -1, f'ce_loss_on_task_{task_idx+1}', ce_loss)

def evaluate_single_task(model, loader, task_idx, device, n_samples, record_metric_fn=None, current_task=None):
    accuracy, ce_loss = evaluate_model(model, loader, device, n_samples)
    
    if record_metric_fn is not None and current_task is not None:
        record_task_metrics(record_metric_fn, current_task, task_idx, accuracy, ce_loss)
    
    return accuracy, ce_loss

def evaluate_all_tasks(model, test_loaders, current_task, device, n_eval_samples, record_metric_fn=None):

    model.eval()
    accuracies = []
    ce_losses = []
    
    num_tasks_to_eval = min(current_task, len(test_loaders))
    
    eval_pbar = tqdm(range(num_tasks_to_eval), desc="Task Eval", leave=False)
    for task_idx in eval_pbar:
        accuracy, ce_loss = evaluate_single_task(
            model, test_loaders[task_idx], task_idx, 
            device, n_eval_samples, record_metric_fn, current_task
        )
        
        accuracies.append(accuracy)
        ce_losses.append(ce_loss)
        eval_pbar.set_postfix(accuracy=f"{accuracy:.4f}", task=f"{task_idx+1}")
    
    clean_memory(device)
    
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    avg_ce_loss = sum(ce_losses) / len(ce_losses) if ce_losses else 0
    
    return accuracies, ce_losses, avg_accuracy, avg_ce_loss

def print_eval_results(task_idx, accuracies, ce_losses):
    print(f"    Evaluation Results after Task {task_idx}:")
    
    for idx, (acc, ce) in enumerate(zip(accuracies, ce_losses), start=1):
        print(f"    Task {idx}: Acc={acc:.4f} | average CE={ce:.4f}")
    
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    avg_ce_loss = sum(ce_losses) / len(ce_losses) if ce_losses else 0
    
    print(f"    Average: Acc={avg_accuracy:.4f} | average CE={avg_ce_loss:.4f}")
    print("    " + "-" * 60 + "\n")
    
    return avg_accuracy, avg_ce_loss
