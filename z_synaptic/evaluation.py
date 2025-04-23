# evaluation functions for synaptic intelligence
import torch
from tqdm import tqdm
from z_utils.utils import clean_memory

def evaluate_model(model, dataloader, device='cuda'):
    # get accuracy and loss for a model on given data
    model.eval()
    model = model.to(device)
    
    correct = 0
    total = 0
    total_ce_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = inputs.size(0)
            
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += batch_size
            total_ce_loss += loss.item()
            
            del inputs, labels, logits
    
    clean_memory(device)
    
    accuracy = correct / total if total > 0 else 0
    avg_ce_loss = total_ce_loss / total if total > 0 else 0
    
    return accuracy, avg_ce_loss

def evaluate_all_tasks(model, test_loaders, num_tasks_seen, device, record_metric_fn=None):
    # evaluate model on all tasks seen so far and return metrics
    model.eval()
    accuracies = []
    ce_losses = []
    
    # save original task for multi-head models
    original_task = None
    if hasattr(model, 'current_task'):
        original_task = model.current_task
    
    num_tasks_to_eval = min(num_tasks_seen, len(test_loaders))
    
    eval_pbar = tqdm(range(num_tasks_to_eval), desc="Task Eval", leave=False)
    for task_idx in eval_pbar:
        if hasattr(model, 'set_current_task'):
            model.set_current_task(task_idx)
        
        accuracy, ce_loss = evaluate_model(model, test_loaders[task_idx], device)
        
        if record_metric_fn is not None:
            t_idx = num_tasks_seen - 1  # 0-based
            record_metric_fn(t_idx, -1, f'accuracy_on_task_{task_idx+1}', accuracy)
            record_metric_fn(t_idx, -1, f'ce_loss_on_task_{task_idx+1}', ce_loss)
        
        accuracies.append(accuracy)
        ce_losses.append(ce_loss)
        eval_pbar.set_postfix(accuracy=f"{accuracy:.4f}", task=f"{task_idx+1}")
    
    # restore original task head
    if original_task is not None and hasattr(model, 'set_current_task'):
        model.set_current_task(original_task)
    
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