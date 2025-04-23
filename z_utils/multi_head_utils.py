# utils for multi-head for VCL

import torch
from tqdm import tqdm
from z_utils.utils import clean_memory

def initialize_multi_head_model(model_class, train_loader, test_loader, device, 
                             use_ml_initialization, ml_epochs, lr, 
                             init_std, adaptive_std, adaptive_std_epsilon, exp_dir=None):
    model = model_class().to(device)
    
    if use_ml_initialization:
        is_flexible = hasattr(model, 'shared_layers')
        
        if is_flexible:
            from z_models.ml_initialization import get_ml_initialized_vcl_model
            
            model = get_ml_initialized_vcl_model(
                train_loader=train_loader,
                vcl_model=model,
                test_loader=test_loader,
                ml_epochs=ml_epochs,
                lr=lr,
                init_std=init_std,
                adaptive_std=adaptive_std,
                adaptive_std_epsilon=adaptive_std_epsilon,
                device=device,
                exp_dir=exp_dir
            )
        else:
            from z_models.ml_initialization import train_standard_mlp
            
            input_size = model.lin1.in_features
            hidden_size = model.lin1.out_features
            head_size = model.heads[0].out_features
            
            ml_model = train_standard_mlp(
                train_loader, test_loader=test_loader, max_epochs=ml_epochs, 
                lr=5e-3, device=device,
                input_size=input_size, hidden_size=hidden_size, output_size=head_size,
                exp_dir=exp_dir
            )
            
            with torch.no_grad():
                for i in range(1, 3):
                    mh_layer = getattr(model, f"lin{i}")
                    ml_layer = getattr(ml_model, f"lin{i}")
                    mh_layer.weight_mu.copy_(ml_layer.weight.to(device))
                    mh_layer.bias_mu.copy_(ml_layer.bias.to(device))
                
                head = model.heads[0]
                head.weight_mu.copy_(ml_model.lin3.weight.to(device))
                head.bias_mu.copy_(ml_model.lin3.bias.to(device))
            
            model.set_init_std(init_std, adaptive_std, adaptive_std_epsilon)
    else:
        model.set_init_std(init_std, adaptive_std, adaptive_std_epsilon)
    
    model.set_current_task(0)
    return model

def evaluate_multi_head_model(model, dataloader, task_idx, device='cuda', n_samples=100):
    model.eval()
    model = model.to(device)
    
    current_task = model.current_task
    
    if hasattr(model, 'set_current_task'):
        model.set_current_task(task_idx)
    
    correct = 0
    total = 0
    total_ce_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = inputs.size(0)
            
            probs = model.predict_softmax_samples(inputs, task_idx=task_idx, n_samples=n_samples)
            
            _, predicted = torch.max(probs, dim=1)
            correct += (predicted == labels).sum().item()
            total += batch_size
            
 
            true_class_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            batch_ce_loss = -torch.log(true_class_probs + 1e-10).mean().item()
            total_ce_loss += batch_ce_loss * batch_size
            
            del inputs, labels, probs, predicted
    
    if hasattr(model, 'set_current_task'):
        model.set_current_task(current_task)
    
    clean_memory(device)
    
    accuracy = correct / total if total > 0 else 0
    avg_ce_loss = total_ce_loss / total if total > 0 else 0
    
    return accuracy, avg_ce_loss

def evaluate_all_tasks_multi_head(model, test_loaders, current_task, device, n_eval_samples, record_metric_fn=None):
    # evaluate model on all tasks seen so far
    model.eval()
    accuracies = []
    ce_losses = []
    
    num_tasks_to_eval = current_task
    if hasattr(model, 'num_tasks'):
        num_tasks_to_eval = min(num_tasks_to_eval, model.num_tasks)
    num_tasks_to_eval = min(num_tasks_to_eval, len(test_loaders))
    
    eval_pbar = tqdm(range(num_tasks_to_eval), desc="Task Eval", leave=False)
    for task_idx in eval_pbar:
        accuracy, ce_loss = evaluate_multi_head_model(
            model, test_loaders[task_idx], task_idx, device, n_eval_samples
        )
        
        if record_metric_fn:
            t_idx = current_task - 1
            record_metric_fn(t_idx, -1, f'accuracy_on_task_{task_idx+1}', accuracy)
            record_metric_fn(t_idx, -1, f'ce_loss_on_task_{task_idx+1}', ce_loss)
        
        accuracies.append(accuracy)
        ce_losses.append(ce_loss)
        eval_pbar.set_postfix(accuracy=f"{accuracy:.4f}", task=f"{task_idx+1}")
    
    clean_memory(device)
    
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    avg_ce_loss = sum(ce_losses) / len(ce_losses) if ce_losses else 0
    
    return accuracies, ce_losses, avg_accuracy, avg_ce_loss

def prepare_task(model, task_idx, is_prediction_model=False):
    if hasattr(model, 'set_current_task'):
        model.set_current_task(task_idx)
        
    if not is_prediction_model and task_idx > 0 and hasattr(model, 'initialize_new_head'):
        model.initialize_new_head(task_idx)
        
    return model

def modify_vcl_for_multi_head(train_vcl_fn):
    def multi_head_train_vcl(*args, **kwargs):
        kwargs.setdefault('evaluate_all_tasks_fn', evaluate_all_tasks_multi_head)
        kwargs.setdefault('initialize_model_fn', initialize_multi_head_model)
        kwargs.setdefault('pre_task_hook', prepare_task)
        
        return train_vcl_fn(*args, **kwargs)
    
    return multi_head_train_vcl