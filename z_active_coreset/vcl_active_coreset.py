# handles active coreset selection for vcl training
# supports single/multi head architectures

import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

from z_utils.training_utils import (
    train_model_for_task, setup_directories,
    save_checkpoint, create_plots
)
from z_utils.utils import (
    clean_loader, clean_memory,
    evaluate_all_tasks, print_eval_results
)
from z_core.coreset import (
    initialize_coreset, active_select_coreset,
    update_coreset
)
from z_core.vcl import (
    initialize_model_for_vcl, train_propagation_model,
    train_prediction_model, create_task_specific_models,
    evaluate_models
)

def train_vcl_active(
    model_class,
    train_loader_factories,
    test_loader_factories,
    coreset_size,
    lambda_mix=0.5,
    use_kcenter=False,
    kcenter_batch_size=1024,
    epochs_per_task=5,
    pred_epochs_multiplier=1.0,
    lr=1e-3,
    device='cuda',
    num_workers=0,
    record_metric_fn=None,
    exp_dir=None,
    use_ml_initialization=False,
    ml_epochs=5,
    n_train_samples=5,
    n_eval_samples=100,
    init_std=0.001,
    adaptive_std=False,
    adaptive_std_epsilon=0.01,
    initialize_model_fn=None,
    evaluate_all_tasks_fn=None,
    pre_task_hook=None,
    different_perm_init=False,
    use_task_specific_prediction=False,
    early_stopping_threshold=None
):
    """trains vcl model with active coreset selection after each task
    
    model_class: factory func that creates model instances
    train_loader_factories: list of funcs that create training dataloaders
    test_loader_factories: list of funcs that create test dataloaders  
    coreset_size: num examples to store per task
    lambda_mix: mix between uncertainity and diversity for selection
    use_kcenter: whether to use k-center for diversity
    epochs_per_task: num epochs to train each task
    device: cuda/cpu
    exp_dir: where to save results/checkpoints
    """
    start_time = time.time()
    exp_dir, ckpt_dir=setup_directories(exp_dir)  
    coreset_ds = initialize_coreset()

    first_train = train_loader_factories[0]()
    ml_test = test_loader_factories[0](force_persistent=True)
    propagation_model = initialize_model_for_vcl(
        model_class, first_train, ml_test, device,
        use_ml_initialization, ml_epochs, lr, init_std,
        adaptive_std, adaptive_std_epsilon, exp_dir,
        initialize_model_fn, different_perm_init
    )
    clean_loader(ml_test)

    test_loaders = [None] * len(test_loader_factories)
    test_loaders[0] = test_loader_factories[0]()

    avg_accs, task_accs = [], []
    prediction_models = {}

    task_pbar = tqdm(range(1, len(train_loader_factories)+1), desc="Tasks")
    for task in task_pbar:
        t = task-1
        print(f"\nTask {task}")

        train_loader = first_train if t==0 else train_loader_factories[t]()
        if pre_task_hook:
            propagation_model = pre_task_hook(
                propagation_model, t, is_prediction_model=False
            )

        propagation_model = train_propagation_model(
            propagation_model, train_loader, t,
            epochs_per_task, lr, device,
            record_metric_fn, exp_dir, n_train_samples
        )

        coreset_x, coreset_y, sel_idx = active_select_coreset(
            propagation_model, train_loader,
            coreset_size, lambda_mix,
            use_kcenter, device, kcenter_batch_size
        )
        coreset_ds = update_coreset(
            coreset_ds, coreset_x, coreset_y, device, t
        )

        pred_epochs=int(epochs_per_task*pred_epochs_multiplier)  
        if use_task_specific_prediction:
            propagation_model, active_model, prediction_models = create_task_specific_models(
                model_class, propagation_model, coreset_ds, t,
                pred_epochs, lr, device, num_workers,
                pre_task_hook, None,
                exp_dir, n_train_samples
            )
        else:
            active_model = train_prediction_model(
                model_class, propagation_model, coreset_ds, t,
                pred_epochs, lr, device, num_workers,
                pre_task_hook, None,
                exp_dir, n_train_samples
            )

        save_checkpoint(active_model, ckpt_dir, task)

        if t == 0:
            first_train = None
        clean_loader(train_loader)
        clean_memory(device)

        # eval on all tasks seen so far
        for i in range(task):
            if test_loaders[i] is None:
                test_loaders[i] = test_loader_factories[i]()

        models_to_eval = prediction_models if use_task_specific_prediction else active_model
        accuracies, avg_acc = evaluate_models(
            models=models_to_eval,
            test_loaders=test_loaders[:task],
            task_idx=task,
            device=device,
            n_samples=n_eval_samples,
            record_metric_fn=record_metric_fn,
            evaluate_fn=evaluate_all_tasks_fn,
            use_task_specific=use_task_specific_prediction
        )
        task_accs.append(accuracies)
        avg_accs.append(avg_acc)
        task_pbar.set_postfix(avg_acc=f"{avg_acc:.4f}")

        if early_stopping_threshold is not None and avg_acc < early_stopping_threshold:
            print(f"\nEarly stopping: average accuracy {avg_acc:.4f} below threshold {early_stopping_threshold}")
            return None

        if not use_task_specific_prediction:
            prediction_models = active_model

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f}s ({total_time/60:.2f} min)")
    create_plots(exp_dir, avg_accs, task_accs)

    return prediction_models
