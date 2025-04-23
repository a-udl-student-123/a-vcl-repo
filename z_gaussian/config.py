

GAUSSIAN_VCL_CONFIG = {
    "method": "coreset_vcl",
    
    # train
    "epochs": 10,
    "num_tasks": 4, 
    "hidden_size": 100,
    "lr": 1e-3,
    "num_workers": 1,
    "batch_size": 256,
    
    # coreset 
    "coreset_size": 0, 
    "use_kcenter": True,
    "kcenter_batch_size": 1024,
    
    # initialization params
    "use_ml_initialization": True,
    "ml_epochs": 20,
    "init_std": 0.001,
    "adaptive_std": False,  
    "adaptive_std_epsilon": 0.01,
    "different_perm_init": False,  # TODO: test if this helps
    
    "n_eval_samples": 100,
    "n_train_samples": 1,
    "pred_epochs_multiplier": 1.0,
    
    "early_stopping_threshold": None,
    
    "exp_dir": "gaussian_vcl",
}