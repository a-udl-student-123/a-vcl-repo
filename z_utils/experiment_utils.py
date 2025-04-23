
from pathlib import Path

def is_experiment_completed(exp_dir, num_tasks):
    # quick check if we already ran
    exp_dir = Path(exp_dir)
    metrics_path = exp_dir / "metrics.csv"
    return metrics_path.exists()

def format_param_for_dirname(param_name, param_value):
    # format param values for dir names - handles bools and floats specially
    if isinstance(param_value, bool):
        return f"{param_name}{'T' if param_value else 'F'}"
    elif isinstance(param_value, float):
        if param_value < 0.0001:
            return f"{param_name}{param_value:.0e}" 
        return f"{param_name}{param_value}"
    else:
        return f"{param_name}{param_value}"

def create_experiment_name(config):
    parts = [f"{config['method']}"]
    
    if 'lr' in config:
        parts.append(f"lr{config['lr']}")
    if 'init_std' in config:
        parts.append(f"std{config['init_std']}")
    
    if config.get('method') == 'coreset_vcl' and 'coreset_size' in config:
        parts.append(f"coreset{config['coreset_size']}")
        if config.get('use_kcenter', False):
            parts.append("kcenter")
    
    if 'seed' in config:
        parts.append(f"seed{config['seed']}")
    
    return "_".join(parts) 