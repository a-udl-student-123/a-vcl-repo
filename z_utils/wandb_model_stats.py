import torch
import numpy as np

def compute_layer_stats(layer):
    stats = {}
    
    weight_mu_abs = torch.abs(layer.weight_mu)
    stats["weight_mu_abs"] = weight_mu_abs.sum().item()
    stats["weight_mu_abs_mean"] = weight_mu_abs.mean().item()
    stats["weight_mu_abs_std"] = weight_mu_abs.std().item()
    stats["weight_mu_count"] = layer.weight_mu.numel()
    
    bias_mu_abs = torch.abs(layer.bias_mu)
    stats["bias_mu_abs"] = bias_mu_abs.sum().item()
    stats["bias_mu_abs_mean"] = bias_mu_abs.mean().item()
    stats["bias_mu_abs_std"] = bias_mu_abs.std().item()
    stats["bias_mu_count"] = layer.bias_mu.numel()
    
    weight_std = layer._get_sigma(layer.weight_rho)
    stats["weight_std_sum"] = weight_std.sum().item()
    stats["weight_std_mean"] = weight_std.mean().item()
    stats["weight_std_std"] = weight_std.std().item()
    stats["weight_std_count"] = weight_std.numel()
    
    bias_std = layer._get_sigma(layer.bias_rho)
    stats["bias_std_sum"] = bias_std.sum().item()
    stats["bias_std_mean"] = bias_std.mean().item()
    stats["bias_std_std"] = bias_std.std().item()
    stats["bias_std_count"] = bias_std.numel()
    
    return stats

def compute_extended_vcl_model_stats(model):
    layer_stats = {}
    combined_stats = {
        "weight_mu_abs": 0.0, "weight_mu_count": 0, "weight_mu_values": [],
        "bias_mu_abs": 0.0, "bias_mu_count": 0, "bias_mu_values": [],
        "weight_std_values": [], "bias_std_values": [],
    }
    
    if hasattr(model, 'shared_layers'):
        for i, layer in enumerate(model.shared_layers):
            stats = compute_layer_stats(layer)
            layer_stats[i] = {
                "std": stats["weight_std_mean"],
                "mean_abs": stats["weight_mu_abs_mean"]
            }
            
            for key in ["weight_mu_abs", "weight_mu_count", "bias_mu_abs", "bias_mu_count"]:
                combined_stats[key] += stats[key]
            
            combined_stats["weight_mu_values"].append(torch.abs(layer.weight_mu).flatten())
            combined_stats["bias_mu_values"].append(torch.abs(layer.bias_mu).flatten())
            combined_stats["weight_std_values"].append(layer._get_sigma(layer.weight_rho).flatten())
            combined_stats["bias_std_values"].append(layer._get_sigma(layer.bias_rho).flatten())
    else:
        for i in range(1, 3):
            layer = getattr(model, f"lin{i}")
            stats = compute_layer_stats(layer)
            layer_stats[i-1] = {
                "std": stats["weight_std_mean"],
                "mean_abs": stats["weight_mu_abs_mean"]
            }
            
            for key in ["weight_mu_abs", "weight_mu_count", "bias_mu_abs", "bias_mu_count"]:
                combined_stats[key] += stats[key]
            
            combined_stats["weight_mu_values"].append(torch.abs(layer.weight_mu).flatten())
            combined_stats["bias_mu_values"].append(torch.abs(layer.bias_mu).flatten())
            combined_stats["weight_std_values"].append(layer._get_sigma(layer.weight_rho).flatten())
            combined_stats["bias_std_values"].append(layer._get_sigma(layer.bias_rho).flatten())
    
    is_multi_head = hasattr(model, 'heads')
    output_layer = model.heads[model.current_task] if is_multi_head else model.lin3
    
    output_layer_idx = len(layer_stats)
    
    stats = compute_layer_stats(output_layer)
    layer_stats[output_layer_idx] = {
        "std": stats["weight_std_mean"],
        "mean_abs": stats["weight_mu_abs_mean"]
    }
    
    for key in ["weight_mu_abs", "weight_mu_count", "bias_mu_abs", "bias_mu_count"]:
        combined_stats[key] += stats[key]
    
    combined_stats["weight_mu_values"].append(torch.abs(output_layer.weight_mu).flatten())
    combined_stats["bias_mu_values"].append(torch.abs(output_layer.bias_mu).flatten())
    combined_stats["weight_std_values"].append(layer._get_sigma(output_layer.weight_rho).flatten())
    combined_stats["bias_std_values"].append(layer._get_sigma(output_layer.bias_rho).flatten())
    
    all_weight_mu = torch.cat(combined_stats["weight_mu_values"])
    all_bias_mu = torch.cat(combined_stats["bias_mu_values"]) 
    all_weight_std = torch.cat(combined_stats["weight_std_values"])
    all_bias_std = torch.cat(combined_stats["bias_std_values"])
    all_params_mu = torch.cat([all_weight_mu, all_bias_mu])
    all_params_std = torch.cat([all_weight_std, all_bias_std])
    
    results = {
        "avg_weight_mu_abs": combined_stats["weight_mu_abs"] / combined_stats["weight_mu_count"] if combined_stats["weight_mu_count"] > 0 else 0,
        "avg_bias_mu_abs": combined_stats["bias_mu_abs"] / combined_stats["bias_mu_count"] if combined_stats["bias_mu_count"] > 0 else 0,
        "avg_weight_std": all_weight_std.mean().item(),
        "avg_bias_std": all_bias_std.mean().item(),
        "avg_param_std": all_params_std.mean().item(),
        "std_weight_mu": all_weight_mu.std().item(),
        "std_bias_mu": all_bias_mu.std().item(),
        "std_param_mu": all_params_mu.std().item(),
        "layer_stats": layer_stats
    }
    
    return results