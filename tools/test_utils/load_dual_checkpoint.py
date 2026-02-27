import os
import torch
from mmcv.runner import load_checkpoint

def load_dual_checkpoints(model, base_checkpoint_path, module_checkpoint_path=None, module_prefixes=[]):
    """
    Load two checkpoints: base model + additional module
    
    Args:
        model: The model to load parameters into
        base_checkpoint_path: Path to the base model checkpoint
        module_checkpoint_path: Path to the additional module checkpoint (optional)
        module_prefixes: List of prefixes for module parameter names
    
    Returns:
        checkpoint: The base model checkpoint (containing meta information)
    """
    # Load base model checkpoint
    print(f"Loading base checkpoint from {base_checkpoint_path}")
    base_checkpoint = load_checkpoint(model, base_checkpoint_path, map_location='cpu', strict=False)
    
    # Load additional module checkpoint if provided
    if module_checkpoint_path and os.path.exists(module_checkpoint_path):
        print(f"Loading module checkpoint from {module_checkpoint_path}")
        module_checkpoint = torch.load(module_checkpoint_path, map_location='cpu')
        
        module_state_dict = module_checkpoint['state_dict']
        # for k, v in module_state_dict.items():
        #     print(f"Module parameter: {k}, shape: {v.shape}")
        
        # Filter parameters by prefixes if specified
        if module_prefixes:
            print(f"Filtering parameters with prefixes: {module_prefixes}")
            filtered_state_dict = {}
            for prefix in module_prefixes:
                prefix_params = {k: v for k, v in module_state_dict.items() 
                               if prefix in k}
                filtered_state_dict.update(prefix_params)
                print(f"Found {len(prefix_params)} parameters with prefix '{prefix}'")
            print(f"Total {len(filtered_state_dict)} parameters with all specified prefixes")
        else:
            filtered_state_dict = module_state_dict
            print(f"Loading {len(filtered_state_dict)} parameters from module checkpoint")
        
        # Get current model state dict
        current_state_dict = model.state_dict()
        
        # Only load parameters that exist in the model
        valid_params = {}
        for k, v in filtered_state_dict.items():
            if k in current_state_dict:
                if current_state_dict[k].shape == v.shape:
                    valid_params[k] = v
                    # print(f"Loading parameter: {k}")
                else:
                    print(f"Shape mismatch for {k}: model {current_state_dict[k].shape} vs checkpoint {v.shape}")
            else:
                print(f"Parameter {k} not found in model")
        
        # Load valid parameters
        if valid_params:
            model.load_state_dict(valid_params, strict=False)
            print(f"Successfully loaded {len(valid_params)} parameters from module checkpoint")
        else:
            print("No valid parameters found in module checkpoint")
    
    return base_checkpoint

