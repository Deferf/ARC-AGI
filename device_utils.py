"""
Device utilities for PyTorch with support for CPU, CUDA, and Metal (MPS).
"""

import torch
import warnings
from typing import Union, Optional


def get_available_device(preference: str = 'auto') -> str:
    """
    Get the best available device based on preference and availability.
    
    Args:
        preference: Device preference ('auto', 'mps', 'cuda', 'cpu')
        
    Returns:
        Device string ('mps', 'cuda', or 'cpu')
    """
    if preference == 'auto':
        # Priority: MPS (Metal) > CUDA > CPU
        if torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                warnings.warn("MPS backend is available but not built. Falling back to next available device.")
            else:
                return 'mps'
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    elif preference == 'mps':
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return 'mps'
        else:
            warnings.warn(f"MPS (Metal) requested but not available. Falling back to auto selection.")
            return get_available_device('auto')
    
    elif preference == 'cuda':
        if torch.cuda.is_available():
            return 'cuda'
        else:
            warnings.warn(f"CUDA requested but not available. Falling back to auto selection.")
            return get_available_device('auto')
    
    else:  # cpu or any other value
        return 'cpu'


def get_device_info(device: Optional[str] = None) -> dict:
    """
    Get information about the specified device or current device.
    
    Args:
        device: Device string or None to auto-detect
        
    Returns:
        Dictionary with device information
    """
    if device is None:
        device = get_available_device()
    
    info = {
        'device': device,
        'available': False,
        'name': 'CPU',
        'properties': {}
    }
    
    if device == 'mps':
        info['available'] = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        if info['available']:
            info['name'] = 'Apple Metal Performance Shaders'
            # MPS doesn't provide detailed properties like CUDA
            info['properties'] = {
                'backend': 'Metal',
                'platform': 'Apple Silicon'
            }
    
    elif device == 'cuda':
        info['available'] = torch.cuda.is_available()
        if info['available']:
            info['name'] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info['properties'] = {
                'total_memory': props.total_memory,
                'major': props.major,
                'minor': props.minor,
                'multi_processor_count': props.multi_processor_count
            }
    
    else:  # cpu
        info['available'] = True
        info['name'] = 'CPU'
        info['properties'] = {
            'threads': torch.get_num_threads()
        }
    
    return info


def setup_device(device: Union[str, torch.device] = 'auto', verbose: bool = True) -> torch.device:
    """
    Setup and return a torch device with proper configuration.
    
    Args:
        device: Device preference or torch.device object
        verbose: Whether to print device information
        
    Returns:
        torch.device object
    """
    if isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = get_available_device(device)
    
    # Create torch device
    torch_device = torch.device(device_str)
    
    # Device-specific setup
    if device_str == 'mps':
        # MPS-specific configuration
        # Set memory growth if needed (MPS handles this automatically)
        pass
    
    elif device_str.startswith('cuda'):
        # CUDA-specific configuration
        if torch.cuda.is_available():
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    if verbose:
        info = get_device_info(device_str)
        print(f"Using device: {info['name']} ({info['device']})")
        if info['properties']:
            for key, value in info['properties'].items():
                if key == 'total_memory':
                    print(f"  - {key}: {value / 1024**3:.2f} GB")
                else:
                    print(f"  - {key}: {value}")
    
    return torch_device


def move_to_device(data, device: Union[str, torch.device]):
    """
    Recursively move data to the specified device.
    
    Args:
        data: Data to move (tensor, list, dict, etc.)
        device: Target device
        
    Returns:
        Data moved to device
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    else:
        return data


def get_memory_stats(device: Optional[str] = None) -> dict:
    """
    Get memory statistics for the specified device.
    
    Args:
        device: Device string or None to auto-detect
        
    Returns:
        Dictionary with memory statistics
    """
    if device is None:
        device = get_available_device()
    
    stats = {
        'device': device,
        'allocated': 0,
        'reserved': 0,
        'free': 0
    }
    
    if device == 'cuda' and torch.cuda.is_available():
        stats['allocated'] = torch.cuda.memory_allocated() / 1024**2  # MB
        stats['reserved'] = torch.cuda.memory_reserved() / 1024**2  # MB
        stats['free'] = (torch.cuda.get_device_properties(0).total_memory - 
                        torch.cuda.memory_allocated()) / 1024**2  # MB
    
    elif device == 'mps':
        # MPS doesn't provide detailed memory statistics yet
        # This is a limitation of the current PyTorch MPS backend
        stats['note'] = "Detailed memory statistics not available for MPS backend"
    
    return stats


# Convenience function for backward compatibility
def get_device(preference: str = 'auto', verbose: bool = True) -> str:
    """
    Get device string with backward compatibility.
    
    Args:
        preference: Device preference
        verbose: Whether to print device information
        
    Returns:
        Device string
    """
    device = get_available_device(preference)
    if verbose:
        setup_device(device, verbose=True)
    return device