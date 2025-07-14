#!/usr/bin/env python3
"""
Test script to verify Metal (MPS) support for PyTorch.
"""

import torch
import time
import numpy as np
from device_utils import get_available_device, get_device_info, setup_device, get_memory_stats


def test_device_detection():
    """Test device detection functionality."""
    print("=== Device Detection Test ===")
    
    # Test auto detection
    auto_device = get_available_device('auto')
    print(f"Auto-detected device: {auto_device}")
    
    # Test each device preference
    for pref in ['mps', 'cuda', 'cpu']:
        device = get_available_device(pref)
        print(f"Device with preference '{pref}': {device}")
    
    # Get device info
    info = get_device_info(auto_device)
    print(f"\nDevice info for {auto_device}:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print()


def test_tensor_operations(device_str: str):
    """Test basic tensor operations on the specified device."""
    print(f"=== Tensor Operations Test on {device_str} ===")
    
    try:
        device = torch.device(device_str)
        
        # Create tensors
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Test operations
        print("Testing tensor operations...")
        
        # Matrix multiplication
        start = time.time()
        z = torch.matmul(x, y)
        matmul_time = time.time() - start
        print(f"  Matrix multiplication (1000x1000): {matmul_time:.4f}s")
        
        # Element-wise operations
        start = time.time()
        result = torch.sin(x) + torch.cos(y)
        elem_time = time.time() - start
        print(f"  Element-wise operations: {elem_time:.4f}s")
        
        # Reduction operations
        start = time.time()
        mean_val = x.mean()
        std_val = x.std()
        reduction_time = time.time() - start
        print(f"  Reduction operations: {reduction_time:.4f}s")
        
        # Neural network operations
        linear = torch.nn.Linear(1000, 500).to(device)
        start = time.time()
        output = linear(x)
        nn_time = time.time() - start
        print(f"  Neural network forward pass: {nn_time:.4f}s")
        
        print("  ✓ All operations completed successfully")
        
    except Exception as e:
        print(f"  ✗ Error during tensor operations: {e}")
    
    print()


def test_model_training(device_str: str):
    """Test a simple model training loop."""
    print(f"=== Model Training Test on {device_str} ===")
    
    try:
        device = torch.device(device_str)
        
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss()
        
        # Create dummy data
        batch_size = 32
        x = torch.randn(batch_size, 100, device=device)
        y = torch.randn(batch_size, 10, device=device)
        
        # Training steps
        print("Running 10 training steps...")
        start = time.time()
        
        for step in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            if step == 0 or step == 9:
                print(f"  Step {step}: Loss = {loss.item():.4f}")
        
        training_time = time.time() - start
        print(f"  Training time: {training_time:.4f}s")
        print("  ✓ Training completed successfully")
        
    except Exception as e:
        print(f"  ✗ Error during training: {e}")
    
    print()


def test_memory_management(device_str: str):
    """Test memory management functionality."""
    print(f"=== Memory Management Test on {device_str} ===")
    
    if device_str == 'cpu':
        print("  Memory statistics not available for CPU")
        print()
        return
    
    try:
        # Get initial memory stats
        stats_before = get_memory_stats(device_str)
        print("Memory stats before allocation:")
        for key, value in stats_before.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.2f} MB")
            else:
                print(f"  {key}: {value}")
        
        # Allocate some tensors
        device = torch.device(device_str)
        tensors = []
        for i in range(5):
            tensors.append(torch.randn(1000, 1000, device=device))
        
        # Get memory stats after allocation
        stats_after = get_memory_stats(device_str)
        print("\nMemory stats after allocation:")
        for key, value in stats_after.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.2f} MB")
            else:
                print(f"  {key}: {value}")
        
        # Clean up
        del tensors
        if device_str == 'cuda':
            torch.cuda.empty_cache()
        
        print("  ✓ Memory management test completed")
        
    except Exception as e:
        print(f"  ✗ Error during memory management test: {e}")
    
    print()


def benchmark_devices():
    """Benchmark available devices."""
    print("=== Device Benchmark ===")
    
    # Get available devices
    devices_to_test = ['cpu']
    
    if torch.cuda.is_available():
        devices_to_test.append('cuda')
    
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        devices_to_test.append('mps')
    
    print(f"Testing devices: {devices_to_test}")
    print()
    
    # Benchmark matrix multiplication
    sizes = [(100, 100), (500, 500), (1000, 1000)]
    results = {}
    
    for device_str in devices_to_test:
        results[device_str] = []
        device = torch.device(device_str)
        
        for size in sizes:
            x = torch.randn(size, device=device)
            y = torch.randn(size, device=device)
            
            # Warm up
            _ = torch.matmul(x, y)
            
            # Time it
            start = time.time()
            for _ in range(10):
                _ = torch.matmul(x, y)
            elapsed = (time.time() - start) / 10
            
            results[device_str].append(elapsed)
    
    # Print results
    print("Matrix multiplication benchmark (average of 10 runs):")
    print(f"{'Size':<15} {' '.join(f'{dev:<10}' for dev in devices_to_test)}")
    print("-" * (15 + 11 * len(devices_to_test)))
    
    for i, size in enumerate(sizes):
        size_str = f"{size[0]}x{size[1]}"
        times = [f"{results[dev][i]:.4f}s" for dev in devices_to_test]
        print(f"{size_str:<15} {' '.join(f'{t:<10}' for t in times)}")
    
    print()


def main():
    """Run all tests."""
    print("PyTorch Metal (MPS) Support Test Suite")
    print("=" * 50)
    print()
    
    # Test device detection
    test_device_detection()
    
    # Setup device
    print("=== Device Setup ===")
    device = setup_device('auto', verbose=True)
    device_str = str(device)
    print()
    
    # Run tests on the selected device
    test_tensor_operations(device_str)
    test_model_training(device_str)
    test_memory_management(device_str)
    
    # Benchmark available devices
    benchmark_devices()
    
    print("All tests completed!")


if __name__ == "__main__":
    main()