#!/usr/bin/env python3
"""
Test script to demonstrate Metal backend support and evaluation image generation.
"""

import torch
from evaluation_utils import get_device, save_input_output_pair, save_evaluation_summary
from arc_data_loader import ARCTask
from task_based_autoencoder import TaskBasedAutoencoder, evaluate_task_based_model
import numpy as np


def create_test_task():
    """Create a simple test task for demonstration."""
    # Simple pattern: input has a single colored cell, output fills a 3x3 square around it
    train_pairs = []
    test_pairs = []
    
    # Training examples
    for i in range(3):
        input_grid = [[0 for _ in range(5)] for _ in range(5)]
        output_grid = [[0 for _ in range(5)] for _ in range(5)]
        
        # Place a colored cell
        x, y = np.random.randint(1, 4), np.random.randint(1, 4)
        color = np.random.randint(1, 10)
        input_grid[x][y] = color
        
        # Fill 3x3 square in output
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if 0 <= x+dx < 5 and 0 <= y+dy < 5:
                    output_grid[x+dx][y+dy] = color
        
        train_pairs.append({
            'input': input_grid,
            'output': output_grid
        })
    
    # Test example
    input_grid = [[0 for _ in range(5)] for _ in range(5)]
    output_grid = [[0 for _ in range(5)] for _ in range(5)]
    x, y = 2, 2
    color = 5
    input_grid[x][y] = color
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if 0 <= x+dx < 5 and 0 <= y+dy < 5:
                output_grid[x+dx][y+dy] = color
    
    test_pairs.append({
        'input': input_grid,
        'output': output_grid
    })
    
    return ARCTask("test_task", train_pairs, test_pairs)


def main():
    print("=== Metal Backend Support and Image Generation Test ===\n")
    
    # Test device detection
    print("1. Testing device detection:")
    print(f"   Auto device: {get_device('auto')}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   MPS (Metal) available: {torch.backends.mps.is_available()}")
    print(f"   Selected device: {get_device('auto')}")
    
    # Get device
    device = get_device('auto')
    print(f"\n2. Using device: {device}")
    
    # Create a simple model
    print("\n3. Creating model...")
    model = TaskBasedAutoencoder(
        grid_size=10,
        num_colors=10,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=2,
        latent_dim=256
    ).to(device)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Model device: {next(model.parameters()).device}")
    
    # Create test tasks
    print("\n4. Creating test tasks...")
    test_tasks = [create_test_task() for _ in range(3)]
    
    # Run evaluation with image generation
    print("\n5. Running evaluation with image generation...")
    results = evaluate_task_based_model(
        model=model,
        test_tasks=test_tasks,
        device=device,
        save_images=True,
        output_dir='test_evaluation_images'
    )
    
    print("\n6. Evaluation results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n7. Generated images saved to: test_evaluation_images/")
    
    # Test individual image saving
    print("\n8. Testing individual image saving...")
    sample_input = torch.randint(0, 10, (5, 5))
    sample_output = torch.randint(0, 10, (5, 5))
    sample_pred = torch.randint(0, 10, (5, 5))
    
    save_input_output_pair(
        input_grid=sample_input,
        output_grid=sample_output,
        predicted_grid=sample_pred,
        task_id="sample",
        pair_idx=0,
        output_dir='test_evaluation_images'
    )
    
    print("   Sample image saved!")
    
    print("\n=== Test completed successfully! ===")


if __name__ == "__main__":
    main()