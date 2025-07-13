#!/usr/bin/env python3
"""
Example usage of the ARC Transformer model.
This script demonstrates how to create, train, and use the transformer model.
"""

import torch
import numpy as np
from transformer import ARCGridTransformer, Transformer
from arc_data_loader import visualize_grid, grid_to_string, string_to_grid


def create_sample_grids():
    """Create sample ARC-style grids for demonstration."""
    
    # Sample 1: Simple pattern - copy input
    input_grid1 = [
        [0, 1, 0],
        [1, 2, 1],
        [0, 1, 0]
    ]
    output_grid1 = [
        [0, 1, 0],
        [1, 2, 1],
        [0, 1, 0]
    ]
    
    # Sample 2: Pattern with rotation
    input_grid2 = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    output_grid2 = [
        [7, 4, 1],
        [8, 5, 2],
        [9, 6, 3]
    ]
    
    # Sample 3: Color change pattern
    input_grid3 = [
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]
    ]
    output_grid3 = [
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4]
    ]
    
    return [
        (input_grid1, output_grid1),
        (input_grid2, output_grid2),
        (input_grid3, output_grid3)
    ]


def demonstrate_basic_transformer():
    """Demonstrate basic transformer functionality."""
    print("=== Basic Transformer Demo ===")
    
    # Create a simple transformer
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=100,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=2
    )
    
    print(f"Basic transformer created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create sample sequences
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    src = torch.randint(1, 100, (batch_size, src_len))
    tgt = torch.randint(1, 100, (batch_size, tgt_len))
    
    # Forward pass
    with torch.no_grad():
        output = model(src, tgt)
        print(f"Input shape: {src.shape}")
        print(f"Target shape: {tgt.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output logits shape: {output.shape}")
    
    print("Basic transformer demo completed!\n")


def demonstrate_arc_transformer():
    """Demonstrate ARC-specific transformer functionality."""
    print("=== ARC Transformer Demo ===")
    
    # Create ARC transformer
    model = ARCGridTransformer(
        grid_size=5,
        num_colors=10,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=2
    )
    
    print(f"ARC transformer created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create sample grids
    sample_pairs = create_sample_grids()
    
    for i, (input_grid, output_grid) in enumerate(sample_pairs):
        print(f"\nSample {i+1}:")
        print("Input grid:")
        print(grid_to_string(torch.tensor(input_grid)))
        print("\nExpected output:")
        print(grid_to_string(torch.tensor(output_grid)))
        
        # Convert to tensors
        input_tensor = torch.tensor(input_grid).unsqueeze(0)  # Add batch dimension
        target_tensor = torch.tensor(output_grid).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor, target_tensor)
            print(f"Model output shape: {output.shape}")
            
            # Get predictions
            predictions = output.argmax(dim=-1)
            predicted_grid = model.sequence_to_grid(predictions[0], 3, 3)
            print("\nPredicted output:")
            print(grid_to_string(predicted_grid))
    
    print("\nARC transformer demo completed!\n")


def demonstrate_grid_conversions():
    """Demonstrate grid conversion utilities."""
    print("=== Grid Conversion Demo ===")
    
    # Create a sample grid
    grid = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    
    print("Original grid:")
    print(grid_to_string(grid))
    
    # Convert to string and back
    grid_str = grid_to_string(grid)
    reconstructed_grid = string_to_grid(grid_str)
    
    print("\nReconstructed grid:")
    print(grid_to_string(reconstructed_grid))
    
    # Test equality
    print(f"\nGrids are equal: {torch.equal(grid, reconstructed_grid)}")
    
    print("Grid conversion demo completed!\n")


def demonstrate_training_setup():
    """Demonstrate how to set up training."""
    print("=== Training Setup Demo ===")
    
    # Create model
    model = ARCGridTransformer(
        grid_size=5,
        num_colors=10,
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_layers=2
    )
    
    # Create sample data
    sample_pairs = create_sample_grids()
    
    # Prepare training data
    input_grids = []
    target_grids = []
    
    for input_grid, output_grid in sample_pairs:
        # Pad to 5x5
        padded_input = [[0] * 5 for _ in range(5)]
        padded_output = [[0] * 5 for _ in range(5)]
        
        for i in range(3):
            for j in range(3):
                padded_input[i][j] = input_grid[i][j]
                padded_output[i][j] = output_grid[i][j]
        
        input_grids.append(padded_input)
        target_grids.append(padded_output)
    
    # Convert to tensors
    input_tensor = torch.tensor(input_grids, dtype=torch.long)
    target_tensor = torch.tensor(target_grids, dtype=torch.long)
    
    print(f"Training data shape: {input_tensor.shape}")
    print(f"Target data shape: {target_tensor.shape}")
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    # Training loop (just a few steps for demo)
    model.train()
    for step in range(5):
        optimizer.zero_grad()
        
        # Forward pass
        batch_size = input_tensor.size(0)
        input_seqs = input_tensor.view(batch_size, -1)
        target_seqs = target_tensor.view(batch_size, -1)
        
        target_input = target_seqs[:, :-1]
        target_output = target_seqs[:, 1:]
        
        output = model.transformer(input_seqs, target_input)
        
        # Calculate loss
        loss = criterion(
            output.reshape(-1, output.size(-1)),
            target_output.reshape(-1)
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}, Loss: {loss.item():.4f}")
    
    print("Training setup demo completed!\n")


def demonstrate_visualization():
    """Demonstrate grid visualization."""
    print("=== Visualization Demo ===")
    
    # Create a colorful grid
    grid = torch.tensor([
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 1],
        [3, 4, 5, 1, 2],
        [4, 5, 1, 2, 3],
        [5, 1, 2, 3, 4]
    ])
    
    print("Grid to visualize:")
    print(grid_to_string(grid))
    
    try:
        # Try to visualize (requires matplotlib)
        visualize_grid(grid, title="Sample ARC Grid")
        print("Visualization displayed (if matplotlib is available)")
    except ImportError:
        print("Matplotlib not available - skipping visualization")
    
    print("Visualization demo completed!\n")


def main():
    """Run all demonstrations."""
    print("ARC Transformer Implementation - Example Usage")
    print("=" * 50)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    # Run demonstrations
    demonstrate_basic_transformer()
    demonstrate_arc_transformer()
    demonstrate_grid_conversions()
    demonstrate_training_setup()
    demonstrate_visualization()
    
    print("All demonstrations completed!")
    print("\nTo train the model on actual ARC data:")
    print("1. Ensure ARC dataset is in data/training/ and data/evaluation/")
    print("2. Run: python train_transformer.py --train_dir data/training --val_dir data/evaluation")
    print("\nFor more options, see: python train_transformer.py --help")


if __name__ == "__main__":
    main()