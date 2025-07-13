#!/usr/bin/env python3
"""
Example usage of the Transformer Autoencoder.
Encoder: 1+900+900 tokens -> 1024-dimensional vector
Decoder: 1024-dimensional vector -> 900+900 tokens
"""

import torch
import numpy as np
from transformer_autoencoder import TransformerAutoencoder, ARCAutoencoder
from arc_data_loader import visualize_grid, grid_to_string


def create_sample_sequences():
    """Create sample sequences for demonstration."""
    
    # Sample 1: Simple pattern
    input_seq1 = torch.randint(0, 100, (900,))  # 900 tokens
    output_seq1 = torch.randint(0, 100, (900,))  # 900 tokens
    
    # Sample 2: Sequential pattern
    input_seq2 = torch.arange(900) % 10  # 0-9 repeating
    output_seq2 = (torch.arange(900) + 1) % 10  # 1-0 repeating
    
    # Sample 3: Alternating pattern
    input_seq3 = torch.tensor([i % 2 for i in range(900)])  # Alternating 0,1
    output_seq3 = torch.tensor([(i + 1) % 2 for i in range(900)])  # Alternating 1,0
    
    return [
        (input_seq1, output_seq1),
        (input_seq2, output_seq2),
        (input_seq3, output_seq3)
    ]


def demonstrate_basic_autoencoder():
    """Demonstrate basic transformer autoencoder functionality."""
    print("=== Basic Transformer Autoencoder Demo ===")
    
    # Create autoencoder
    autoencoder = TransformerAutoencoder(
        vocab_size=100,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        num_layers=4,
        latent_dim=1024
    )
    
    print(f"Autoencoder created with {sum(p.numel() for p in autoencoder.parameters()):,} parameters")
    print(f"Input sequence length: {autoencoder.input_seq_len}")
    print(f"Output sequence length: {autoencoder.output_seq_len}")
    print(f"Latent dimension: {autoencoder.latent_dim}")
    
    # Create sample data
    sample_pairs = create_sample_sequences()
    
    for i, (input_seq, output_seq) in enumerate(sample_pairs):
        print(f"\nSample {i+1}:")
        
        # Create combined input sequence: [start_token] + input_seq + output_seq
        start_token = torch.tensor([1], dtype=torch.long)
        combined_input = torch.cat([start_token, input_seq, output_seq])
        
        print(f"Input sequence shape: {combined_input.shape}")
        print(f"Target sequence shape: {output_seq.shape}")
        
        # Forward pass
        with torch.no_grad():
            latent, output_logits = autoencoder(combined_input.unsqueeze(0))  # Add batch dimension
            
            print(f"Latent vector shape: {latent.shape}")
            print(f"Output logits shape: {output_logits.shape}")
            
            # Get predictions
            predictions = output_logits.argmax(dim=-1)
            print(f"Predictions shape: {predictions.shape}")
            
            # Calculate accuracy
            accuracy = (predictions[0] == output_seq).float().mean().item()
            print(f"Accuracy: {accuracy:.4f}")
    
    print("\nBasic autoencoder demo completed!\n")


def demonstrate_arc_autoencoder():
    """Demonstrate ARC-specific autoencoder functionality."""
    print("=== ARC Autoencoder Demo ===")
    
    # Create ARC autoencoder
    arc_autoencoder = ARCAutoencoder(
        grid_size=30,
        num_colors=10,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        num_layers=4,
        latent_dim=1024
    )
    
    print(f"ARC Autoencoder created with {sum(p.numel() for p in arc_autoencoder.parameters()):,} parameters")
    
    # Create sample grids
    batch_size = 2
    grid_size = 30
    
    # Sample 1: Simple pattern
    input_grid1 = torch.randint(0, 10, (batch_size, grid_size, grid_size))
    output_grid1 = torch.randint(0, 10, (batch_size, grid_size, grid_size))
    
    # Sample 2: Copy pattern
    input_grid2 = torch.randint(0, 10, (batch_size, grid_size, grid_size))
    output_grid2 = input_grid2.clone()  # Copy input to output
    
    # Sample 3: Shift pattern
    input_grid3 = torch.randint(0, 10, (batch_size, grid_size, grid_size))
    output_grid3 = torch.roll(input_grid3, shifts=1, dims=2)  # Shift right by 1
    
    samples = [
        (input_grid1, output_grid1, "Random pattern"),
        (input_grid2, output_grid2, "Copy pattern"),
        (input_grid3, output_grid3, "Shift pattern")
    ]
    
    for i, (input_grid, output_grid, description) in enumerate(samples):
        print(f"\nSample {i+1}: {description}")
        print(f"Input grid shape: {input_grid.shape}")
        print(f"Output grid shape: {output_grid.shape}")
        
        # Forward pass
        with torch.no_grad():
            latent, output_logits = arc_autoencoder(input_grid, output_grid)
            
            print(f"Latent vector shape: {latent.shape}")
            print(f"Output logits shape: {output_logits.shape}")
            
            # Get predictions
            predictions = output_logits.argmax(dim=-1)
            reconstructed_output = predictions.view(batch_size, grid_size, grid_size)
            
            print(f"Reconstructed output shape: {reconstructed_output.shape}")
            
            # Calculate accuracy
            accuracy = (reconstructed_output == output_grid).float().mean().item()
            print(f"Accuracy: {accuracy:.4f}")
            
            # Show sample grid
            if i == 0:  # Show first sample
                print("\nSample input grid (first batch):")
                print(grid_to_string(input_grid[0]))
                print("\nSample output grid (first batch):")
                print(grid_to_string(output_grid[0]))
                print("\nSample reconstructed grid (first batch):")
                print(grid_to_string(reconstructed_output[0]))
    
    print("\nARC autoencoder demo completed!\n")


def demonstrate_encoding_decoding():
    """Demonstrate separate encoding and decoding operations."""
    print("=== Encoding/Decoding Demo ===")
    
    # Create autoencoder
    autoencoder = TransformerAutoencoder(
        vocab_size=100,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        num_layers=4,
        latent_dim=1024
    )
    
    # Create sample input
    input_seq = torch.randint(0, 100, (900,))
    output_seq = torch.randint(0, 100, (900,))
    start_token = torch.tensor([1], dtype=torch.long)
    combined_input = torch.cat([start_token, input_seq, output_seq])
    
    print(f"Input sequence shape: {combined_input.shape}")
    
    # Separate encode and decode
    with torch.no_grad():
        # Encode
        latent = autoencoder.encode(combined_input.unsqueeze(0))
        print(f"Encoded latent shape: {latent.shape}")
        
        # Decode
        output_logits = autoencoder.decode(latent)
        print(f"Decoded output shape: {output_logits.shape}")
        
        # Generate from latent
        generated_tokens = autoencoder.generate(latent)
        print(f"Generated tokens shape: {generated_tokens.shape}")
        
        # Compare original and generated
        original_output = output_seq
        generated_output = generated_tokens[0]
        
        accuracy = (generated_output == original_output).float().mean().item()
        print(f"Generation accuracy: {accuracy:.4f}")
    
    print("Encoding/Decoding demo completed!\n")


def demonstrate_latent_space():
    """Demonstrate latent space operations."""
    print("=== Latent Space Demo ===")
    
    # Create autoencoder
    autoencoder = TransformerAutoencoder(
        vocab_size=100,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        num_layers=4,
        latent_dim=1024
    )
    
    # Create multiple samples
    num_samples = 3
    latents = []
    outputs = []
    
    for i in range(num_samples):
        input_seq = torch.randint(0, 100, (900,))
        output_seq = torch.randint(0, 100, (900,))
        start_token = torch.tensor([1], dtype=torch.long)
        combined_input = torch.cat([start_token, input_seq, output_seq])
        
        with torch.no_grad():
            latent = autoencoder.encode(combined_input.unsqueeze(0))
            latents.append(latent)
            outputs.append(output_seq)
    
    # Stack latents
    latents = torch.cat(latents, dim=0)
    print(f"Latent vectors shape: {latents.shape}")
    
    # Calculate distances between latents
    print("\nLatent space distances:")
    for i in range(num_samples):
        for j in range(i+1, num_samples):
            distance = torch.norm(latents[i] - latents[j]).item()
            print(f"Distance between sample {i+1} and {j+1}: {distance:.4f}")
    
    # Interpolate between latents
    print("\nLatent interpolation:")
    alpha = 0.5
    interpolated_latent = alpha * latents[0] + (1 - alpha) * latents[1]
    
    with torch.no_grad():
        interpolated_output = autoencoder.generate(interpolated_latent.unsqueeze(0))
        print(f"Interpolated output shape: {interpolated_output.shape}")
    
    print("Latent space demo completed!\n")


def demonstrate_training_setup():
    """Demonstrate how to set up training."""
    print("=== Training Setup Demo ===")
    
    # Create model
    autoencoder = TransformerAutoencoder(
        vocab_size=100,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=2,
        latent_dim=1024
    )
    
    # Create sample training data
    batch_size = 4
    input_seqs = []
    target_seqs = []
    
    for _ in range(batch_size):
        input_seq = torch.randint(0, 100, (900,))
        output_seq = torch.randint(0, 100, (900,))
        start_token = torch.tensor([1], dtype=torch.long)
        combined_input = torch.cat([start_token, input_seq, output_seq])
        
        input_seqs.append(combined_input)
        target_seqs.append(output_seq)
    
    # Stack into batches
    input_batch = torch.stack(input_seqs)
    target_batch = torch.stack(target_seqs)
    
    print(f"Input batch shape: {input_batch.shape}")
    print(f"Target batch shape: {target_batch.shape}")
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    # Training loop (just a few steps for demo)
    autoencoder.train()
    for step in range(3):
        optimizer.zero_grad()
        
        # Forward pass
        latent, output_logits = autoencoder(input_batch)
        
        # Calculate loss
        loss = criterion(
            output_logits.reshape(-1, output_logits.size(-1)),
            target_batch.reshape(-1)
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}, Loss: {loss.item():.4f}")
    
    print("Training setup demo completed!\n")


def main():
    """Run all demonstrations."""
    print("Transformer Autoencoder Implementation - Example Usage")
    print("=" * 60)
    print("Specifications:")
    print("- Encoder: 1+900+900 tokens -> 1024-dimensional vector")
    print("- Decoder: 1024-dimensional vector -> 900+900 tokens")
    print("=" * 60)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    # Run demonstrations
    demonstrate_basic_autoencoder()
    demonstrate_arc_autoencoder()
    demonstrate_encoding_decoding()
    demonstrate_latent_space()
    demonstrate_training_setup()
    
    print("All demonstrations completed!")
    print("\nTo train the autoencoder on actual data:")
    print("1. Ensure ARC dataset is in data/training/ and data/evaluation/")
    print("2. Run: python train_autoencoder.py --train_dir data/training --val_dir data/evaluation")
    print("3. For ARC mode: python train_autoencoder.py --arc_mode")
    print("\nFor more options, see: python train_autoencoder.py --help")


if __name__ == "__main__":
    main()