#!/usr/bin/env python3
"""
Demonstration of KV Cache for optimizing autoregressive inference in transformers.

This script shows how to use the KV cache implementation to significantly speed up
autoregressive generation compared to naive approaches.
"""

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional

# Import our KV cache implementation
from kv_cache import OptimizedTransformer, KVCache, benchmark_generation

# Import the original transformer for comparison
from transformer import Transformer


class NaiveTransformer(Transformer):
    """
    Wrapper around the original transformer to provide a consistent interface
    for benchmarking against the optimized version.
    """
    
    def generate(self, src: torch.Tensor, max_length: int = 50, 
                start_token: int = 1, end_token: int = 2,
                temperature: float = 1.0, top_k: Optional[int] = None,
                top_p: Optional[float] = None) -> torch.Tensor:
        """
        Naive generation without KV cache (for comparison).
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # Initialize generation
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for step in range(max_length):
                # Create masks
                src_mask = self.generate_square_subsequent_mask(src.size(1)).to(device)
                tgt_mask = self.generate_square_subsequent_mask(generated.size(1)).to(device)
                
                # Forward pass (recomputes everything each time)
                output = self.forward(src, generated, src_mask, tgt_mask)
                
                # Get next token logits
                next_token_logits = output[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end token
                if (next_token == end_token).any():
                    break
        
        return generated[:, 1:]  # Remove start token


def benchmark_comparison(seq_lengths: List[int], 
                        batch_size: int = 2,
                        num_runs: int = 5) -> Dict[str, List[float]]:
    """
    Benchmark naive vs optimized generation across different sequence lengths.
    
    Args:
        seq_lengths: List of sequence lengths to test
        batch_size: Batch size for testing
        num_runs: Number of runs per test
        
    Returns:
        Dictionary with timing results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    naive_model = NaiveTransformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=256,  # Smaller model for faster testing
        num_heads=8,
        d_ff=1024,
        num_layers=4
    ).to(device)
    
    optimized_model = OptimizedTransformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        num_layers=4
    ).to(device)
    
    # Copy weights from naive to optimized for fair comparison
    with torch.no_grad():
        for naive_param, opt_param in zip(naive_model.parameters(), optimized_model.parameters()):
            opt_param.copy_(naive_param)
    
    naive_times = []
    optimized_times = []
    speedups = []
    
    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        
        # Create test input
        src = torch.randint(1, 1000, (batch_size, seq_len), device=device)
        
        # Benchmark naive generation
        print("Benchmarking naive generation...")
        naive_model.eval()
        start_time = time.time()
        for _ in range(num_runs):
            _ = naive_model.generate(src, max_length=seq_len)
        naive_time = (time.time() - start_time) / num_runs
        naive_times.append(naive_time)
        
        # Benchmark optimized generation
        print("Benchmarking optimized generation...")
        optimized_model.eval()
        optimized_model.clear_cache()
        start_time = time.time()
        for _ in range(num_runs):
            _ = optimized_model.generate(src, max_length=seq_len)
        optimized_time = (time.time() - start_time) / num_runs
        optimized_times.append(optimized_time)
        
        # Calculate speedup
        speedup = naive_time / optimized_time
        speedups.append(speedup)
        
        print(f"Naive: {naive_time:.4f}s, Optimized: {optimized_time:.4f}s, Speedup: {speedup:.2f}x")
    
    return {
        'seq_lengths': seq_lengths,
        'naive_times': naive_times,
        'optimized_times': optimized_times,
        'speedups': speedups
    }


def plot_benchmark_results(results: Dict[str, List[float]]):
    """
    Plot benchmark results showing the performance improvement.
    
    Args:
        results: Dictionary with benchmark results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot timing comparison
    ax1.plot(results['seq_lengths'], results['naive_times'], 'o-', label='Naive Generation', linewidth=2, markersize=8)
    ax1.plot(results['seq_lengths'], results['optimized_times'], 's-', label='KV Cache Generation', linewidth=2, markersize=8)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Generation Time (seconds)')
    ax1.set_title('Generation Time Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot speedup
    ax2.plot(results['seq_lengths'], results['speedups'], 'o-', color='green', linewidth=2, markersize=8)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Performance Improvement with KV Cache')
    ax2.grid(True, alpha=0.3)
    
    # Add speedup annotations
    for i, speedup in enumerate(results['speedups']):
        ax2.annotate(f'{speedup:.1f}x', 
                    (results['seq_lengths'][i], speedup),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.tight_layout()
    plt.savefig('kv_cache_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_memory_usage():
    """
    Demonstrate memory usage patterns with and without KV cache.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a small model for demonstration
    model = OptimizedTransformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=2
    ).to(device)
    
    batch_size = 1
    seq_len = 10
    
    # Create test input
    src = torch.randint(1, 1000, (batch_size, seq_len), device=device)
    
    print("Memory usage demonstration:")
    print("=" * 50)
    
    # Check memory before generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Initial GPU memory: {initial_memory:.2f} MB")
    
    # Generate without cache
    model.clear_cache()
    _ = model.generate(src, max_length=seq_len)
    
    if torch.cuda.is_available():
        memory_after_generation = torch.cuda.memory_allocated() / 1024**2
        print(f"Memory after generation: {memory_after_generation:.2f} MB")
        print(f"Memory increase: {memory_after_generation - initial_memory:.2f} MB")
    
    # Check cache memory usage
    if model.kv_cache is not None:
        cache_memory = sum(tensor.numel() * tensor.element_size() for tensor in model.kv_cache.cache.values()) / 1024**2
        print(f"KV cache memory usage: {cache_memory:.2f} MB")
    
    print("=" * 50)


def demonstrate_cache_behavior():
    """
    Demonstrate how the KV cache behaves during generation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a small model
    model = OptimizedTransformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=64,
        num_heads=2,
        d_ff=256,
        num_layers=1
    ).to(device)
    
    batch_size = 1
    src_len = 5
    
    # Create test input
    src = torch.randint(1, 1000, (batch_size, src_len), device=device)
    
    print("KV Cache Behavior Demonstration:")
    print("=" * 50)
    
    # Generate sequence
    generated = model.generate(src, max_length=10)
    
    print(f"Generated sequence: {generated[0].tolist()}")
    print(f"Sequence length: {generated.size(1)}")
    
    if model.kv_cache is not None:
        print(f"Final cache sequence length: {model.kv_cache.get_seq_len()}")
        print(f"Number of cached layers: {model.kv_cache.num_layers}")
        print(f"Cache keys: {list(model.kv_cache.cache.keys())}")
    
    print("=" * 50)


def main():
    """
    Main demonstration function.
    """
    print("KV Cache for Autoregressive Inference - Demonstration")
    print("=" * 60)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA not available, using CPU")
    
    print("\n1. Demonstrating cache behavior...")
    demonstrate_cache_behavior()
    
    print("\n2. Demonstrating memory usage...")
    demonstrate_memory_usage()
    
    print("\n3. Running performance benchmark...")
    # Test with different sequence lengths
    seq_lengths = [10, 20, 30, 40, 50]
    results = benchmark_comparison(seq_lengths, batch_size=1, num_runs=3)
    
    print("\n4. Plotting results...")
    try:
        plot_benchmark_results(results)
        print("Results saved to 'kv_cache_benchmark.png'")
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        print("Benchmark results:")
        for i, seq_len in enumerate(results['seq_lengths']):
            print(f"Length {seq_len}: {results['speedups'][i]:.2f}x speedup")
    
    print("\n5. Example usage with sampling strategies...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = OptimizedTransformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=2
    ).to(device)
    
    src = torch.randint(1, 1000, (1, 5), device=device)
    
    # Generate with different sampling strategies
    print("Generating with greedy sampling...")
    greedy_output = model.generate(src, max_length=10, temperature=1.0)
    
    print("Generating with temperature sampling...")
    temp_output = model.generate(src, max_length=10, temperature=0.8)
    
    print("Generating with top-k sampling...")
    topk_output = model.generate(src, max_length=10, temperature=1.0, top_k=10)
    
    print("Generating with top-p sampling...")
    topp_output = model.generate(src, max_length=10, temperature=1.0, top_p=0.9)
    
    print(f"Greedy output: {greedy_output[0].tolist()}")
    print(f"Temperature output: {temp_output[0].tolist()}")
    print(f"Top-k output: {topk_output[0].tolist()}")
    print(f"Top-p output: {topp_output[0].tolist()}")
    
    print("\nDemonstration completed!")


if __name__ == "__main__":
    main()