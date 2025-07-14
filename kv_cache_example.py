#!/usr/bin/env python3
"""
KV Cache Example - Educational Implementation

This script demonstrates the key concepts of KV cache optimization
for autoregressive inference without requiring PyTorch.
"""

import time
import math
from typing import List, Dict, Tuple, Optional
import random

# Mock tensor class for demonstration
class MockTensor:
    """Mock tensor class to demonstrate KV cache concepts."""
    
    def __init__(self, shape: Tuple[int, ...], data: List = None):
        self.shape = shape
        self.data = data or [random.random() for _ in range(math.prod(shape))]
    
    def __getitem__(self, key):
        return MockTensor((1,), [self.data[0]])
    
    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim] if dim < len(self.shape) else 1
    
    def __repr__(self):
        return f"MockTensor{self.shape}"


class MockKVCache:
    """
    Mock KV cache implementation to demonstrate the concept.
    
    In a real implementation, this would store actual PyTorch tensors
    containing the key and value matrices from attention layers.
    """
    
    def __init__(self, num_layers: int, num_heads: int, max_seq_len: int):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.current_seq_len = 0
        
        # Initialize cache storage
        self.cache = {}
        for layer_idx in range(num_layers):
            # Self-attention cache
            self.cache[f'layer_{layer_idx}_self_k'] = []
            self.cache[f'layer_{layer_idx}_self_v'] = []
            # Cross-attention cache
            self.cache[f'layer_{layer_idx}_cross_k'] = []
            self.cache[f'layer_{layer_idx}_cross_v'] = []
    
    def update_cache(self, layer_idx: int, attention_type: str, k: MockTensor, v: MockTensor):
        """Update cache with new key and value tensors."""
        cache_key_k = f'layer_{layer_idx}_{attention_type}_k'
        cache_key_v = f'layer_{layer_idx}_{attention_type}_v'
        
        self.cache[cache_key_k].append(k)
        self.cache[cache_key_v].append(v)
        self.current_seq_len += 1
    
    def get_cached_kv(self, layer_idx: int, attention_type: str) -> Tuple[List, List]:
        """Get cached key and value tensors."""
        cache_key_k = f'layer_{layer_idx}_{attention_type}_k'
        cache_key_v = f'layer_{layer_idx}_{attention_type}_v'
        
        return self.cache[cache_key_k], self.cache[cache_key_v]
    
    def clear_cache(self):
        """Clear the entire cache."""
        for key in self.cache:
            self.cache[key].clear()
        self.current_seq_len = 0


class MockTransformer:
    """
    Mock transformer implementation to demonstrate KV cache usage.
    
    This shows the conceptual flow of how KV cache optimizes generation.
    """
    
    def __init__(self, num_layers: int = 6, num_heads: int = 8):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.kv_cache = None
    
    def compute_attention(self, query: MockTensor, key: MockTensor, value: MockTensor, 
                         use_cache: bool = False, layer_idx: int = 0, 
                         attention_type: str = 'self') -> MockTensor:
        """
        Mock attention computation with optional KV caching.
        
        In a real implementation, this would:
        1. Compute Q, K, V from input
        2. Apply attention mechanism
        3. Cache K, V if requested
        """
        # Simulate computation time
        time.sleep(0.01)
        
        # Mock key and value tensors
        k_tensor = MockTensor((1, self.num_heads, 1, 64))  # [batch, heads, seq_len, d_k]
        v_tensor = MockTensor((1, self.num_heads, 1, 64))
        
        if use_cache and self.kv_cache is not None:
            # Get cached K, V tensors
            cached_k, cached_v = self.kv_cache.get_cached_kv(layer_idx, attention_type)
            
            # In real implementation: concatenate cached K,V with new K,V
            print(f"  Using cached K,V from {len(cached_k)} previous tokens")
            
            # Update cache with new K, V
            self.kv_cache.update_cache(layer_idx, attention_type, k_tensor, v_tensor)
        else:
            print(f"  Computing K,V from scratch for all tokens")
        
        # Mock attention output
        return MockTensor((1, 1, 512))  # [batch, seq_len, d_model]
    
    def forward_pass(self, input_tokens: List[int], use_cache: bool = False) -> MockTensor:
        """
        Forward pass through the transformer.
        
        Args:
            input_tokens: List of input token IDs
            use_cache: Whether to use KV cache
        """
        print(f"Forward pass with {len(input_tokens)} tokens, cache: {use_cache}")
        
        # Mock input embedding
        input_tensor = MockTensor((1, len(input_tokens), 512))
        
        # Process through each layer
        for layer_idx in range(self.num_layers):
            print(f"  Layer {layer_idx}:")
            
            # Self-attention
            self_attn_output = self.compute_attention(
                input_tensor, input_tensor, input_tensor,
                use_cache, layer_idx, 'self'
            )
            
            # Cross-attention (for decoder layers)
            cross_attn_output = self.compute_attention(
                self_attn_output, input_tensor, input_tensor,
                use_cache, layer_idx, 'cross'
            )
        
        return MockTensor((1, len(input_tokens), 1000))  # Output logits


def demonstrate_generation_without_cache():
    """Demonstrate naive generation without KV cache."""
    print("=" * 60)
    print("NAIVE GENERATION (Without KV Cache)")
    print("=" * 60)
    
    model = MockTransformer()
    input_tokens = [1, 2, 3, 4, 5]  # Initial context
    
    print("Generating sequence step by step:")
    total_time = 0
    
    for step in range(5):  # Generate 5 more tokens
        current_tokens = input_tokens + [random.randint(1, 100)] * step
        print(f"\nStep {step + 1}: Generating token {len(current_tokens) + 1}")
        print(f"Current sequence: {current_tokens}")
        
        start_time = time.time()
        output = model.forward_pass(current_tokens, use_cache=False)
        step_time = time.time() - start_time
        total_time += step_time
        
        print(f"Step {step + 1} completed in {step_time:.3f}s")
    
    print(f"\nTotal generation time: {total_time:.3f}s")
    print("Note: Each step recomputes attention for ALL previous tokens!")


def demonstrate_generation_with_cache():
    """Demonstrate optimized generation with KV cache."""
    print("\n" + "=" * 60)
    print("OPTIMIZED GENERATION (With KV Cache)")
    print("=" * 60)
    
    model = MockTransformer()
    model.kv_cache = MockKVCache(num_layers=6, num_heads=8, max_seq_len=100)
    input_tokens = [1, 2, 3, 4, 5]  # Initial context
    
    print("Generating sequence step by step:")
    total_time = 0
    
    # Initial forward pass (computes and caches K,V for all input tokens)
    print(f"\nInitial pass: Processing {len(input_tokens)} input tokens")
    start_time = time.time()
    output = model.forward_pass(input_tokens, use_cache=True)
    init_time = time.time() - start_time
    total_time += init_time
    print(f"Initial pass completed in {init_time:.3f}s")
    
    # Generate new tokens one by one
    for step in range(5):  # Generate 5 more tokens
        new_token = [random.randint(1, 100)]  # Single new token
        print(f"\nStep {step + 1}: Generating token {len(input_tokens) + step + 2}")
        print(f"New token: {new_token}")
        
        start_time = time.time()
        output = model.forward_pass(new_token, use_cache=True)
        step_time = time.time() - start_time
        total_time += step_time
        
        print(f"Step {step + 1} completed in {step_time:.3f}s")
    
    print(f"\nTotal generation time: {total_time:.3f}s")
    print("Note: Only new tokens are processed in each step!")


def demonstrate_performance_comparison():
    """Demonstrate the performance difference between approaches."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    sequence_lengths = [10, 20, 50, 100]
    
    print("Sequence Length | Without Cache | With Cache | Speedup")
    print("-" * 55)
    
    for seq_len in sequence_lengths:
        # Simulate time without cache (quadratic scaling)
        time_without_cache = seq_len * seq_len * 0.01
        
        # Simulate time with cache (linear scaling after initial pass)
        time_with_cache = seq_len * 0.01 + (seq_len - 1) * 0.001
        
        speedup = time_without_cache / time_with_cache
        
        print(f"{seq_len:14d} | {time_without_cache:12.3f}s | {time_with_cache:10.3f}s | {speedup:7.1f}x")
    
    print("\nKey observations:")
    print("- Without cache: O(n²) complexity per generation step")
    print("- With cache: O(n) complexity per generation step")
    print("- Speedup increases with sequence length")


def demonstrate_memory_usage():
    """Demonstrate memory usage patterns."""
    print("\n" + "=" * 60)
    print("MEMORY USAGE ANALYSIS")
    print("=" * 60)
    
    # Typical model configuration
    num_layers = 6
    num_heads = 8
    d_k = 64
    d_model = 512
    
    print("KV Cache Memory Requirements:")
    print(f"- Number of layers: {num_layers}")
    print(f"- Number of heads: {num_heads}")
    print(f"- Head dimension: {d_k}")
    print(f"- Model dimension: {d_model}")
    
    # Calculate memory per token
    bytes_per_token = num_layers * 2 * num_heads * d_k * 4  # 4 bytes per float32
    kb_per_token = bytes_per_token / 1024
    
    print(f"\nMemory per token: {kb_per_token:.2f} KB")
    
    # Memory for different sequence lengths
    sequence_lengths = [100, 500, 1000, 2000]
    print("\nTotal cache memory:")
    for seq_len in sequence_lengths:
        total_memory = kb_per_token * seq_len
        print(f"- {seq_len} tokens: {total_memory:.1f} KB ({total_memory/1024:.2f} MB)")
    
    print("\nMemory optimization tips:")
    print("- Use half precision (float16) to reduce memory by 50%")
    print("- Cache only essential layers for very large models")
    print("- Implement cache eviction for very long sequences")


def main():
    """Main demonstration function."""
    print("KV Cache for Autoregressive Inference - Educational Demo")
    print("=" * 70)
    print("This demo shows how KV cache optimizes transformer generation")
    print("by storing and reusing previously computed key-value tensors.")
    print()
    
    # Run demonstrations
    demonstrate_generation_without_cache()
    demonstrate_generation_with_cache()
    demonstrate_performance_comparison()
    demonstrate_memory_usage()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("KV Cache provides significant benefits for autoregressive generation:")
    print()
    print("✓ Reduces computational complexity from O(n²) to O(n)")
    print("✓ Provides speedups of 2-50x depending on sequence length")
    print("✓ Works with all sampling strategies (greedy, temperature, top-k, top-p)")
    print("✓ Maintains full backward compatibility with existing models")
    print("✓ Requires minimal code changes to implement")
    print()
    print("The implementation includes:")
    print("- Core KV cache functionality (kv_cache.py)")
    print("- Enhanced transformer with backward compatibility")
    print("- Comprehensive benchmarking tools")
    print("- Detailed documentation and examples")


if __name__ == "__main__":
    main()