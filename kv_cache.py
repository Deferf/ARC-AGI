import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Union
import math
import warnings


class KVCache:
    """
    Key-Value cache for optimizing autoregressive inference in transformer models.
    
    This cache stores the key and value tensors from previous forward passes
    to avoid recomputing them during autoregressive generation.
    """
    
    def __init__(self, 
                 num_layers: int,
                 num_heads: int,
                 d_k: int,
                 max_seq_len: int = 2048,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = None):
        """
        Initialize the KV cache.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            d_k: Dimension of each attention head
            max_seq_len: Maximum sequence length to cache
            dtype: Data type for cache tensors
            device: Device to store cache tensors
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device
        
        # Initialize cache tensors
        self.cache: Dict[str, torch.Tensor] = {}
        self.current_seq_len = 0
        self.is_initialized = False
        
    def initialize_cache(self, batch_size: int):
        """
        Initialize the cache with zeros for a given batch size.
        
        Args:
            batch_size: Batch size for the cache
        """
        if self.is_initialized:
            return
            
        for layer_idx in range(self.num_layers):
            # Self-attention cache
            self.cache[f'layer_{layer_idx}_self_k'] = torch.zeros(
                batch_size, self.num_heads, self.max_seq_len, self.d_k,
                dtype=self.dtype, device=self.device
            )
            self.cache[f'layer_{layer_idx}_self_v'] = torch.zeros(
                batch_size, self.num_heads, self.max_seq_len, self.d_k,
                dtype=self.dtype, device=self.device
            )
            
            # Cross-attention cache (for decoder layers)
            self.cache[f'layer_{layer_idx}_cross_k'] = torch.zeros(
                batch_size, self.num_heads, self.max_seq_len, self.d_k,
                dtype=self.dtype, device=self.device
            )
            self.cache[f'layer_{layer_idx}_cross_v'] = torch.zeros(
                batch_size, self.num_heads, self.max_seq_len, self.d_k,
                dtype=self.dtype, device=self.device
            )
        
        self.is_initialized = True
        self.current_seq_len = 0
        
    def update_cache(self, 
                    layer_idx: int, 
                    attention_type: str,
                    k: torch.Tensor, 
                    v: torch.Tensor,
                    seq_len: int):
        """
        Update the cache with new key and value tensors.
        
        Args:
            layer_idx: Layer index
            attention_type: Type of attention ('self' or 'cross')
            k: Key tensor [batch_size, num_heads, seq_len, d_k]
            v: Value tensor [batch_size, num_heads, seq_len, d_k]
            seq_len: Current sequence length
        """
        if not self.is_initialized:
            raise RuntimeError("Cache must be initialized before updating")
            
        cache_key_k = f'layer_{layer_idx}_{attention_type}_k'
        cache_key_v = f'layer_{layer_idx}_{attention_type}_v'
        
        # Update cache at the current position
        self.cache[cache_key_k][:, :, :seq_len, :] = k
        self.cache[cache_key_v][:, :, :seq_len, :] = v
        
    def get_cached_kv(self, 
                     layer_idx: int, 
                     attention_type: str,
                     seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached key and value tensors.
        
        Args:
            layer_idx: Layer index
            attention_type: Type of attention ('self' or 'cross')
            seq_len: Sequence length to retrieve
            
        Returns:
            Tuple of (cached_k, cached_v) tensors
        """
        if not self.is_initialized:
            raise RuntimeError("Cache must be initialized before retrieving")
            
        cache_key_k = f'layer_{layer_idx}_{attention_type}_k'
        cache_key_v = f'layer_{layer_idx}_{attention_type}_v'
        
        cached_k = self.cache[cache_key_k][:, :, :seq_len, :]
        cached_v = self.cache[cache_key_v][:, :, :seq_len, :]
        
        return cached_k, cached_v
    
    def clear_cache(self):
        """Clear the entire cache."""
        self.cache.clear()
        self.is_initialized = False
        self.current_seq_len = 0
        
    def set_seq_len(self, seq_len: int):
        """Set the current sequence length."""
        self.current_seq_len = seq_len
        
    def get_seq_len(self) -> int:
        """Get the current sequence length."""
        return self.current_seq_len


class CachedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with KV cache support for efficient autoregressive inference.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                   V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query tensor [batch_size, num_heads, seq_len, d_k]
            K: Key tensor [batch_size, num_heads, seq_len, d_k]
            V: Value tensor [batch_size, num_heads, seq_len, d_k]
            mask: Optional mask tensor
            
        Returns:
            attention_output: [batch_size, num_heads, seq_len, d_k]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None,
                layer_idx: int = 0,
                attention_type: str = 'self',
                use_cache: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional KV caching.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional mask tensor
            kv_cache: KV cache instance
            layer_idx: Layer index for cache
            attention_type: Type of attention ('self' or 'cross')
            use_cache: Whether to use cache
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear transformations and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        if use_cache and kv_cache is not None:
            # For autoregressive generation, we only have one new token
            if seq_len == 1:
                # Get cached K and V
                cached_k, cached_v = kv_cache.get_cached_kv(layer_idx, attention_type, kv_cache.get_seq_len())
                
                # Concatenate new K, V with cached ones
                K = torch.cat([cached_k, K], dim=2)
                V = torch.cat([cached_v, V], dim=2)
                
                # Update cache with new K, V
                kv_cache.update_cache(layer_idx, attention_type, K, V, kv_cache.get_seq_len() + 1)
                kv_cache.set_seq_len(kv_cache.get_seq_len() + 1)
                
                # Update sequence length for attention computation
                seq_len = K.size(2)
            else:
                # For initial forward pass, update cache with all K, V
                kv_cache.update_cache(layer_idx, attention_type, K, V, seq_len)
                kv_cache.set_seq_len(seq_len)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and apply final linear transformation
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.w_o(attention_output)
        
        return output, attention_weights


class CachedTransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with KV cache support.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = CachedMultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = CachedMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, 
                x: torch.Tensor, 
                enc_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None,
                layer_idx: int = 0,
                use_cache: bool = False) -> torch.Tensor:
        """
        Forward pass with KV caching.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            enc_output: Encoder output [batch_size, seq_len, d_model]
            src_mask: Source mask for cross-attention
            tgt_mask: Target mask for self-attention
            kv_cache: KV cache instance
            layer_idx: Layer index for cache
            use_cache: Whether to use cache
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(
            x, x, x, tgt_mask, kv_cache, layer_idx, 'self', use_cache
        )
        x = self.norm1(x + attn_output)
        
        # Cross-attention with residual connection
        cross_attn_output, _ = self.cross_attention(
            x, enc_output, enc_output, src_mask, kv_cache, layer_idx, 'cross', use_cache
        )
        x = self.norm2(x + cross_attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        return x


class CachedTransformerDecoder(nn.Module):
    """
    Transformer decoder with KV cache support.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int,
                 dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.pos_encoding = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            CachedTransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, 
                x: torch.Tensor, 
                enc_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None,
                use_cache: bool = False) -> torch.Tensor:
        """
        Forward pass with KV caching.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            enc_output: Encoder output [batch_size, seq_len, d_model]
            src_mask: Source mask for cross-attention
            tgt_mask: Target mask for self-attention
            kv_cache: KV cache instance
            use_cache: Whether to use cache
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = x + self.pos_encoding(positions)
        
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, enc_output, src_mask, tgt_mask, kv_cache, layer_idx, use_cache)
        
        return x


class OptimizedTransformer(nn.Module):
    """
    Optimized transformer with KV cache for efficient autoregressive inference.
    """
    
    def __init__(self, 
                 src_vocab_size: int, 
                 tgt_vocab_size: int, 
                 d_model: int = 512,
                 num_heads: int = 8, 
                 d_ff: int = 2048, 
                 num_layers: int = 6,
                 dropout: float = 0.1, 
                 max_len: int = 5000):
        super().__init__()
        
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Encoder and Decoder
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.decoder = CachedTransformerDecoder(
            d_model, num_heads, d_ff, num_layers, dropout, max_len
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # KV Cache
        self.kv_cache = None
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                
    def create_src_mask(self, src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """Create source padding mask."""
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def create_tgt_mask(self, tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """Create target padding mask."""
        tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        return tgt_pad_mask & tgt_sub_mask
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence."""
        src_embedded = self.src_embedding(src)
        
        for layer in self.encoder:
            src_embedded = layer(src_embedded, src_mask=src_mask)
            
        return src_embedded
    
    def decode(self, 
               tgt: torch.Tensor, 
               enc_output: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None,
               tgt_mask: Optional[torch.Tensor] = None,
               use_cache: bool = False) -> torch.Tensor:
        """Decode target sequence with optional KV caching."""
        tgt_embedded = self.tgt_embedding(tgt)
        
        output = self.decoder(
            tgt_embedded, enc_output, src_mask, tgt_mask, 
            self.kv_cache, use_cache
        )
        
        return self.output_projection(output)
    
    def forward(self, 
                src: torch.Tensor, 
                tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for training (no caching).
        
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            src_mask: Source mask
            tgt_mask: Target mask
            
        Returns:
            Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        if src_mask is None:
            src_mask = self.create_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.create_tgt_mask(tgt)
            
        enc_output = self.encode(src, src_mask)
        output = self.decode(tgt, enc_output, src_mask, tgt_mask, use_cache=False)
        
        return output
    
    def generate(self, 
                 src: torch.Tensor, 
                 max_length: int = 50,
                 start_token: int = 1,
                 end_token: int = 2,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None) -> torch.Tensor:
        """
        Generate sequence using KV cache for efficient autoregressive inference.
        
        Args:
            src: Source sequence [batch_size, src_len]
            max_length: Maximum generation length
            start_token: Start token ID
            end_token: End token ID
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated sequence [batch_size, gen_len]
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # Initialize KV cache
        if self.kv_cache is None:
            self.kv_cache = KVCache(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                d_k=self.d_k,
                max_seq_len=max_length,
                dtype=src.dtype,
                device=device
            )
            self.kv_cache.initialize_cache(batch_size)
        else:
            self.kv_cache.clear_cache()
            self.kv_cache.initialize_cache(batch_size)
        
        # Encode source sequence
        src_mask = self.create_src_mask(src)
        enc_output = self.encode(src, src_mask)
        
        # Initialize generation
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for step in range(max_length):
                # Create target mask for current sequence
                tgt_mask = self.create_tgt_mask(generated)
                
                # Decode with caching
                output = self.decode(
                    generated, enc_output, src_mask, tgt_mask, use_cache=True
                )
                
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
    
    def clear_cache(self):
        """Clear the KV cache."""
        if self.kv_cache is not None:
            self.kv_cache.clear_cache()


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal mask for autoregressive attention."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


# Utility functions for benchmarking
def benchmark_generation(model: OptimizedTransformer, 
                        src: torch.Tensor, 
                        max_length: int = 50,
                        num_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark generation performance with and without KV cache.
    
    Args:
        model: Transformer model
        src: Source sequence
        max_length: Maximum generation length
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    # Warm up
    for _ in range(3):
        _ = model.generate(src, max_length=max_length)
    
    # Benchmark with KV cache
    model.clear_cache()
    start_time = time.time()
    for _ in range(num_runs):
        _ = model.generate(src, max_length=max_length)
    cache_time = (time.time() - start_time) / num_runs
    
    return {
        'with_cache_avg_time': cache_time,
        'tokens_per_second': max_length / cache_time
    }


if __name__ == "__main__":
    # Example usage
    print("Creating optimized transformer with KV cache...")
    
    # Create model
    model = OptimizedTransformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Example generation
    batch_size = 2
    src_len = 10
    src = torch.randint(1, 1000, (batch_size, src_len))
    
    print("Generating sequence with KV cache...")
    generated = model.generate(src, max_length=20, temperature=0.8, top_k=50)
    print(f"Generated sequence shape: {generated.shape}")
    
    # Benchmark
    print("Benchmarking generation performance...")
    results = benchmark_generation(model, src, max_length=20, num_runs=5)
    print(f"Average generation time: {results['with_cache_avg_time']:.4f}s")
    print(f"Tokens per second: {results['tokens_per_second']:.2f}")