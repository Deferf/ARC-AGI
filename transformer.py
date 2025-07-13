import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    Adds position information to input embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [seq_len, batch_size, d_model]
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism as described in "Attention is All You Need".
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
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional mask tensor
            
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        
        # Linear transformations and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and apply final linear transformation
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.w_o(attention_output)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """
    Feed-forward network with two linear transformations and ReLU activation.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with self-attention and feed-forward network.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional mask tensor
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with self-attention, cross-attention, and feed-forward network.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            enc_output: Encoder output [batch_size, seq_len, d_model]
            src_mask: Source mask for cross-attention
            tgt_mask: Target mask for self-attention
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with residual connection
        cross_attn_output, _ = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with multiple encoder layers.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, 
                 dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional mask tensor
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder with multiple decoder layers.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int,
                 dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            enc_output: Encoder output [batch_size, seq_len, d_model]
            src_mask: Source mask for cross-attention
            tgt_mask: Target mask for self-attention
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
            
        return x


class Transformer(nn.Module):
    """
    Complete transformer model with encoder and decoder.
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 num_heads: int = 8, d_ff: int = 2048, num_layers: int = 6,
                 dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout, max_len)
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, num_layers, dropout, max_len)
        
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize weights
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
                
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder self-attention."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the transformer.
        
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            src_mask: Source mask
            tgt_mask: Target mask (if None, will generate causal mask)
            
        Returns:
            Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
            if tgt.is_cuda:
                tgt_mask = tgt_mask.cuda()
        
        src_embedded = self.src_embedding(src)
        tgt_embedded = self.tgt_embedding(tgt)
        
        enc_output = self.encoder(src_embedded, src_mask)
        dec_output = self.decoder(tgt_embedded, enc_output, src_mask, tgt_mask)
        
        output = self.output_projection(dec_output)
        
        return output


class ARCGridTransformer(nn.Module):
    """
    Specialized transformer for ARC grid tasks.
    Converts 2D grids to sequences and back.
    """
    def __init__(self, grid_size: int = 30, num_colors: int = 10, d_model: int = 512,
                 num_heads: int = 8, d_ff: int = 2048, num_layers: int = 6,
                 dropout: float = 0.1):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.d_model = d_model
        
        # Grid to sequence conversion
        self.grid_embedding = nn.Embedding(num_colors + 1, d_model)  # +1 for padding
        
        # Transformer backbone
        self.transformer = Transformer(
            src_vocab_size=num_colors + 1,
            tgt_vocab_size=num_colors + 1,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout
        )
        
    def grid_to_sequence(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Convert 2D grid to 1D sequence.
        
        Args:
            grid: [batch_size, height, width] or [height, width]
        Returns:
            sequence: [batch_size, height * width] or [height * width]
        """
        if grid.dim() == 2:
            return grid.flatten()
        else:
            return grid.view(grid.size(0), -1)
    
    def sequence_to_grid(self, sequence: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Convert 1D sequence back to 2D grid.
        
        Args:
            sequence: [batch_size, height * width] or [height * width]
            height: Grid height
            width: Grid width
        Returns:
            grid: [batch_size, height, width] or [height, width]
        """
        if sequence.dim() == 1:
            return sequence.view(height, width)
        else:
            return sequence.view(sequence.size(0), height, width)
    
    def forward(self, input_grid: torch.Tensor, target_grid: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ARC grid transformation.
        
        Args:
            input_grid: Input grid [batch_size, height, width]
            target_grid: Target grid [batch_size, height, width]
        Returns:
            Output logits [batch_size, target_height * target_width, num_colors + 1]
        """
        # Convert grids to sequences
        input_seq = self.grid_to_sequence(input_grid)
        target_seq = self.grid_to_sequence(target_grid)
        
        # Pass through transformer
        output = self.transformer(input_seq, target_seq)
        
        return output
    
    def generate(self, input_grid: torch.Tensor, max_length: int = 900) -> torch.Tensor:
        """
        Generate output grid from input grid.
        
        Args:
            input_grid: Input grid [batch_size, height, width]
            max_length: Maximum sequence length
        Returns:
            Generated grid [batch_size, height, width]
        """
        self.eval()
        with torch.no_grad():
            input_seq = self.grid_to_sequence(input_grid)
            
            # Start with start token
            batch_size = input_seq.size(0)
            generated = torch.zeros(batch_size, 1, dtype=torch.long)
            if input_seq.is_cuda:
                generated = generated.cuda()
            
            for i in range(max_length):
                output = self.transformer(input_seq, generated)
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Stop if we generate end token or reach max length
                if (next_token == self.num_colors).any():
                    break
            
            # Convert back to grid (assuming square grid for simplicity)
            grid_size = int(math.sqrt(generated.size(1)))
            return self.sequence_to_grid(generated[:, 1:], grid_size, grid_size)


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create padding mask for transformer.
    
    Args:
        seq: Input sequence [batch_size, seq_len]
        pad_idx: Padding token index
    Returns:
        mask: [batch_size, 1, 1, seq_len]
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Create causal mask for decoder self-attention.
    
    Args:
        seq_len: Sequence length
    Returns:
        mask: [seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


# Example usage and training utilities
class TransformerTrainer:
    """
    Utility class for training transformer models.
    """
    def __init__(self, model: nn.Module, learning_rate: float = 0.0001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
        
    def train_step(self, src: torch.Tensor, tgt: torch.Tensor) -> float:
        """
        Single training step.
        
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
        Returns:
            loss: Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Remove last token from target for input, remove first token for output
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        output = self.model(src, tgt_input)
        loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, src: torch.Tensor, tgt: torch.Tensor) -> float:
        """
        Evaluate model on validation data.
        
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
        Returns:
            loss: Validation loss
        """
        self.model.eval()
        with torch.no_grad():
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output = self.model(src, tgt_input)
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            
        return loss.item()


if __name__ == "__main__":
    # Example usage
    print("Creating transformer model...")
    
    # Create a simple transformer for demonstration
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6
    )
    
    # Create ARC-specific transformer
    arc_model = ARCGridTransformer(
        grid_size=30,
        num_colors=10,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        num_layers=4
    )
    
    print(f"Transformer model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"ARC Transformer model created with {sum(p.numel() for p in arc_model.parameters()):,} parameters")
    
    # Example forward pass
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    src = torch.randint(1, 1000, (batch_size, src_len))
    tgt = torch.randint(1, 1000, (batch_size, tgt_len))
    
    output = model(src, tgt)
    print(f"Output shape: {output.shape}")
    
    # Example ARC grid forward pass
    grid_size = 5
    input_grid = torch.randint(0, 10, (batch_size, grid_size, grid_size))
    target_grid = torch.randint(0, 10, (batch_size, grid_size, grid_size))
    
    arc_output = arc_model(input_grid, target_grid)
    print(f"ARC output shape: {arc_output.shape}")