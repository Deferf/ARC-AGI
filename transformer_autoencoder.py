import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List
from transformer import MultiHeadAttention, PositionalEncoding, FeedForward


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
            enc_output: Encoder output [batch_size, enc_seq_len, d_model]
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


class AutoencoderEncoder(nn.Module):
    """
    Transformer encoder that takes 1+900+900 tokens and outputs a 1024-dimensional vector.
    """
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8, 
                 d_ff: int = 2048, num_layers: int = 6, latent_dim: int = 1024,
                 max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.max_len = max_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Global pooling and projection to latent space
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.latent_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, latent_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
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
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tokens [batch_size, seq_len] where seq_len = 1 + 900 + 900 = 1801
            mask: Optional mask tensor
        Returns:
            Latent vector [batch_size, latent_dim]
        """
        batch_size, seq_len = x.shape
        
        # Token embedding
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Global pooling: [batch_size, seq_len, d_model] -> [batch_size, d_model]
        x = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        x = self.global_pool(x).squeeze(-1)  # [batch_size, d_model]
        
        # Project to latent space
        latent = self.latent_projection(x)  # [batch_size, latent_dim]
        
        return latent


class AutoencoderDecoder(nn.Module):
    """
    Transformer decoder that takes a 1024-dimensional vector and outputs 900+900 tokens.
    """
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
                 d_ff: int = 2048, num_layers: int = 6, latent_dim: int = 1024,
                 max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.output_seq_len = 1800  # 900 + 900
        
        # Latent projection to decoder dimension
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model * self.output_seq_len)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder self-attention."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, latent: torch.Tensor, tgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            latent: Latent vector [batch_size, latent_dim]
            tgt: Target sequence for teacher forcing [batch_size, seq_len] (optional)
        Returns:
            Output logits [batch_size, output_seq_len, vocab_size]
        """
        batch_size = latent.size(0)
        
        # Project latent to decoder sequence
        decoder_input = self.latent_projection(latent)  # [batch_size, d_model * output_seq_len]
        decoder_input = decoder_input.view(batch_size, self.output_seq_len, self.d_model)
        
        # Add positional encoding
        decoder_input = self.pos_encoding(decoder_input.transpose(0, 1)).transpose(0, 1)
        decoder_input = self.dropout(decoder_input)
        
        # Create causal mask for self-attention
        tgt_mask = self.generate_square_subsequent_mask(self.output_seq_len)
        if latent.is_cuda:
            tgt_mask = tgt_mask.cuda()
        
        # Pass through transformer layers
        # For autoencoder, we use the same input for both encoder and decoder
        enc_output = decoder_input
        x = decoder_input
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask=None, tgt_mask=tgt_mask)
        
        # Output projection
        output = self.output_projection(x)  # [batch_size, output_seq_len, vocab_size]
        
        return output


class TransformerAutoencoder(nn.Module):
    """
    Complete transformer-based autoencoder.
    Encoder: 1+900+900 tokens -> 1024-dimensional vector
    Decoder: 1024-dimensional vector -> 900+900 tokens
    """
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
                 d_ff: int = 2048, num_layers: int = 6, latent_dim: int = 1024,
                 max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.input_seq_len = 1801  # 1 + 900 + 900
        self.output_seq_len = 1800  # 900 + 900
        
        # Encoder and decoder
        self.encoder = AutoencoderEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            latent_dim=latent_dim,
            max_len=max_len,
            dropout=dropout
        )
        
        self.decoder = AutoencoderDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            latent_dim=latent_dim,
            max_len=max_len,
            dropout=dropout
        )
        
        # Special tokens
        self.pad_token = 0
        self.start_token = 1
        self.end_token = 2
        
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode input sequence to latent vector.
        
        Args:
            x: Input tokens [batch_size, input_seq_len]
            mask: Optional mask tensor
        Returns:
            Latent vector [batch_size, latent_dim]
        """
        return self.encoder(x, mask)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to output sequence.
        
        Args:
            latent: Latent vector [batch_size, latent_dim]
        Returns:
            Output logits [batch_size, output_seq_len, vocab_size]
        """
        return self.decoder(latent)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.
        
        Args:
            x: Input tokens [batch_size, input_seq_len]
            mask: Optional mask tensor
        Returns:
            Tuple of (latent_vector, output_logits)
        """
        latent = self.encode(x, mask)
        output = self.decode(latent)
        return latent, output
    
    def generate(self, latent: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
        """
        Generate output sequence from latent vector.
        
        Args:
            latent: Latent vector [batch_size, latent_dim]
            max_length: Maximum generation length (defaults to output_seq_len)
        Returns:
            Generated tokens [batch_size, output_seq_len]
        """
        if max_length is None:
            max_length = self.output_seq_len
        
        self.decoder.eval()
        with torch.no_grad():
            output = self.decoder(latent)
            tokens = output.argmax(dim=-1)
        
        return tokens


class ARCAutoencoder(TransformerAutoencoder):
    """
    Specialized autoencoder for ARC grid tasks.
    Converts grids to sequences and back through the autoencoder.
    """
    def __init__(self, grid_size: int = 30, num_colors: int = 10, d_model: int = 512,
                 num_heads: int = 8, d_ff: int = 2048, num_layers: int = 6,
                 latent_dim: int = 1024, dropout: float = 0.1):
        
        # ARC vocabulary: colors (0-9) + special tokens
        vocab_size = num_colors + 3  # +3 for pad, start, end tokens
        
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            latent_dim=latent_dim,
            max_len=2000,
            dropout=dropout
        )
        
        self.grid_size = grid_size
        self.num_colors = num_colors
        
    def grid_to_sequence(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        """
        Convert input and output grids to a single sequence.
        
        Args:
            input_grid: Input grid [batch_size, height, width]
            output_grid: Output grid [batch_size, height, width]
        Returns:
            Combined sequence [batch_size, input_seq_len]
        """
        batch_size = input_grid.size(0)
        
        # Flatten grids
        input_seq = input_grid.view(batch_size, -1)  # [batch_size, height*width]
        output_seq = output_grid.view(batch_size, -1)  # [batch_size, height*width]
        
        # Combine: [start_token] + input_seq + output_seq
        start_tokens = torch.full((batch_size, 1), self.start_token, 
                                dtype=torch.long, device=input_grid.device)
        
        combined_seq = torch.cat([start_tokens, input_seq, output_seq], dim=1)
        
        return combined_seq
    
    def sequence_to_grids(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert sequence back to input and output grids.
        
        Args:
            sequence: Combined sequence [batch_size, input_seq_len]
        Returns:
            Tuple of (input_grid, output_grid)
        """
        batch_size = sequence.size(0)
        
        # Remove start token and split
        input_seq = sequence[:, 1:901]  # First 900 tokens
        output_seq = sequence[:, 901:]  # Last 900 tokens
        
        # Reshape to grids (assuming 30x30)
        input_grid = input_seq.view(batch_size, self.grid_size, self.grid_size)
        output_grid = output_seq.view(batch_size, self.grid_size, self.grid_size)
        
        return input_grid, output_grid
    
    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for ARC grids.
        
        Args:
            input_grid: Input grid [batch_size, height, width]
            output_grid: Output grid [batch_size, height, width]
        Returns:
            Tuple of (latent_vector, reconstructed_output_logits)
        """
        # Convert grids to sequence
        input_seq = self.grid_to_sequence(input_grid, output_grid)
        
        # Encode and decode
        latent, output_logits = super().forward(input_seq)
        
        return latent, output_logits
    
    def reconstruct(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct output grid from input and output grids.
        
        Args:
            input_grid: Input grid [batch_size, height, width]
            output_grid: Output grid [batch_size, height, width]
        Returns:
            Reconstructed output grid [batch_size, height, width]
        """
        latent, output_logits = self.forward(input_grid, output_grid)
        
        # Get predicted tokens
        predicted_tokens = output_logits.argmax(dim=-1)
        
        # Convert back to grid
        reconstructed_output = predicted_tokens.view(input_grid.size(0), self.grid_size, self.grid_size)
        
        return reconstructed_output


class AutoencoderTrainer:
    """
    Trainer for transformer autoencoder models.
    """
    def __init__(self, model: nn.Module, device: str = 'cuda',
                 learning_rate: float = 0.0001, weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        self.writer = None
        self.global_step = 0
        
    def train_step(self, input_seq: torch.Tensor, target_seq: torch.Tensor) -> float:
        """
        Single training step.
        
        Args:
            input_seq: Input sequence [batch_size, input_seq_len]
            target_seq: Target sequence [batch_size, output_seq_len]
        Returns:
            loss: Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        latent, output_logits = self.model(input_seq)
        
        # Calculate loss (only on output part)
        loss = self.criterion(
            output_logits.reshape(-1, output_logits.size(-1)),
            target_seq.reshape(-1)
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader) -> dict:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_seq, target_seq = batch
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                latent, output_logits = self.model(input_seq)
                
                loss = self.criterion(
                    output_logits.reshape(-1, output_logits.size(-1)),
                    target_seq.reshape(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Calculate accuracy
                predictions = output_logits.argmax(dim=-1)
                correct_predictions += (predictions == target_seq).sum().item()
                total_predictions += target_seq.numel()
        
        avg_loss = total_loss / num_batches
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }


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


if __name__ == "__main__":
    # Example usage
    print("Creating transformer autoencoder...")
    
    # Create autoencoder
    autoencoder = TransformerAutoencoder(
        vocab_size=1000,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        latent_dim=1024
    )
    
    print(f"Autoencoder created with {sum(p.numel() for p in autoencoder.parameters()):,} parameters")
    
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
    
    # Example forward pass
    batch_size = 2
    input_seq_len = 1801  # 1 + 900 + 900
    output_seq_len = 1800  # 900 + 900
    
    input_seq = torch.randint(0, 1000, (batch_size, input_seq_len))
    target_seq = torch.randint(0, 1000, (batch_size, output_seq_len))
    
    latent, output = autoencoder(input_seq)
    print(f"Input shape: {input_seq.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Output shape: {output.shape}")
    
    # Example ARC forward pass
    grid_size = 30
    input_grid = torch.randint(0, 10, (batch_size, grid_size, grid_size))
    output_grid = torch.randint(0, 10, (batch_size, grid_size, grid_size))
    
    arc_latent, arc_output = arc_autoencoder(input_grid, output_grid)
    print(f"ARC input grid shape: {input_grid.shape}")
    print(f"ARC latent shape: {arc_latent.shape}")
    print(f"ARC output shape: {arc_output.shape}")