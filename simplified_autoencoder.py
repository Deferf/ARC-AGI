#!/usr/bin/env python3
"""
Simplified Transformer Autoencoder for ARC tasks.
Contains only essential components for training and inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple
import json
import os
from torch.utils.data import Dataset, DataLoader
import argparse


class PositionalEncoding(nn.Module):
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
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
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
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, V)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.w_o(attention_output)
        
        return output, attention_weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerDecoderLayer(nn.Module):
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
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        cross_attn_output, _ = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class AutoencoderEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8, 
                 d_ff: int = 2048, num_layers: int = 6, latent_dim: int = 1024,
                 max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.max_len = max_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.latent_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, latent_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = x.shape
        
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        latent = self.latent_projection(x)
        
        return latent


class AutoencoderDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
                 d_ff: int = 2048, num_layers: int = 6, latent_dim: int = 1024,
                 max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.output_seq_len = 1800  # 900 + 900
        
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model * self.output_seq_len)
        )
        
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, latent: torch.Tensor, tgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = latent.size(0)
        
        decoder_input = self.latent_projection(latent)
        decoder_input = decoder_input.view(batch_size, self.output_seq_len, self.d_model)
        
        decoder_input = self.pos_encoding(decoder_input.transpose(0, 1)).transpose(0, 1)
        decoder_input = self.dropout(decoder_input)
        
        tgt_mask = self.generate_square_subsequent_mask(self.output_seq_len)
        if latent.is_cuda:
            tgt_mask = tgt_mask.cuda()
        
        enc_output = decoder_input
        x = decoder_input
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask=None, tgt_mask=tgt_mask)
        
        output = self.output_projection(x)
        
        return output


class TransformerAutoencoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
                 d_ff: int = 2048, num_layers: int = 6, latent_dim: int = 1024,
                 max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = AutoencoderEncoder(
            vocab_size, d_model, num_heads, d_ff, num_layers, latent_dim, max_len, dropout
        )
        
        self.decoder = AutoencoderDecoder(
            vocab_size, d_model, num_heads, d_ff, num_layers, latent_dim, max_len, dropout
        )
        
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(x, mask)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x, mask)
        output = self.decode(latent)
        return latent, output


class ARCAutoencoder(TransformerAutoencoder):
    def __init__(self, grid_size: int = 30, num_colors: int = 10, d_model: int = 512,
                 num_heads: int = 8, d_ff: int = 2048, num_layers: int = 6,
                 latent_dim: int = 1024, dropout: float = 0.1):
        
        # ARC vocabulary: colors (0-9) + special tokens
        vocab_size = num_colors + 2  # colors + start token + padding token
        super().__init__(vocab_size, d_model, num_heads, d_ff, num_layers, latent_dim, 2000, dropout)
        
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.start_token = num_colors
        self.pad_token = num_colors + 1
    
    def grid_to_sequence(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        batch_size = input_grid.size(0)
        
        # Flatten grids
        input_seq = input_grid.view(batch_size, -1)  # [batch_size, grid_size^2]
        output_seq = output_grid.view(batch_size, -1)  # [batch_size, grid_size^2]
        
        # Create combined sequence: [start_token] + input_seq + output_seq
        start_tokens = torch.full((batch_size, 1), self.start_token, dtype=torch.long, device=input_grid.device)
        combined_seq = torch.cat([start_tokens, input_seq, output_seq], dim=1)  # [batch_size, 1 + 2*grid_size^2]
        
        return combined_seq
    
    def sequence_to_grids(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = sequence.size(0)
        seq_len = sequence.size(1)
        
        # Remove start token and split into input and output
        input_seq = sequence[:, 1:self.grid_size**2 + 1]  # [batch_size, grid_size^2]
        output_seq = sequence[:, self.grid_size**2 + 1:]  # [batch_size, grid_size^2]
        
        # Reshape to grids
        input_grid = input_seq.view(batch_size, self.grid_size, self.grid_size)
        output_grid = output_seq.view(batch_size, self.grid_size, self.grid_size)
        
        return input_grid, output_grid
    
    def forward(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert grids to sequence
        combined_seq = self.grid_to_sequence(input_grid, output_grid)
        
        # Encode and decode
        latent, output_logits = super().forward(combined_seq)
        
        return latent, output_logits
    
    def reconstruct(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> torch.Tensor:
        latent, output_logits = self.forward(input_grid, output_grid)
        
        # Get predicted tokens
        predicted_tokens = output_logits.argmax(dim=-1)  # [batch_size, output_seq_len]
        
        # Convert back to grid
        _, reconstructed_output = self.sequence_to_grids(predicted_tokens)
        
        return reconstructed_output


class ARCDataset(Dataset):
    def __init__(self, data_dir: str, max_grid_size: int = 30):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.pairs = []
        self._load_data()
    
    def _load_data(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} not found")
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, 'r') as f:
                    json_data = json.load(f)
                
                for pair in json_data.get('train', []):
                    self.pairs.append({
                        'input': pair['input'],
                        'output': pair['output']
                    })
    
    def _pad_grid(self, grid, target_size: int):
        current_height = len(grid)
        current_width = len(grid[0]) if grid else 0
        
        padded_grid = [[0] * target_size for _ in range(target_size)]
        
        for i in range(min(current_height, target_size)):
            for j in range(min(current_width, target_size)):
                padded_grid[i][j] = grid[i][j]
        
        return padded_grid
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        input_grid = pair['input']
        output_grid = pair['output']
        
        input_grid = self._pad_grid(input_grid, self.max_grid_size)
        output_grid = self._pad_grid(output_grid, self.max_grid_size)
        
        input_tensor = torch.tensor(input_grid, dtype=torch.long)
        output_tensor = torch.tensor(output_grid, dtype=torch.long)
        
        return input_tensor, output_tensor


def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    """Simple training function"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for input_grid, output_grid in train_loader:
            input_grid = input_grid.to(device)
            output_grid = output_grid.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            latent, output_logits = model(input_grid, output_grid)
            
            # Calculate loss (only on output part)
            loss = criterion(
                output_logits.reshape(-1, output_logits.size(-1)),
                output_grid.reshape(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for input_grid, output_grid in val_loader:
                    input_grid = input_grid.to(device)
                    output_grid = output_grid.to(device)
                    
                    latent, output_logits = model(input_grid, output_grid)
                    loss = criterion(
                        output_logits.reshape(-1, output_logits.size(-1)),
                        output_grid.reshape(-1)
                    )
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            print(f"Validation Loss: {avg_val_loss:.4f}")


def inference(model, input_grid, device='cuda'):
    """Simple inference function"""
    model.eval()
    model.to(device)
    
    if isinstance(input_grid, list):
        input_grid = torch.tensor(input_grid, dtype=torch.long)
    
    input_grid = input_grid.unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        # Create dummy output grid for forward pass
        dummy_output = torch.zeros_like(input_grid)
        
        # Get latent representation
        latent, output_logits = model(input_grid, dummy_output)
        
        # Get predicted output
        predicted_tokens = output_logits.argmax(dim=-1)
        
        # Convert to grid
        batch_size = predicted_tokens.size(0)
        output_seq = predicted_tokens[0, :900]  # Take first 900 tokens as output
        output_grid = output_seq.view(30, 30)
        
        return output_grid.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Simplified ARC Autoencoder')
    parser.add_argument('--mode', choices=['train', 'inference'], required=True)
    parser.add_argument('--data_dir', type=str, default='data/training')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Create model
    model = ARCAutoencoder(
        grid_size=30,
        num_colors=10,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        latent_dim=1024
    )
    
    if args.mode == 'train':
        # Load data
        train_dataset = ARCDataset(args.data_dir, max_grid_size=30)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        # Train model
        train_model(model, train_loader, None, num_epochs=args.epochs, device=args.device)
        
        # Save model
        torch.save(model.state_dict(), 'autoencoder_model.pth')
        print("Model saved to autoencoder_model.pth")
    
    elif args.mode == 'inference':
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint))
            print(f"Model loaded from {args.checkpoint}")
        
        # Example inference
        input_grid = [
            [1, 2, 3, 0, 0],
            [4, 5, 6, 0, 0],
            [7, 8, 9, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        
        output_grid = inference(model, input_grid, device=args.device)
        print("Input grid:")
        for row in input_grid:
            print(row)
        print("\nPredicted output grid:")
        print(output_grid[:5, :5])  # Show first 5x5


if __name__ == "__main__":
    main()