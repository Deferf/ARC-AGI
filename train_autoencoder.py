#!/usr/bin/env python3
"""
Training script for Transformer Autoencoder.
Encoder: 1+900+900 tokens -> 1024-dimensional vector
Decoder: 1024-dimensional vector -> 900+900 tokens
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from tqdm import tqdm
import random

from transformer_autoencoder import TransformerAutoencoder, ARCAutoencoder, AutoencoderTrainer
from arc_data_loader import create_arc_dataloaders, ARCTestDataset, visualize_grid, grid_to_string
from device_utils import setup_device, move_to_device, get_memory_stats


class AutoencoderDataset(torch.utils.data.Dataset):
    """
    Dataset for autoencoder training that creates 1+900+900 token sequences.
    """
    def __init__(self, data_dir: str, max_grid_size: int = 30, augment: bool = True):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.augment = augment
        self.pairs = []
        
        self._load_data()
    
    def _load_data(self):
        """Load ARC data and create training pairs."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} not found")
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.data_dir, filename)
                
                with open(filepath, 'r') as f:
                    json_data = json.load(f)
                
                # Create pairs from training data
                for pair in json_data.get('train', []):
                    self.pairs.append({
                        'input': pair['input'],
                        'output': pair['output']
                    })
    
    def _pad_grid(self, grid: List[List[int]], target_size: int) -> List[List[int]]:
        """Pad grid to target size with zeros."""
        current_height = len(grid)
        current_width = len(grid[0]) if grid else 0
        
        padded_grid = [[0] * target_size for _ in range(target_size)]
        
        for i in range(min(current_height, target_size)):
            for j in range(min(current_width, target_size)):
                padded_grid[i][j] = grid[i][j]
        
        return padded_grid
    
    def _augment_grid(self, grid: List[List[int]]) -> List[List[int]]:
        """Apply data augmentation to grid."""
        if not self.augment:
            return grid
        
        # Random rotation (0, 90, 180, 270 degrees)
        rotation = random.choice([0, 1, 2, 3])
        for _ in range(rotation):
            grid = list(zip(*grid[::-1]))  # Rotate 90 degrees clockwise
            grid = [list(row) for row in grid]
        
        # Random horizontal flip
        if random.random() < 0.5:
            grid = [row[::-1] for row in grid]
        
        # Random vertical flip
        if random.random() < 0.5:
            grid = grid[::-1]
        
        return grid
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx]
        
        # Get input and output grids
        input_grid = pair['input']
        output_grid = pair['output']
        
        # Apply augmentation
        if self.augment:
            input_grid = self._augment_grid(input_grid)
            output_grid = self._augment_grid(output_grid)
        
        # Pad grids to maximum size
        input_grid = self._pad_grid(input_grid, self.max_grid_size)
        output_grid = self._pad_grid(output_grid, self.max_grid_size)
        
        # Convert to tensors
        input_tensor = torch.tensor(input_grid, dtype=torch.long)
        output_tensor = torch.tensor(output_grid, dtype=torch.long)
        
        # Flatten grids
        input_seq = input_tensor.flatten()  # [900] for 30x30
        output_seq = output_tensor.flatten()  # [900] for 30x30
        
        # Create combined sequence: [start_token] + input_seq + output_seq
        start_token = torch.tensor([1], dtype=torch.long)  # Start token
        combined_seq = torch.cat([start_token, input_seq, output_seq])  # [1 + 900 + 900 = 1801]
        
        # Target is just the output sequence
        target_seq = output_seq  # [900]
        
        return combined_seq, target_seq


class AutoencoderDataLoader:
    """
    Data loader for autoencoder training.
    """
    def __init__(self, train_dir: str, val_dir: Optional[str] = None,
                 batch_size: int = 32, max_grid_size: int = 30,
                 num_workers: int = 4):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.max_grid_size = max_grid_size
        self.num_workers = num_workers
    
    def get_loaders(self):
        """Get training and validation data loaders."""
        train_dataset = AutoencoderDataset(
            self.train_dir, 
            max_grid_size=self.max_grid_size, 
            augment=True
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = None
        if self.val_dir:
            val_dataset = AutoencoderDataset(
                self.val_dir,
                max_grid_size=self.max_grid_size,
                augment=False
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        
        return train_loader, val_loader


class AutoencoderTrainerExtended(AutoencoderTrainer):
    """
    Extended trainer with additional functionality for autoencoder training.
    """
    def __init__(self, model: nn.Module, device: str = 'cuda',
                 learning_rate: float = 0.0001, weight_decay: float = 1e-5):
        super().__init__(model, device, learning_rate, weight_decay)
    
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (input_seq, target_seq) in enumerate(progress_bar):
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Encode and decode
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
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}'
            })
            
            # Log to TensorBoard
            # if self.writer:
            #     self.writer.add_scalar('Loss/train_step', loss.item(), self.global_step)
            #     self.global_step += 1
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for input_seq, target_seq in tqdm(val_loader, desc="Validation"):
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
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], 
                       checkpoint_dir: str, is_best: bool = False):
        """Save model checkpoint."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'global_step': self.global_step
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'autoencoder_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_autoencoder.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best autoencoder with validation loss: {metrics.get('val_loss', 0):.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        metrics = checkpoint['metrics']
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"Loaded checkpoint from epoch {epoch}")
        print(f"Metrics: {metrics}")
        
        return epoch, metrics
    
    def train(self, train_loader, val_loader, num_epochs: int, 
              checkpoint_dir: str = 'checkpoints', log_dir: str = 'logs'):
        """Main training loop."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate(val_loader) if val_loader else {'val_loss': float('inf'), 'val_accuracy': 0.0}
            
            # Update learning rate
            self.scheduler.step()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Print progress
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.2f}s)")
            print(f"Train Loss: {metrics['train_loss']:.4f}")
            if val_loader:
                print(f"Val Loss: {metrics['val_loss']:.4f}")
                print(f"Val Accuracy: {metrics['val_accuracy']:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = metrics['val_loss']
            
            self.save_checkpoint(epoch, metrics, checkpoint_dir, is_best)
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, metrics, checkpoint_dir, False)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")


def evaluate_autoencoder(model: TransformerAutoencoder, test_loader, device: str = 'cuda') -> Dict[str, float]:
    """Evaluate autoencoder on test set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for input_seq, target_seq in tqdm(test_loader, desc="Testing"):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            latent, output_logits = model(input_seq)
            
            loss = nn.CrossEntropyLoss(ignore_index=0)(
                output_logits.reshape(-1, output_logits.size(-1)),
                target_seq.reshape(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            predictions = output_logits.argmax(dim=-1)
            correct_predictions += (predictions == target_seq).sum().item()
            total_predictions += target_seq.numel()
    
    avg_loss = total_loss / num_batches
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    return {
        'test_loss': avg_loss,
        'test_accuracy': accuracy
    }


def generate_from_latent(model: TransformerAutoencoder, latent: torch.Tensor, 
                        device: str = 'cuda') -> torch.Tensor:
    """Generate output sequence from latent vector."""
    model.eval()
    with torch.no_grad():
        latent = latent.to(device)
        output = model.decode(latent)
        tokens = output.argmax(dim=-1)
    
    return tokens


def main():
    parser = argparse.ArgumentParser(description='Train Transformer Autoencoder')
    parser.add_argument('--train_dir', type=str, default='data/training',
                       help='Directory containing training tasks')
    parser.add_argument('--val_dir', type=str, default='data/evaluation',
                       help='Directory containing validation tasks')
    parser.add_argument('--test_dir', type=str, default='data/evaluation',
                       help='Directory containing test tasks')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--d_model', type=int, default=512,
                       help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--latent_dim', type=int, default=1024,
                       help='Latent dimension')
    parser.add_argument('--max_grid_size', type=int, default=30,
                       help='Maximum grid size')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save logs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/mps/cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--evaluate', action='store_true',
                       help='Only evaluate model, do not train')
    parser.add_argument('--arc_mode', action='store_true',
                       help='Use ARC-specific autoencoder')
    
    args = parser.parse_args()
    
    # Set device with Metal support
    torch_device = setup_device(args.device, verbose=True)
    device = str(torch_device)
    
    # Create model
    if args.arc_mode:
        model = ARCAutoencoder(
            grid_size=args.max_grid_size,
            num_colors=10,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_model * 4,
            num_layers=args.num_layers,
            latent_dim=args.latent_dim
        )
        print("Created ARC Autoencoder")
    else:
        # For general autoencoder, we need to estimate vocab size
        vocab_size = 1000  # Default, can be adjusted based on data
        model = TransformerAutoencoder(
            vocab_size=vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_model * 4,
            num_layers=args.num_layers,
            latent_dim=args.latent_dim
        )
        print("Created General Transformer Autoencoder")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input sequence length: {model.input_seq_len}")
    print(f"Output sequence length: {model.output_seq_len}")
    print(f"Latent dimension: {model.latent_dim}")
    
    # Create trainer
    trainer = AutoencoderTrainerExtended(model, device=device, learning_rate=args.learning_rate)
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    if args.evaluate:
        # Load test data
        test_dataset = AutoencoderDataset(args.test_dir, max_grid_size=args.max_grid_size)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        
        # Evaluate model
        test_metrics = evaluate_autoencoder(model, test_loader, device)
        print(f"Test metrics: {test_metrics}")
    
    else:
        # Training mode
        print("Creating data loaders...")
        data_loader = AutoencoderDataLoader(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            max_grid_size=args.max_grid_size,
            num_workers=4
        )
        
        train_loader, val_loader = data_loader.get_loaders()
        
        print(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"Validation samples: {len(val_loader.dataset)}")
        
        # Start training
        print("Starting training...")
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir
        )


if __name__ == "__main__":
    main()