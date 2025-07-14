#!/usr/bin/env python3
"""
Training script for ARC Transformer model.
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

from transformer import ARCGridTransformer, TransformerTrainer
from arc_data_loader import create_arc_dataloaders, ARCTestDataset, visualize_grid, grid_to_string
from device_utils import setup_device, move_to_device, get_memory_stats


class ARCTrainer:
    """
    Comprehensive trainer for ARC Transformer models.
    """
    def __init__(self, model: ARCGridTransformer, device: str = 'cuda',
                 learning_rate: float = 0.0001, weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        self.writer = None
        self.global_step = 0
        
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (input_grids, target_grids) in enumerate(progress_bar):
            input_grids = input_grids.to(self.device)
            target_grids = target_grids.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Convert grids to sequences for transformer
            batch_size = input_grids.size(0)
            input_seqs = input_grids.view(batch_size, -1)  # [batch_size, height*width]
            target_seqs = target_grids.view(batch_size, -1)  # [batch_size, height*width]
            
            # Remove last token from target for input, remove first token for output
            target_input = target_seqs[:, :-1]
            target_output = target_seqs[:, 1:]
            
            # Forward pass through transformer
            output = self.model.transformer(input_seqs, target_input)
            
            # Calculate loss
            loss = self.criterion(
                output.reshape(-1, output.size(-1)),
                target_output.reshape(-1)
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
            if self.writer:
                self.writer.add_scalar('Loss/train_step', loss.item(), self.global_step)
                self.global_step += 1
        
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
            for input_grids, target_grids in tqdm(val_loader, desc="Validation"):
                input_grids = input_grids.to(self.device)
                target_grids = target_grids.to(self.device)
                
                batch_size = input_grids.size(0)
                input_seqs = input_grids.view(batch_size, -1)
                target_seqs = target_grids.view(batch_size, -1)
                
                target_input = target_seqs[:, :-1]
                target_output = target_seqs[:, 1:]
                
                output = self.model.transformer(input_seqs, target_input)
                
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    target_output.reshape(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Calculate accuracy
                predictions = output.argmax(dim=-1)
                correct_predictions += (predictions == target_output).sum().item()
                total_predictions += target_output.numel()
        
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
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation loss: {metrics.get('val_loss', 0):.4f}")
    
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
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Print progress
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.2f}s)")
            print(f"Train Loss: {metrics['train_loss']:.4f}")
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
        
        # REMOVED: if self.writer: self.writer.close()


def evaluate_model(model: ARCGridTransformer, test_loader, device: str = 'cuda') -> Dict[str, float]:
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_grids = batch['input'].to(device)
            target_grids = batch['output'].to(device)
            
            batch_size = input_grids.size(0)
            input_seqs = input_grids.view(batch_size, -1)
            target_seqs = target_grids.view(batch_size, -1)
            
            target_input = target_seqs[:, :-1]
            target_output = target_seqs[:, 1:]
            
            output = model.transformer(input_seqs, target_input)
            
            loss = nn.CrossEntropyLoss(ignore_index=0)(
                output.reshape(-1, output.size(-1)),
                target_output.reshape(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            predictions = output.argmax(dim=-1)
            correct_predictions += (predictions == target_output).sum().item()
            total_predictions += target_output.numel()
    
    avg_loss = total_loss / num_batches
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    return {
        'test_loss': avg_loss,
        'test_accuracy': accuracy
    }


def generate_solutions(model: ARCGridTransformer, test_loader, device: str = 'cuda',
                      output_dir: str = 'solutions'):
    """Generate solutions for test tasks."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    solutions = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating solutions"):
            task_ids = batch['task_id']
            pair_indices = batch['pair_index']
            input_grids = batch['input'].to(device)
            
            batch_size = input_grids.size(0)
            
            for i in range(batch_size):
                task_id = task_ids[i]
                pair_index = pair_indices[i]
                input_grid = input_grids[i:i+1]  # Keep batch dimension
                
                # Generate solution
                try:
                    generated_grid = model.generate(input_grid, max_length=900)
                    
                    # Convert to list format for JSON
                    generated_list = generated_grid[0].cpu().numpy().tolist()
                    
                    # Store solution
                    if task_id not in solutions:
                        solutions[task_id] = {}
                    solutions[task_id][f'test_{pair_index}'] = generated_list
                    
                except Exception as e:
                    print(f"Error generating solution for {task_id}: {e}")
                    # Use input as fallback
                    fallback_grid = input_grid[0].cpu().numpy().tolist()
                    if task_id not in solutions:
                        solutions[task_id] = {}
                    solutions[task_id][f'test_{pair_index}'] = fallback_grid
    
    # Save solutions
    solutions_file = os.path.join(output_dir, 'solutions.json')
    with open(solutions_file, 'w') as f:
        json.dump(solutions, f, indent=2)
    
    print(f"Saved solutions to {solutions_file}")
    return solutions


def main():
    parser = argparse.ArgumentParser(description='Train ARC Transformer')
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
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')
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
    parser.add_argument('--generate', action='store_true',
                       help='Generate solutions for test tasks')
    
    args = parser.parse_args()
    
    # Set device with Metal support
    torch_device = setup_device(args.device, verbose=True)
    device = str(torch_device)
    
    # Create model
    model = ARCGridTransformer(
        grid_size=args.max_grid_size,
        num_colors=10,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_model * 4,
        num_layers=args.num_layers,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = ARCTrainer(model, device=device, learning_rate=args.learning_rate)
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    if args.evaluate or args.generate:
        # Load test data
        test_dataset = ARCTestDataset(args.test_dir, max_grid_size=args.max_grid_size)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        
        if args.evaluate:
            # Evaluate model
            test_metrics = evaluate_model(model, test_loader, device)
            print(f"Test metrics: {test_metrics}")
        
        if args.generate:
            # Generate solutions
            solutions = generate_solutions(model, test_loader, device)
            print(f"Generated solutions for {len(solutions)} tasks")
    
    else:
        # Training mode
        print("Creating data loaders...")
        train_loader, val_loader = create_arc_dataloaders(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            max_grid_size=args.max_grid_size,
            num_workers=4
        )
        
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