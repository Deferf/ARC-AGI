#!/usr/bin/env python3
"""
Training script for Task-Based Autoencoder.
This script implements the complete training pipeline for the enhanced task-based autoencoder
that processes batches of tasks with training and testing entries, averages decoder outputs
across task elements, and generates test outputs autoregressively.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from tqdm import tqdm
import random
import shutil

from enhanced_task_autoencoder import (
    EnhancedTaskBasedAutoencoder,
    EnhancedTaskBatch,
    EnhancedTaskBasedDataLoader,
    create_enhanced_task_dataloader,
    evaluate_enhanced_model
)
from arc_data_loader import ARCTask, visualize_grid, grid_to_string
from device_utils import setup_device, move_to_device, get_memory_stats


class TaskBasedTrainer:
    """
    Trainer for the task-based autoencoder system.
    """
    def __init__(self, model: EnhancedTaskBasedAutoencoder, device: str = 'cuda',
                 learning_rate: float = 0.0001, weight_decay: float = 1e-5):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_history = []
        self.val_history = []
        
    def train_epoch(self, train_loader: EnhancedTaskBasedDataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_entries = 0
        task_losses = {}
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, task_batch in enumerate(progress_bar):
            # Train on the task batch
            metrics = self.model.train_on_task_batch_enhanced(task_batch, self.device)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            
            # Note: The loss is already computed in train_on_task_batch_enhanced
            # Here we need to compute it again for the backward pass
            loss = self._compute_batch_loss(task_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            batch_loss = metrics['loss']
            batch_entries = metrics['num_entries']
            
            total_loss += batch_loss * batch_entries
            total_entries += batch_entries
            
            # Track per-task losses
            for key, value in metrics.items():
                if key.startswith('task_') and key.endswith('_loss'):
                    task_idx = key.replace('task_', '').replace('_loss', '')
                    if task_idx not in task_losses:
                        task_losses[task_idx] = []
                    task_losses[task_idx].append(value)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'entries': batch_entries
            })
        
        avg_loss = total_loss / total_entries if total_entries > 0 else 0.0
        
        # Calculate average per-task losses
        avg_task_losses = {}
        for task_idx, losses in task_losses.items():
            avg_task_losses[f'task_{task_idx}_loss'] = np.mean(losses)
        
        return {
            'loss': avg_loss,
            'total_entries': total_entries,
            **avg_task_losses
        }
    
    def _compute_batch_loss(self, task_batch: EnhancedTaskBatch) -> torch.Tensor:
        """Compute loss for a task batch."""
        training_entries = task_batch.get_training_entries()
        
        total_loss = 0.0
        num_entries = len(training_entries)
        
        for input_grid, output_grid, task_idx in training_entries:
            # Convert to sequence format
            input_seq = self.model.grid_to_sequence(input_grid, output_grid)
            input_seq = input_seq.unsqueeze(0).to(self.device)
            
            # Create target sequence (just the output part)
            target_seq = output_grid.flatten().unsqueeze(0).to(self.device)
            
            # Forward pass
            latent, output_logits = self.model(input_seq)
            
            # Calculate loss (only on output part)
            output_logits = output_logits[:, -900:]  # Last 900 tokens are output
            loss = self.criterion(output_logits.view(-1, self.model.vocab_size), target_seq.view(-1))
            
            total_loss += loss
        
        return total_loss / num_entries if num_entries > 0 else torch.tensor(0.0, device=self.device)
    
    def validate(self, val_loader: EnhancedTaskBasedDataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_entries = 0
        task_losses = {}
        
        with torch.no_grad():
            for task_batch in val_loader:
                # Compute validation loss
                metrics = self.model.train_on_task_batch_enhanced(task_batch, self.device)
                
                batch_loss = metrics['loss']
                batch_entries = metrics['num_entries']
                
                total_loss += batch_loss * batch_entries
                total_entries += batch_entries
                
                # Track per-task losses
                for key, value in metrics.items():
                    if key.startswith('task_') and key.endswith('_loss'):
                        task_idx = key.replace('task_', '').replace('_loss', '')
                        if task_idx not in task_losses:
                            task_losses[task_idx] = []
                        task_losses[task_idx].append(value)
        
        avg_loss = total_loss / total_entries if total_entries > 0 else 0.0
        
        # Calculate average per-task losses
        avg_task_losses = {}
        for task_idx, losses in task_losses.items():
            avg_task_losses[f'task_{task_idx}_loss'] = np.mean(losses)
        
        return {
            'loss': avg_loss,
            'total_entries': total_entries,
            **avg_task_losses
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
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pt')
            shutil.copy(checkpoint_path, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        return checkpoint['epoch'], checkpoint['metrics']
    
    def train(self, train_loader: EnhancedTaskBasedDataLoader, 
              val_loader: Optional[EnhancedTaskBasedDataLoader], 
              num_epochs: int, checkpoint_dir: str = 'checkpoints', 
              log_dir: str = 'logs', save_freq: int = 5):
        """Complete training loop."""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch + 1)
            self.train_history.append(train_metrics)
            
            # Validation phase
            val_metrics = None
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.val_history.append(val_metrics)
                
                # Update learning rate scheduler
                self.scheduler.step(val_metrics['loss'])
                
                # Check if this is the best model
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(epoch + 1, val_metrics, checkpoint_dir, is_best=True)
            
            # Print metrics
            self._print_metrics(train_metrics, val_metrics)
            
            # Save checkpoint periodically
            if (epoch + 1) % save_freq == 0:
                metrics_to_save = val_metrics if val_metrics else train_metrics
                self.save_checkpoint(epoch + 1, metrics_to_save, checkpoint_dir)
        
        print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    
    def _print_metrics(self, train_metrics: Dict[str, float], 
                      val_metrics: Optional[Dict[str, float]]):
        """Print training and validation metrics."""
        print(f"Training Loss: {train_metrics['loss']:.4f}")
        print(f"Training Entries: {train_metrics['total_entries']}")
        
        if val_metrics:
            print(f"Validation Loss: {val_metrics['loss']:.4f}")
            print(f"Validation Entries: {val_metrics['total_entries']}")
        
        # Print per-task losses
        print("\nPer-task training losses:")
        for key, value in train_metrics.items():
            if key.startswith('task_') and key.endswith('_loss'):
                print(f"  {key}: {value:.4f}")
        
        if val_metrics:
            print("\nPer-task validation losses:")
            for key, value in val_metrics.items():
                if key.startswith('task_') and key.endswith('_loss'):
                    print(f"  {key}: {value:.4f}")


def create_sample_tasks_for_training(num_tasks: int = 10) -> List[ARCTask]:
    """Create sample tasks for training."""
    tasks = []
    
    for i in range(num_tasks):
        # Create different types of patterns
        pattern_type = i % 4
        
        if pattern_type == 0:
            # Copy pattern
            task_data = {
                'train': [
                    {
                        'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        'output': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
                    },
                    {
                        'input': [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                        'output': [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
                    }
                ],
                'test': [
                    {
                        'input': [[2, 3, 4], [5, 6, 7], [8, 9, 10]],
                        'output': [[2, 3, 4], [5, 6, 7], [8, 9, 10]]
                    }
                ]
            }
        elif pattern_type == 1:
            # Rotation pattern
            task_data = {
                'train': [
                    {
                        'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        'output': [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
                    },
                    {
                        'input': [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                        'output': [[6, 3, 0], [7, 4, 1], [8, 5, 2]]
                    }
                ],
                'test': [
                    {
                        'input': [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                        'output': [[3, 2, 1], [3, 2, 1], [3, 2, 1]]
                    }
                ]
            }
        elif pattern_type == 2:
            # Color shift pattern
            task_data = {
                'train': [
                    {
                        'input': [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                        'output': [[2, 2, 2], [3, 3, 3], [4, 4, 4]]
                    },
                    {
                        'input': [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                        'output': [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
                    }
                ],
                'test': [
                    {
                        'input': [[5, 5, 5], [6, 6, 6], [7, 7, 7]],
                        'output': [[6, 6, 6], [7, 7, 7], [8, 8, 8]]
                    }
                ]
            }
        else:
            # Mirror pattern
            task_data = {
                'train': [
                    {
                        'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        'output': [[3, 2, 1], [6, 5, 4], [9, 8, 7]]
                    },
                    {
                        'input': [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                        'output': [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
                    }
                ],
                'test': [
                    {
                        'input': [[2, 3, 4], [5, 6, 7], [8, 9, 10]],
                        'output': [[4, 3, 2], [7, 6, 5], [10, 9, 8]]
                    }
                ]
            }
        
        task = ARCTask.from_json(f"task_{i}", task_data)
        tasks.append(task)
    
    return tasks


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Task-Based Autoencoder')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing ARC task JSON files')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Number of tasks per batch')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save logs')
    parser.add_argument('--max_grid_size', type=int, default=10,
                       help='Maximum grid size')
    parser.add_argument('--d_model', type=int, default=128,
                       help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=512,
                       help='Feed-forward dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of transformer layers')
    parser.add_argument('--latent_dim', type=int, default=256,
                       help='Latent dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/mps/cuda/cpu)')
    parser.add_argument('--save_freq', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--save_eval_images', action='store_true',
                       help='Save input/output/prediction images during evaluation')
    parser.add_argument('--eval_output_dir', type=str, default='evaluation_images',
                       help='Directory to save evaluation images')
    parser.add_argument('--evaluate', action='store_true',
                       help='Only evaluate model, do not train')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to load for evaluation')
    
    args = parser.parse_args()
    
    # Set device with Metal backend support
    from evaluation_utils import get_device
    device = get_device(args.device)
    
    print(f"Using device: {device}")
    
    # Create model
    model = EnhancedTaskBasedAutoencoder(
        grid_size=args.max_grid_size,
        num_colors=10,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        dropout=args.dropout
    )
    
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = TaskBasedTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Evaluation mode
    if args.evaluate:
        print("Running evaluation...")
        if args.val_dir and os.path.exists(args.val_dir):
            # Load evaluation data
            val_loader = create_enhanced_task_dataloader(
                args.val_dir, args.batch_size, args.max_grid_size
            )
            # Get all tasks from the loader
            test_tasks = val_loader.tasks
        else:
            # Create sample tasks for evaluation
            test_tasks = create_sample_tasks_for_training(10)
        
        # Evaluate model
        results = evaluate_enhanced_model(
            model, test_tasks, device,
            save_images=args.save_eval_images,
            output_dir=args.eval_output_dir
        )
        
        print("\nEvaluation Results:")
        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        print(f"Prediction Accuracy: {results['prediction_accuracy']:.4f}")
        print(f"Total Tasks: {results['total_tasks']}")
        print(f"Total Predictions: {results['total_predictions']}")
        print(f"Correct Predictions: {results['correct_predictions']}")
        
        if args.save_eval_images:
            print(f"\nEvaluation images saved to: {args.eval_output_dir}")
        
        return
    
    # Load or create data
    if args.data_dir and os.path.exists(args.data_dir):
        print(f"Loading tasks from {args.data_dir}")
        train_loader = create_enhanced_task_dataloader(
            args.data_dir, args.batch_size, args.max_grid_size
        )
        # For simplicity, we'll use the same data for validation
        val_loader = create_enhanced_task_dataloader(
            args.data_dir, args.batch_size, args.max_grid_size
        )
    else:
        print("Creating sample tasks for training")
        tasks = create_sample_tasks_for_training(20)
        
        # Split into train and validation
        random.shuffle(tasks)
        split_idx = int(len(tasks) * (1 - args.val_split))
        train_tasks = tasks[:split_idx]
        val_tasks = tasks[split_idx:]
        
        train_loader = EnhancedTaskBasedDataLoader(
            train_tasks, args.batch_size, args.max_grid_size
        )
        val_loader = EnhancedTaskBasedDataLoader(
            val_tasks, args.batch_size, args.max_grid_size
        )
    
    print(f"Training on {len(train_loader)} batches")
    print(f"Validating on {len(val_loader)} batches")
    
    # Train the model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        save_freq=args.save_freq
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()