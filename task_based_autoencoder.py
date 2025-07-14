import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformer_autoencoder import TransformerAutoencoder, ARCAutoencoder
from arc_data_loader import ARCTask


class TaskBatch:
    """
    Represents a batch of tasks, where each task contains training and testing entries.
    """
    def __init__(self, tasks: List[ARCTask], max_grid_size: int = 30):
        self.tasks = tasks
        self.max_grid_size = max_grid_size
        self.task_ids = [task.task_id for task in tasks]
        
    def get_training_entries(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get all training entries from all tasks in the batch."""
        training_entries = []
        for task in self.tasks:
            for pair in task.train_pairs:
                input_grid = self._pad_grid(pair['input'])
                output_grid = self._pad_grid(pair['output'])
                training_entries.append((input_grid, output_grid))
        return training_entries
    
    def get_testing_entries(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get all testing entries from all tasks in the batch."""
        testing_entries = []
        for task in self.tasks:
            for pair in task.test_pairs:
                input_grid = self._pad_grid(pair['input'])
                output_grid = self._pad_grid(pair['output'])
                testing_entries.append((input_grid, output_grid))
        return testing_entries
    
    def _pad_grid(self, grid: List[List[int]]) -> torch.Tensor:
        """Pad grid to target size with zeros."""
        current_height = len(grid)
        current_width = len(grid[0]) if grid else 0
        
        padded_grid = [[0] * self.max_grid_size for _ in range(self.max_grid_size)]
        
        for i in range(min(current_height, self.max_grid_size)):
            for j in range(min(current_width, self.max_grid_size)):
                padded_grid[i][j] = grid[i][j]
        
        return torch.tensor(padded_grid, dtype=torch.long)


class TaskBasedAutoencoder(ARCAutoencoder):
    """
    Extended autoencoder that processes task-based batches with autoregressive generation.
    """
    def __init__(self, grid_size: int = 30, num_colors: int = 10, d_model: int = 512,
                 num_heads: int = 8, d_ff: int = 2048, num_layers: int = 6,
                 latent_dim: int = 1024, dropout: float = 0.1):
        super().__init__(grid_size, num_colors, d_model, num_heads, d_ff, num_layers, latent_dim, dropout)
        
    def process_task_batch(self, task_batch: TaskBatch, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """
        Process a batch of tasks, averaging decoder outputs across training entries
        and using them for autoregressive generation of test outputs.
        
        Args:
            task_batch: Batch of tasks with training and testing entries
            device: Device to run computation on
            
        Returns:
            Dictionary containing:
            - 'task_latents': Average latent representations for each task
            - 'training_outputs': Decoder outputs for training entries
            - 'test_predictions': Autoregressively generated test outputs
        """
        # Get training and testing entries
        training_entries = task_batch.get_training_entries()
        testing_entries = task_batch.get_testing_entries()
        
        if not training_entries:
            raise ValueError("No training entries found in task batch")
        
        # Process training entries to get latent representations
        task_latents = []
        training_outputs = []
        
        # Group training entries by task
        task_training_groups = self._group_entries_by_task(training_entries, task_batch.tasks)
        
        for task_idx, task_entries in enumerate(task_training_groups):
            task_latent_list = []
            task_output_list = []
            
            for input_grid, output_grid in task_entries:
                # Convert to sequence format
                input_seq = self.grid_to_sequence(input_grid, output_grid)
                input_seq = input_seq.unsqueeze(0).to(device)  # Add batch dimension
                
                # Encode to get latent representation
                with torch.no_grad():
                    latent = self.encode(input_seq)
                    task_latent_list.append(latent)
                    
                    # Decode to get output
                    decoded_output = self.decode(latent)
                    task_output_list.append(decoded_output)
            
            # Average latent representations across training entries for this task
            if task_latent_list:
                avg_latent = torch.mean(torch.cat(task_latent_list, dim=0), dim=0, keepdim=True)
                task_latents.append(avg_latent)
                
                # Store training outputs
                training_outputs.extend(task_output_list)
        
        # Process testing entries autoregressively
        test_predictions = []
        task_testing_groups = self._group_entries_by_task(testing_entries, task_batch.tasks)
        
        for task_idx, task_entries in enumerate(task_testing_groups):
            if task_idx < len(task_latents):
                avg_latent = task_latents[task_idx]
                
                for input_grid, _ in task_entries:
                    # Generate output autoregressively using the averaged latent
                    predicted_output = self._generate_autoregressive(
                        avg_latent, input_grid, device
                    )
                    test_predictions.append(predicted_output)
        
        return {
            'task_latents': torch.cat(task_latents, dim=0) if task_latents else torch.empty(0),
            'training_outputs': training_outputs,
            'test_predictions': test_predictions
        }
    
    def _group_entries_by_task(self, entries: List[Tuple[torch.Tensor, torch.Tensor]], 
                              tasks: List[ARCTask]) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Group entries by their corresponding tasks."""
        task_groups = [[] for _ in tasks]
        
        # For simplicity, we'll assume entries are in the same order as tasks
        # In a real implementation, you'd need to track which entry belongs to which task
        entries_per_task = len(entries) // len(tasks)
        
        for i, task in enumerate(tasks):
            start_idx = i * entries_per_task
            end_idx = start_idx + entries_per_task
            task_groups[i] = entries[start_idx:end_idx]
        
        return task_groups
    
    def _generate_autoregressive(self, latent: torch.Tensor, input_grid: torch.Tensor, 
                                device: str = 'cuda', max_steps: int = 100) -> torch.Tensor:
        """
        Generate output grid autoregressively using the input grid as context.
        
        Args:
            latent: Average latent representation for the task
            input_grid: Input grid to use as context
            device: Device to run computation on
            max_steps: Maximum number of generation steps
            
        Returns:
            Generated output grid
        """
        # Initialize output grid with zeros
        output_grid = torch.zeros_like(input_grid)
        
        # Convert input grid to sequence format for context
        input_seq = input_grid.flatten()
        
        # Start with the latent representation
        current_latent = latent.to(device)
        
        # Autoregressive generation
        for step in range(max_steps):
            # Use current latent and input context to generate next token
            # This is a simplified version - in practice, you'd want more sophisticated
            # autoregressive generation with proper masking
            
            # Decode current latent
            decoded = self.decode(current_latent)
            
            # Extract the output part (last 900 tokens)
            output_part = decoded[:, -900:]  # Assuming 30x30 grid = 900 tokens
            
            # Reshape to grid format
            output_grid = output_part.view(1, self.grid_size, self.grid_size)
            
            # For simplicity, we'll stop after one iteration
            # In a more sophisticated implementation, you'd continue generating
            # based on the current output and input context
            break
        
        return output_grid.squeeze(0)  # Remove batch dimension
    
    def train_on_task_batch(self, task_batch: TaskBatch, device: str = 'cuda') -> Dict[str, float]:
        """
        Train the model on a task batch using both training and testing entries.
        
        Args:
            task_batch: Batch of tasks to train on
            device: Device to run computation on
            
        Returns:
            Dictionary containing training metrics
        """
        self.train()
        
        # Get all training entries
        training_entries = task_batch.get_training_entries()
        
        total_loss = 0.0
        num_entries = len(training_entries)
        
        for input_grid, output_grid in training_entries:
            # Convert to sequence format
            input_seq = self.grid_to_sequence(input_grid, output_grid)
            input_seq = input_seq.unsqueeze(0).to(device)  # Add batch dimension
            
            # Create target sequence (just the output part)
            target_seq = output_grid.flatten().unsqueeze(0).to(device)
            
            # Forward pass
            latent, output_logits = self(input_seq)
            
            # Calculate loss (only on output part)
            output_logits = output_logits[:, -900:]  # Last 900 tokens are output
            loss = F.cross_entropy(output_logits.view(-1, self.vocab_size), target_seq.view(-1))
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_entries if num_entries > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'num_entries': num_entries
        }


class TaskBasedDataLoader:
    """
    Data loader that creates task-based batches.
    """
    def __init__(self, tasks: List[ARCTask], batch_size: int = 4, max_grid_size: int = 30):
        self.tasks = tasks
        self.batch_size = batch_size
        self.max_grid_size = max_grid_size
        
    def __iter__(self):
        """Create batches of tasks."""
        for i in range(0, len(self.tasks), self.batch_size):
            batch_tasks = self.tasks[i:i + self.batch_size]
            yield TaskBatch(batch_tasks, self.max_grid_size)
    
    def __len__(self):
        return (len(self.tasks) + self.batch_size - 1) // self.batch_size


def create_task_based_dataloader(data_dir: str, batch_size: int = 4, 
                                max_grid_size: int = 30) -> TaskBasedDataLoader:
    """
    Create a task-based data loader from a directory of ARC task files.
    
    Args:
        data_dir: Directory containing ARC task JSON files
        batch_size: Number of tasks per batch
        max_grid_size: Maximum grid size for padding
        
    Returns:
        TaskBasedDataLoader instance
    """
    import os
    import json
    
    tasks = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            task_id = filename.replace('.json', '')
            filepath = os.path.join(data_dir, filename)
            
            with open(filepath, 'r') as f:
                json_data = json.load(f)
            
            task = ARCTask.from_json(task_id, json_data)
            tasks.append(task)
    
    return TaskBasedDataLoader(tasks, batch_size, max_grid_size)


def evaluate_task_based_model(model: TaskBasedAutoencoder, test_tasks: List[ARCTask], 
                            device: str = 'cuda', save_images: bool = False,
                            output_dir: str = 'evaluation_images') -> Dict[str, float]:
    """
    Evaluate the task-based model on test tasks.
    
    Args:
        model: TaskBasedAutoencoder model
        test_tasks: List of test tasks
        device: Device to run computation on
        save_images: Whether to save input/output/prediction images
        output_dir: Directory to save evaluation images
        
    Returns:
        Dictionary containing evaluation metrics
    """
    from evaluation_utils import save_input_output_pair, save_evaluation_summary
    
    model.eval()
    
    total_accuracy = 0.0
    total_tasks = len(test_tasks)
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for task_idx, task in enumerate(test_tasks):
            # Create a single-task batch
            task_batch = TaskBatch([task])
            
            # Process the task
            results = model.process_task_batch(task_batch, device)
            
            # Compare predictions with ground truth
            test_entries = task_batch.get_testing_entries()
            predictions = results['test_predictions']
            
            task_accuracy = 0.0
            num_test_entries = len(test_entries)
            
            for i, (input_grid, expected_output) in enumerate(test_entries):
                if i < len(predictions):
                    predicted_output = predictions[i]
                    
                    # Calculate accuracy (exact match)
                    is_correct = torch.equal(predicted_output, expected_output)
                    if is_correct:
                        task_accuracy += 1.0
                        correct_predictions += 1
                    
                    total_predictions += 1
                    
                    # Save images if requested
                    if save_images:
                        save_input_output_pair(
                            input_grid=input_grid,
                            output_grid=expected_output,
                            predicted_grid=predicted_output,
                            task_id=task.task_id,
                            pair_idx=i,
                            output_dir=output_dir
                        )
            
            task_accuracy /= num_test_entries if num_test_entries > 0 else 1.0
            total_accuracy += task_accuracy
    
    avg_accuracy = total_accuracy / total_tasks if total_tasks > 0 else 0.0
    prediction_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    results = {
        'task_accuracy': avg_accuracy,
        'prediction_accuracy': prediction_accuracy,
        'total_tasks': total_tasks,
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions
    }
    
    # Save evaluation summary if images were saved
    if save_images:
        save_evaluation_summary(results, output_dir)
    
    return results