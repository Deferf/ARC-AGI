import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformer_autoencoder import TransformerAutoencoder, ARCAutoencoder
from arc_data_loader import ARCTask


class EnhancedTaskBatch:
    """
    Enhanced task batch with better organization and tracking of task relationships.
    """
    def __init__(self, tasks: List[ARCTask], max_grid_size: int = 30):
        self.tasks = tasks
        self.max_grid_size = max_grid_size
        self.task_ids = [task.task_id for task in tasks]
        
        # Create mapping from entries to task indices
        self.entry_to_task = {}
        self.task_to_entries = {i: {'train': [], 'test': []} for i in range(len(tasks))}
        
        self._build_entry_mappings()
    
    def _build_entry_mappings(self):
        """Build mappings between entries and their corresponding tasks."""
        entry_idx = 0
        
        for task_idx, task in enumerate(self.tasks):
            # Map training entries
            for pair_idx, pair in enumerate(task.train_pairs):
                self.task_to_entries[task_idx]['train'].append(entry_idx)
                self.entry_to_task[entry_idx] = {'task_idx': task_idx, 'type': 'train', 'pair_idx': pair_idx}
                entry_idx += 1
            
            # Map testing entries
            for pair_idx, pair in enumerate(task.test_pairs):
                self.task_to_entries[task_idx]['test'].append(entry_idx)
                self.entry_to_task[entry_idx] = {'task_idx': task_idx, 'type': 'test', 'pair_idx': pair_idx}
                entry_idx += 1
    
    def get_training_entries(self) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
        """Get all training entries with their task indices."""
        training_entries = []
        for task_idx, task in enumerate(self.tasks):
            for pair in task.train_pairs:
                input_grid = self._pad_grid(pair['input'])
                output_grid = self._pad_grid(pair['output'])
                training_entries.append((input_grid, output_grid, task_idx))
        return training_entries
    
    def get_testing_entries(self) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
        """Get all testing entries with their task indices."""
        testing_entries = []
        for task_idx, task in enumerate(self.tasks):
            for pair in task.test_pairs:
                input_grid = self._pad_grid(pair['input'])
                output_grid = self._pad_grid(pair['output'])
                testing_entries.append((input_grid, output_grid, task_idx))
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


class EnhancedTaskBasedAutoencoder(ARCAutoencoder):
    """
    Enhanced autoencoder with sophisticated task-based processing and autoregressive generation.
    """
    def __init__(self, grid_size: int = 30, num_colors: int = 10, d_model: int = 512,
                 num_heads: int = 8, d_ff: int = 2048, num_layers: int = 6,
                 latent_dim: int = 1024, dropout: float = 0.1):
        super().__init__(grid_size, num_colors, d_model, num_heads, d_ff, num_layers, latent_dim, dropout)
        
        # Additional components for enhanced autoregressive generation
        self.context_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def process_task_batch_enhanced(self, task_batch: EnhancedTaskBatch, device: str = 'cuda') -> Dict[str, torch.Tensor]:
        """
        Enhanced processing of task batches with sophisticated autoregressive generation.
        
        Args:
            task_batch: Enhanced task batch
            device: Device to run computation on
            
        Returns:
            Dictionary containing processing results
        """
        # Get training and testing entries with task indices
        training_entries = task_batch.get_training_entries()
        testing_entries = task_batch.get_testing_entries()
        
        if not training_entries:
            raise ValueError("No training entries found in task batch")
        
        # Step 1: Process training entries and compute task-specific representations
        task_representations = self._compute_task_representations(training_entries, device)
        
        # Step 2: Generate test outputs using autoregressive generation
        test_predictions = self._generate_test_outputs(testing_entries, task_representations, device)
        
        return {
            'task_representations': task_representations,
            'test_predictions': test_predictions
        }
    
    def _compute_task_representations(self, training_entries: List[Tuple[torch.Tensor, torch.Tensor, int]], 
                                    device: str = 'cuda') -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Compute task-specific representations by averaging across training entries.
        
        Args:
            training_entries: List of (input_grid, output_grid, task_idx) tuples
            device: Device to run computation on
            
        Returns:
            Dictionary mapping task_idx to task representations
        """
        task_latents = {}
        task_outputs = {}
        
        # Group entries by task
        task_groups = {}
        for input_grid, output_grid, task_idx in training_entries:
            if task_idx not in task_groups:
                task_groups[task_idx] = []
            task_groups[task_idx].append((input_grid, output_grid))
        
        # Process each task
        for task_idx, entries in task_groups.items():
            task_latent_list = []
            task_output_list = []
            
            for input_grid, output_grid in entries:
                # Convert to sequence format
                input_seq = self.grid_to_sequence(input_grid, output_grid)
                input_seq = input_seq.unsqueeze(0).to(device)
                
                # Encode to get latent representation
                with torch.no_grad():
                    latent = self.encode(input_seq)
                    task_latent_list.append(latent)
                    
                    # Decode to get output
                    decoded_output = self.decode(latent)
                    task_output_list.append(decoded_output)
            
            # Average latent representations and outputs
            if task_latent_list:
                avg_latent = torch.mean(torch.cat(task_latent_list, dim=0), dim=0, keepdim=True)
                avg_output = torch.mean(torch.cat(task_output_list, dim=0), dim=0, keepdim=True)
                
                task_latents[task_idx] = avg_latent
                task_outputs[task_idx] = avg_output
        
        return {
            'latents': task_latents,
            'outputs': task_outputs
        }
    
    def _generate_test_outputs(self, testing_entries: List[Tuple[torch.Tensor, torch.Tensor, int]], 
                              task_representations: Dict[int, Dict[str, torch.Tensor]], 
                              device: str = 'cuda') -> List[torch.Tensor]:
        """
        Generate test outputs using enhanced autoregressive generation.
        
        Args:
            testing_entries: List of (input_grid, output_grid, task_idx) tuples
            task_representations: Task-specific representations
            device: Device to run computation on
            
        Returns:
            List of predicted output grids
        """
        predictions = []
        
        for input_grid, _, task_idx in testing_entries:
            if task_idx in task_representations['latents']:
                # Get task-specific representations
                task_latent = task_representations['latents'][task_idx]
                task_output = task_representations['outputs'][task_idx]
                
                # Generate output autoregressively
                predicted_output = self._enhanced_autoregressive_generation(
                    task_latent, task_output, input_grid, device
                )
                predictions.append(predicted_output)
            else:
                # Fallback: use zero grid
                predictions.append(torch.zeros_like(input_grid))
        
        return predictions
    
    def _enhanced_autoregressive_generation(self, task_latent: torch.Tensor, task_output: torch.Tensor,
                                          input_grid: torch.Tensor, device: str = 'cuda',
                                          max_steps: int = 50) -> torch.Tensor:
        """
        Enhanced autoregressive generation using task context and input grid.
        
        Args:
            task_latent: Average latent representation for the task
            task_output: Average decoder output for the task
            input_grid: Input grid to use as context
            device: Device to run computation on
            max_steps: Maximum number of generation steps
            
        Returns:
            Generated output grid
        """
        # Initialize output grid
        output_grid = torch.zeros_like(input_grid)
        
        # Convert input grid to sequence format
        input_seq = input_grid.flatten()
        
        # Start with task latent and output as context
        current_latent = task_latent.to(device)
        task_context = task_output.to(device)
        
        # Extract the output part from task context (last 900 tokens)
        task_output_part = task_context[:, -900:]  # [1, 900]
        
        # Reshape to grid format for processing
        task_output_grid = task_output_part.view(1, self.grid_size, self.grid_size)
        
        # Use task output as initial prediction
        output_grid = task_output_grid.squeeze(0)
        
        # Apply input-specific modifications based on input grid
        # This is where we can implement more sophisticated logic
        # For now, we'll use a simple approach: combine task pattern with input context
        
        # Create a mask based on input grid (non-zero positions)
        input_mask = (input_grid != 0).float()
        
        # Apply the mask to preserve input structure where relevant
        # This is a simplified approach - in practice, you'd want more sophisticated logic
        output_grid = output_grid * (1 - input_mask) + input_grid * input_mask
        
        return output_grid
    
    def train_on_task_batch_enhanced(self, task_batch: EnhancedTaskBatch, device: str = 'cuda') -> Dict[str, float]:
        """
        Enhanced training on task batches with task-aware loss computation.
        
        Args:
            task_batch: Enhanced task batch
            device: Device to run computation on
            
        Returns:
            Dictionary containing training metrics
        """
        self.train()
        
        # Get training entries with task indices
        training_entries = task_batch.get_training_entries()
        
        total_loss = 0.0
        task_losses = {}
        num_entries = len(training_entries)
        
        for input_grid, output_grid, task_idx in training_entries:
            # Convert to sequence format
            input_seq = self.grid_to_sequence(input_grid, output_grid)
            input_seq = input_seq.unsqueeze(0).to(device)
            
            # Create target sequence (just the output part)
            target_seq = output_grid.flatten().unsqueeze(0).to(device)
            
            # Forward pass
            latent, output_logits = self(input_seq)
            
            # Calculate loss (only on output part)
            output_logits = output_logits[:, -900:]  # Last 900 tokens are output
            loss = F.cross_entropy(output_logits.view(-1, self.vocab_size), target_seq.view(-1))
            
            total_loss += loss.item()
            
            # Track per-task loss
            if task_idx not in task_losses:
                task_losses[task_idx] = []
            task_losses[task_idx].append(loss.item())
        
        avg_loss = total_loss / num_entries if num_entries > 0 else 0.0
        
        # Calculate per-task average losses
        avg_task_losses = {}
        for task_idx, losses in task_losses.items():
            avg_task_losses[f'task_{task_idx}_loss'] = np.mean(losses)
        
        return {
            'loss': avg_loss,
            'num_entries': num_entries,
            **avg_task_losses
        }


class EnhancedTaskBasedDataLoader:
    """
    Enhanced data loader for task-based batches.
    """
    def __init__(self, tasks: List[ARCTask], batch_size: int = 4, max_grid_size: int = 30):
        self.tasks = tasks
        self.batch_size = batch_size
        self.max_grid_size = max_grid_size
        
    def __iter__(self):
        """Create enhanced task batches."""
        for i in range(0, len(self.tasks), self.batch_size):
            batch_tasks = self.tasks[i:i + self.batch_size]
            yield EnhancedTaskBatch(batch_tasks, self.max_grid_size)
    
    def __len__(self):
        return (len(self.tasks) + self.batch_size - 1) // self.batch_size


def create_enhanced_task_dataloader(data_dir: str, batch_size: int = 4, 
                                   max_grid_size: int = 30) -> EnhancedTaskBasedDataLoader:
    """
    Create an enhanced task-based data loader.
    
    Args:
        data_dir: Directory containing ARC task JSON files
        batch_size: Number of tasks per batch
        max_grid_size: Maximum grid size for padding
        
    Returns:
        EnhancedTaskBasedDataLoader instance
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
    
    return EnhancedTaskBasedDataLoader(tasks, batch_size, max_grid_size)


def evaluate_enhanced_model(model: EnhancedTaskBasedAutoencoder, test_tasks: List[ARCTask], 
                          device: str = 'cuda', save_images: bool = False,
                          output_dir: str = 'evaluation_images') -> Dict[str, float]:
    """
    Evaluate the enhanced task-based model.
    
    Args:
        model: EnhancedTaskBasedAutoencoder model
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
    task_accuracies = {}
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for task_idx, task in enumerate(test_tasks):
            # Create a single-task batch
            task_batch = EnhancedTaskBatch([task])
            
            # Process the task
            results = model.process_task_batch_enhanced(task_batch, device)
            
            # Compare predictions with ground truth
            test_entries = task_batch.get_testing_entries()
            predictions = results['test_predictions']
            
            task_accuracy = 0.0
            num_test_entries = len(test_entries)
            
            for i, (input_grid, expected_output, _) in enumerate(test_entries):
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
            task_accuracies[f'task_{task_idx}'] = task_accuracy
    
    avg_accuracy = total_accuracy / total_tasks if total_tasks > 0 else 0.0
    prediction_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    results = {
        'accuracy': avg_accuracy,
        'prediction_accuracy': prediction_accuracy,
        'total_tasks': total_tasks,
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        **task_accuracies
    }
    
    # Save evaluation summary if images were saved
    if save_images:
        save_evaluation_summary(results, output_dir)
    
    return results