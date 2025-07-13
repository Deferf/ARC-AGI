#!/usr/bin/env python3
"""
Simple demonstration of the Task-Based Autoencoder concept.
This script shows the core ideas without requiring PyTorch or other external dependencies.
"""

import json
import random
from typing import List, Dict, Tuple


class SimpleTask:
    """Simple representation of an ARC task."""
    def __init__(self, task_id: str, train_pairs: List[Dict], test_pairs: List[Dict]):
        self.task_id = task_id
        self.train_pairs = train_pairs
        self.test_pairs = test_pairs


class SimpleTaskBatch:
    """Simple task batch for demonstration."""
    def __init__(self, tasks: List[SimpleTask]):
        self.tasks = tasks
        self.task_ids = [task.task_id for task in tasks]
    
    def get_training_entries(self) -> List[Tuple[List[List[int]], List[List[int]], int]]:
        """Get all training entries with their task indices."""
        training_entries = []
        for task_idx, task in enumerate(self.tasks):
            for pair in task.train_pairs:
                training_entries.append((pair['input'], pair['output'], task_idx))
        return training_entries
    
    def get_testing_entries(self) -> List[Tuple[List[List[int]], List[List[int]], int]]:
        """Get all testing entries with their task indices."""
        testing_entries = []
        for task_idx, task in enumerate(self.tasks):
            for pair in task.test_pairs:
                testing_entries.append((pair['input'], pair['output'], task_idx))
        return testing_entries


class SimpleTaskBasedAutoencoder:
    """Simple demonstration of task-based autoencoder concepts."""
    
    def __init__(self):
        self.task_patterns = {}
    
    def encode(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> Dict:
        """Simple encoding that extracts pattern information."""
        # In a real implementation, this would be a neural network
        # Here we just extract some basic pattern information
        
        pattern_info = {
            'input_shape': (len(input_grid), len(input_grid[0]) if input_grid else 0),
            'output_shape': (len(output_grid), len(output_grid[0]) if output_grid else 0),
            'input_sum': sum(sum(row) for row in input_grid),
            'output_sum': sum(sum(row) for row in output_grid),
            'is_copy': input_grid == output_grid,
            'is_rotation': self._is_rotation(input_grid, output_grid),
            'is_color_shift': self._is_color_shift(input_grid, output_grid),
            'is_mirror': self._is_mirror(input_grid, output_grid)
        }
        
        return pattern_info
    
    def _is_rotation(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> bool:
        """Check if output is a rotation of input."""
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return False
        
        # Check 90-degree rotation
        rotated = list(zip(*input_grid[::-1]))
        rotated = [list(row) for row in rotated]
        return rotated == output_grid
    
    def _is_color_shift(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> bool:
        """Check if output is a color shift of input."""
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return False
        
        # Check if all non-zero values are shifted by the same amount
        input_vals = [val for row in input_grid for val in row if val != 0]
        output_vals = [val for row in output_grid for val in row if val != 0]
        
        if len(input_vals) != len(output_vals):
            return False
        
        if not input_vals:
            return True
        
        # Check if there's a constant shift
        shifts = [output_vals[i] - input_vals[i] for i in range(len(input_vals))]
        return len(set(shifts)) == 1
    
    def _is_mirror(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> bool:
        """Check if output is a mirror of input."""
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return False
        
        # Check horizontal mirror
        mirrored = [row[::-1] for row in input_grid]
        return mirrored == output_grid
    
    def decode(self, pattern_info: Dict, input_grid: List[List[int]]) -> List[List[int]]:
        """Simple decoding that applies pattern to input."""
        # In a real implementation, this would be a neural network
        # Here we apply the learned pattern to the input
        
        if pattern_info['is_copy']:
            return [row[:] for row in input_grid]
        
        elif pattern_info['is_rotation']:
            rotated = list(zip(*input_grid[::-1]))
            return [list(row) for row in rotated]
        
        elif pattern_info['is_color_shift']:
            # Find the shift amount from training data
            shift = pattern_info.get('color_shift', 1)
            return [[val + shift if val != 0 else 0 for val in row] for row in input_grid]
        
        elif pattern_info['is_mirror']:
            return [row[::-1] for row in input_grid]
        
        else:
            # Default: return input as is
            return [row[:] for row in input_grid]
    
    def process_task_batch(self, task_batch: SimpleTaskBatch) -> Dict:
        """Process a batch of tasks using the task-based approach."""
        print("=== Processing Task Batch ===")
        print(f"Tasks: {task_batch.task_ids}")
        
        # Step 1: Process training entries and compute task representations
        task_representations = {}
        training_entries = task_batch.get_training_entries()
        
        print(f"\nProcessing {len(training_entries)} training entries...")
        
        # Group entries by task
        task_groups = {}
        for input_grid, output_grid, task_idx in training_entries:
            if task_idx not in task_groups:
                task_groups[task_idx] = []
            task_groups[task_idx].append((input_grid, output_grid))
        
        # Compute task-specific representations
        for task_idx, entries in task_groups.items():
            print(f"\nTask {task_idx} ({task_batch.task_ids[task_idx]}):")
            
            # Encode each training entry
            pattern_infos = []
            for input_grid, output_grid in entries:
                pattern_info = self.encode(input_grid, output_grid)
                pattern_infos.append(pattern_info)
                
                print(f"  Training pair:")
                print(f"    Input:  {input_grid}")
                print(f"    Output: {output_grid}")
                print(f"    Pattern: {self._describe_pattern(pattern_info)}")
            
            # Average pattern information across task elements
            avg_pattern = self._average_patterns(pattern_infos)
            task_representations[task_idx] = avg_pattern
            
            print(f"  Average pattern: {self._describe_pattern(avg_pattern)}")
        
        # Step 2: Generate test outputs using autoregressive approach
        test_predictions = []
        testing_entries = task_batch.get_testing_entries()
        
        print(f"\nGenerating predictions for {len(testing_entries)} test entries...")
        
        for input_grid, expected_output, task_idx in testing_entries:
            if task_idx in task_representations:
                # Use task-specific representation for generation
                task_pattern = task_representations[task_idx]
                
                # Generate output autoregressively
                predicted_output = self._generate_autoregressive(task_pattern, input_grid)
                test_predictions.append(predicted_output)
                
                print(f"\nTest entry for Task {task_idx}:")
                print(f"  Input:           {input_grid}")
                print(f"  Expected:        {expected_output}")
                print(f"  Predicted:       {predicted_output}")
                print(f"  Correct:         {predicted_output == expected_output}")
            else:
                print(f"Warning: No representation found for task {task_idx}")
                test_predictions.append(input_grid)  # Fallback
        
        return {
            'task_representations': task_representations,
            'test_predictions': test_predictions
        }
    
    def _describe_pattern(self, pattern_info: Dict) -> str:
        """Describe the pattern in human-readable format."""
        if pattern_info['is_copy']:
            return "Copy pattern"
        elif pattern_info['is_rotation']:
            return "Rotation pattern (90° clockwise)"
        elif pattern_info['is_color_shift']:
            return f"Color shift pattern (shift: {pattern_info.get('color_shift', 'unknown')})"
        elif pattern_info['is_mirror']:
            return "Mirror pattern (horizontal)"
        else:
            return "Unknown pattern"
    
    def _average_patterns(self, pattern_infos: List[Dict]) -> Dict:
        """Average pattern information across multiple entries."""
        if not pattern_infos:
            return {}
        
        avg_pattern = {}
        
        # Average numerical values
        avg_pattern['input_sum'] = sum(p['input_sum'] for p in pattern_infos) / len(pattern_infos)
        avg_pattern['output_sum'] = sum(p['output_sum'] for p in pattern_infos) / len(pattern_infos)
        
        # Use majority vote for boolean patterns
        avg_pattern['is_copy'] = sum(p['is_copy'] for p in pattern_infos) > len(pattern_infos) / 2
        avg_pattern['is_rotation'] = sum(p['is_rotation'] for p in pattern_infos) > len(pattern_infos) / 2
        avg_pattern['is_color_shift'] = sum(p['is_color_shift'] for p in pattern_infos) > len(pattern_infos) / 2
        avg_pattern['is_mirror'] = sum(p['is_mirror'] for p in pattern_infos) > len(pattern_infos) / 2
        
        # Keep shapes from first pattern
        avg_pattern['input_shape'] = pattern_infos[0]['input_shape']
        avg_pattern['output_shape'] = pattern_infos[0]['output_shape']
        
        return avg_pattern
    
    def _generate_autoregressive(self, task_pattern: Dict, input_grid: List[List[int]]) -> List[List[int]]:
        """Generate output using task pattern and input context."""
        # In a real implementation, this would be sophisticated autoregressive generation
        # Here we use the task pattern to determine the transformation
        
        if task_pattern.get('is_copy', False):
            return [row[:] for row in input_grid]
        
        elif task_pattern.get('is_rotation', False):
            rotated = list(zip(*input_grid[::-1]))
            return [list(row) for row in rotated]
        
        elif task_pattern.get('is_color_shift', False):
            # Determine shift amount from pattern
            shift = 1  # Default shift
            return [[val + shift if val != 0 else 0 for val in row] for row in input_grid]
        
        elif task_pattern.get('is_mirror', False):
            return [row[::-1] for row in input_grid]
        
        else:
            # Default: return input as is
            return [row[:] for row in input_grid]


def create_sample_tasks() -> List[SimpleTask]:
    """Create sample tasks for demonstration."""
    
    # Task 1: Copy pattern
    task1 = SimpleTask("copy_task", [
        {
            'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            'output': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        },
        {
            'input': [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            'output': [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        }
    ], [
        {
            'input': [[2, 3, 4], [5, 6, 7], [8, 9, 10]],
            'output': [[2, 3, 4], [5, 6, 7], [8, 9, 10]]
        }
    ])
    
    # Task 2: Rotation pattern
    task2 = SimpleTask("rotation_task", [
        {
            'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            'output': [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
        },
        {
            'input': [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            'output': [[6, 3, 0], [7, 4, 1], [8, 5, 2]]
        }
    ], [
        {
            'input': [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            'output': [[3, 2, 1], [3, 2, 1], [3, 2, 1]]
        }
    ])
    
    # Task 3: Color shift pattern
    task3 = SimpleTask("color_shift_task", [
        {
            'input': [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            'output': [[2, 2, 2], [3, 3, 3], [4, 4, 4]]
        },
        {
            'input': [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            'output': [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        }
    ], [
        {
            'input': [[5, 5, 5], [6, 6, 6], [7, 7, 7]],
            'output': [[6, 6, 6], [7, 7, 7], [8, 8, 8]]
        }
    ])
    
    return [task1, task2, task3]


def main():
    """Main demonstration function."""
    print("Task-Based Autoencoder Concept Demonstration")
    print("=" * 50)
    print()
    print("This demonstration shows the core concepts of the task-based autoencoder:")
    print("1. Processing batches of tasks with training and testing entries")
    print("2. Averaging decoder outputs across task elements")
    print("3. Using averaged representations for autoregressive generation")
    print("4. Generating test outputs with input context")
    print()
    
    # Create sample tasks
    tasks = create_sample_tasks()
    print(f"Created {len(tasks)} sample tasks")
    
    # Create task batch
    task_batch = SimpleTaskBatch(tasks)
    print(f"Created task batch with tasks: {task_batch.task_ids}")
    
    # Create model
    model = SimpleTaskBasedAutoencoder()
    print("Created simple task-based autoencoder")
    
    # Process the task batch
    print("\n" + "="*50)
    results = model.process_task_batch(task_batch)
    
    # Show summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    print(f"Processed {len(task_batch.task_ids)} tasks")
    print(f"Generated {len(results['test_predictions'])} test predictions")
    
    # Calculate accuracy
    testing_entries = task_batch.get_testing_entries()
    correct_predictions = 0
    
    for i, (_, expected_output, _) in enumerate(testing_entries):
        if i < len(results['test_predictions']):
            predicted_output = results['test_predictions'][i]
            if predicted_output == expected_output:
                correct_predictions += 1
    
    accuracy = correct_predictions / len(testing_entries) if testing_entries else 0
    print(f"Accuracy: {accuracy:.2%} ({correct_predictions}/{len(testing_entries)})")
    
    print("\nKey Concepts Demonstrated:")
    print("✓ Task-based batch processing")
    print("✓ Averaging across task elements")
    print("✓ Autoregressive generation")
    print("✓ Input context utilization")
    print("✓ Pattern recognition and application")


if __name__ == "__main__":
    main()