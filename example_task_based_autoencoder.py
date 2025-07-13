#!/usr/bin/env python3
"""
Example usage of the Task-Based Autoencoder system.
This script demonstrates how to use the enhanced task-based autoencoder for processing
batches of tasks with training and testing entries, averaging decoder outputs across
task elements, and generating test outputs autoregressively.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from typing import List, Dict, Tuple
from tqdm import tqdm

from enhanced_task_autoencoder import (
    EnhancedTaskBasedAutoencoder, 
    EnhancedTaskBatch, 
    EnhancedTaskBasedDataLoader,
    create_enhanced_task_dataloader,
    evaluate_enhanced_model
)
from arc_data_loader import ARCTask, visualize_grid, grid_to_string


def create_sample_tasks() -> List[ARCTask]:
    """
    Create sample ARC tasks for demonstration.
    
    Returns:
        List of sample ARCTask objects
    """
    # Task 1: Simple copy pattern
    task1_data = {
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
    
    # Task 2: Rotation pattern
    task2_data = {
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
    
    # Task 3: Color shift pattern
    task3_data = {
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
    
    tasks = [
        ARCTask.from_json("task_1", task1_data),
        ARCTask.from_json("task_2", task2_data),
        ARCTask.from_json("task_3", task3_data)
    ]
    
    return tasks


def demonstrate_task_batch_processing():
    """Demonstrate basic task batch processing."""
    print("=== Task Batch Processing Demo ===")
    
    # Create sample tasks
    tasks = create_sample_tasks()
    
    # Create a task batch
    task_batch = EnhancedTaskBatch(tasks, max_grid_size=10)
    
    print(f"Created task batch with {len(tasks)} tasks:")
    for i, task_id in enumerate(task_batch.task_ids):
        print(f"  Task {i}: {task_id}")
    
    # Get training and testing entries
    training_entries = task_batch.get_training_entries()
    testing_entries = task_batch.get_testing_entries()
    
    print(f"\nTraining entries: {len(training_entries)}")
    print(f"Testing entries: {len(testing_entries)}")
    
    # Show some examples
    print("\nSample training entry:")
    input_grid, output_grid, task_idx = training_entries[0]
    print(f"Task index: {task_idx}")
    print("Input grid:")
    print(grid_to_string(input_grid))
    print("Output grid:")
    print(grid_to_string(output_grid))
    
    print("\nTask batch processing demo completed!\n")


def demonstrate_model_creation_and_processing():
    """Demonstrate model creation and task processing."""
    print("=== Model Creation and Processing Demo ===")
    
    # Create model
    model = EnhancedTaskBasedAutoencoder(
        grid_size=10,
        num_colors=10,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=2,
        latent_dim=256,
        dropout=0.1
    )
    
    print(f"Created enhanced task-based autoencoder with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create sample tasks
    tasks = create_sample_tasks()
    task_batch = EnhancedTaskBatch(tasks, max_grid_size=10)
    
    # Process the task batch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        results = model.process_task_batch_enhanced(task_batch, device)
        
        print(f"Task representations computed for {len(results['task_representations']['latents'])} tasks")
        print(f"Generated {len(results['test_predictions'])} test predictions")
        
        # Show some predictions
        print("\nSample test prediction:")
        predicted_output = results['test_predictions'][0]
        print(grid_to_string(predicted_output))
        
    except Exception as e:
        print(f"Error during processing: {e}")
        print("This might be due to model initialization issues or device compatibility")
    
    print("\nModel creation and processing demo completed!\n")


def demonstrate_training():
    """Demonstrate training on task batches."""
    print("=== Training Demo ===")
    
    # Create model
    model = EnhancedTaskBasedAutoencoder(
        grid_size=10,
        num_colors=10,
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        latent_dim=128,
        dropout=0.1
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create sample tasks
    tasks = create_sample_tasks()
    task_batch = EnhancedTaskBatch(tasks, max_grid_size=10)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"Training on {len(tasks)} tasks...")
    
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        
        # Train on the task batch
        metrics = model.train_on_task_batch_enhanced(task_batch, device)
        
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Entries processed: {metrics['num_entries']}")
        
        # Show per-task losses
        for key, value in metrics.items():
            if key.startswith('task_') and key.endswith('_loss'):
                print(f"  {key}: {value:.4f}")
    
    print("\nTraining demo completed!\n")


def demonstrate_evaluation():
    """Demonstrate model evaluation."""
    print("=== Evaluation Demo ===")
    
    # Create model
    model = EnhancedTaskBasedAutoencoder(
        grid_size=10,
        num_colors=10,
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        latent_dim=128,
        dropout=0.1
    )
    
    # Create sample tasks for evaluation
    test_tasks = create_sample_tasks()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"Evaluating model on {len(test_tasks)} test tasks...")
    
    # Evaluate the model
    metrics = evaluate_enhanced_model(model, test_tasks, device)
    
    print("Evaluation results:")
    print(f"  Overall accuracy: {metrics['accuracy']:.4f}")
    print(f"  Total tasks: {metrics['total_tasks']}")
    
    # Show per-task accuracies
    for key, value in metrics.items():
        if key.startswith('task_') and key != 'total_tasks':
            print(f"  {key}: {value:.4f}")
    
    print("\nEvaluation demo completed!\n")


def demonstrate_data_loader():
    """Demonstrate the enhanced data loader."""
    print("=== Data Loader Demo ===")
    
    # Create sample tasks
    tasks = create_sample_tasks()
    
    # Create data loader
    data_loader = EnhancedTaskBasedDataLoader(tasks, batch_size=2, max_grid_size=10)
    
    print(f"Created data loader with {len(data_loader)} batches")
    print(f"Batch size: {data_loader.batch_size}")
    print(f"Max grid size: {data_loader.max_grid_size}")
    
    # Iterate through batches
    for batch_idx, task_batch in enumerate(data_loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Tasks: {task_batch.task_ids}")
        print(f"  Training entries: {len(task_batch.get_training_entries())}")
        print(f"  Testing entries: {len(task_batch.get_testing_entries())}")
    
    print("\nData loader demo completed!\n")


def demonstrate_autoregressive_generation():
    """Demonstrate autoregressive generation capabilities."""
    print("=== Autoregressive Generation Demo ===")
    
    # Create model
    model = EnhancedTaskBasedAutoencoder(
        grid_size=10,
        num_colors=10,
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        latent_dim=128,
        dropout=0.1
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Create a simple task for demonstration
    task_data = {
        'train': [
            {
                'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                'output': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            }
        ],
        'test': [
            {
                'input': [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                'output': [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
            }
        ]
    }
    
    task = ARCTask.from_json("demo_task", task_data)
    task_batch = EnhancedTaskBatch([task], max_grid_size=10)
    
    print("Processing task with autoregressive generation...")
    
    try:
        # Process the task
        results = model.process_task_batch_enhanced(task_batch, device)
        
        # Show results
        print("Task processing completed!")
        print(f"Generated {len(results['test_predictions'])} predictions")
        
        # Show the prediction
        if results['test_predictions']:
            predicted_output = results['test_predictions'][0]
            print("\nPredicted output:")
            print(grid_to_string(predicted_output))
            
            # Compare with expected
            test_entries = task_batch.get_testing_entries()
            if test_entries:
                _, expected_output, _ = test_entries[0]
                print("\nExpected output:")
                print(grid_to_string(expected_output))
                
                # Check if prediction matches
                is_correct = torch.equal(predicted_output, expected_output)
                print(f"\nPrediction correct: {is_correct}")
        
    except Exception as e:
        print(f"Error during autoregressive generation: {e}")
    
    print("\nAutoregressive generation demo completed!\n")


def main():
    """Run all demonstrations."""
    print("Task-Based Autoencoder System Demonstration")
    print("=" * 50)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")
    
    print()
    
    # Run demonstrations
    demonstrate_task_batch_processing()
    demonstrate_model_creation_and_processing()
    demonstrate_training()
    demonstrate_evaluation()
    demonstrate_data_loader()
    demonstrate_autoregressive_generation()
    
    print("All demonstrations completed!")
    print("\nKey Features Demonstrated:")
    print("1. Task-based batch processing")
    print("2. Averaging decoder outputs across task elements")
    print("3. Autoregressive generation of test outputs")
    print("4. Enhanced data loading and organization")
    print("5. Task-aware training and evaluation")


if __name__ == "__main__":
    main()