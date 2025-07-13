# Task-Based Autoencoder System

This system implements a sophisticated task-based autoencoder that processes batches of tasks containing training and testing entries. The key innovation is that it averages decoder outputs across elements of the same task and uses this averaged representation for autoregressive generation of test outputs.

## Overview

The task-based autoencoder system consists of several key components:

1. **TaskBatch**: Represents a batch of tasks, where each task contains multiple training and testing entries
2. **EnhancedTaskBasedAutoencoder**: Extends the base autoencoder with task-aware processing
3. **Task-Based Data Loading**: Specialized data loaders for handling task-based batches
4. **Autoregressive Generation**: Sophisticated generation using averaged task representations

## Key Features

### 1. Task-Based Batch Processing
- Each batch contains multiple tasks
- Each task has several training entries and one testing entry
- Tasks are processed together to learn task-specific patterns

### 2. Averaging Across Task Elements
- Decoder outputs are averaged across all training entries of the same task
- This creates a task-specific representation that captures the underlying pattern
- The averaged representation is used for generating test outputs

### 3. Autoregressive Generation
- Test outputs are generated autoregressively using the averaged task representation
- Input test grids serve as context for the generation process
- The system combines task patterns with input-specific modifications

### 4. Enhanced Data Organization
- Sophisticated mapping between entries and their corresponding tasks
- Support for different types of ARC patterns (copy, rotation, color shift, mirror)
- Flexible data loading from JSON files or programmatically created tasks

## Architecture

### Core Components

#### EnhancedTaskBatch
```python
class EnhancedTaskBatch:
    def __init__(self, tasks: List[ARCTask], max_grid_size: int = 30):
        # Creates a batch of tasks with proper organization
        # Maps entries to their corresponding tasks
        # Handles training and testing entry separation
```

#### EnhancedTaskBasedAutoencoder
```python
class EnhancedTaskBasedAutoencoder(ARCAutoencoder):
    def process_task_batch_enhanced(self, task_batch, device):
        # 1. Process training entries and compute task representations
        # 2. Average latent representations across task elements
        # 3. Generate test outputs autoregressively
        # 4. Return task representations and predictions
```

### Processing Pipeline

1. **Task Representation Computation**:
   - Encode each training entry to get latent representations
   - Average latent representations across entries of the same task
   - Store task-specific representations

2. **Autoregressive Generation**:
   - Use averaged task representations as context
   - Generate test outputs step by step
   - Combine task patterns with input-specific modifications

3. **Training Process**:
   - Train on task batches with task-aware loss computation
   - Track per-task performance metrics
   - Support for validation and checkpointing

## Usage Examples

### Basic Usage

```python
from enhanced_task_autoencoder import (
    EnhancedTaskBasedAutoencoder,
    EnhancedTaskBatch,
    create_enhanced_task_dataloader
)

# Create model
model = EnhancedTaskBasedAutoencoder(
    grid_size=10,
    num_colors=10,
    d_model=128,
    num_heads=4,
    d_ff=512,
    num_layers=2,
    latent_dim=256
)

# Create task batch
tasks = create_sample_tasks()  # Your task creation function
task_batch = EnhancedTaskBatch(tasks, max_grid_size=10)

# Process task batch
results = model.process_task_batch_enhanced(task_batch, device='cuda')

# Access results
task_representations = results['task_representations']
test_predictions = results['test_predictions']
```

### Training

```python
from train_task_based_autoencoder import TaskBasedTrainer

# Create trainer
trainer = TaskBasedTrainer(
    model=model,
    device='cuda',
    learning_rate=0.0001,
    weight_decay=1e-5
)

# Create data loaders
train_loader = EnhancedTaskBasedDataLoader(train_tasks, batch_size=4)
val_loader = EnhancedTaskBasedDataLoader(val_tasks, batch_size=4)

# Train the model
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    checkpoint_dir='checkpoints',
    log_dir='logs'
)
```

### Command Line Training

```bash
# Train with sample data
python train_task_based_autoencoder.py \
    --batch_size 4 \
    --num_epochs 50 \
    --learning_rate 0.0001 \
    --d_model 128 \
    --num_heads 4 \
    --latent_dim 256

# Train with custom data directory
python train_task_based_autoencoder.py \
    --data_dir /path/to/arc/tasks \
    --batch_size 4 \
    --num_epochs 100 \
    --max_grid_size 30
```

## Data Format

### Task Structure
Each task should follow this JSON structure:

```json
{
    "train": [
        {
            "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "output": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        },
        {
            "input": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            "output": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        }
    ],
    "test": [
        {
            "input": [[2, 3, 4], [5, 6, 7], [8, 9, 10]],
            "output": [[2, 3, 4], [5, 6, 7], [8, 9, 10]]
        }
    ]
}
```

### Supported Pattern Types

1. **Copy Pattern**: Output is identical to input
2. **Rotation Pattern**: Output is rotated version of input
3. **Color Shift Pattern**: Output has shifted color values
4. **Mirror Pattern**: Output is mirrored version of input

## Model Configuration

### Key Parameters

- `grid_size`: Maximum grid size (default: 30)
- `num_colors`: Number of colors in the grid (default: 10)
- `d_model`: Model dimension (default: 512)
- `num_heads`: Number of attention heads (default: 8)
- `d_ff`: Feed-forward dimension (default: 2048)
- `num_layers`: Number of transformer layers (default: 6)
- `latent_dim`: Latent space dimension (default: 1024)
- `dropout`: Dropout rate (default: 0.1)

### Recommended Configurations

#### Small Model (for testing)
```python
model = EnhancedTaskBasedAutoencoder(
    grid_size=10,
    d_model=64,
    num_heads=4,
    d_ff=256,
    num_layers=2,
    latent_dim=128
)
```

#### Medium Model (for development)
```python
model = EnhancedTaskBasedAutoencoder(
    grid_size=20,
    d_model=128,
    num_heads=4,
    d_ff=512,
    num_layers=4,
    latent_dim=256
)
```

#### Large Model (for production)
```python
model = EnhancedTaskBasedAutoencoder(
    grid_size=30,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    latent_dim=1024
)
```

## Evaluation

### Metrics

The system tracks several metrics:

1. **Overall Loss**: Average loss across all training entries
2. **Per-Task Loss**: Individual loss for each task
3. **Accuracy**: Exact match accuracy for test predictions
4. **Task-Specific Accuracy**: Accuracy per individual task

### Evaluation Script

```python
from enhanced_task_autoencoder import evaluate_enhanced_model

# Evaluate model
metrics = evaluate_enhanced_model(model, test_tasks, device='cuda')

print(f"Overall accuracy: {metrics['accuracy']:.4f}")
print(f"Total tasks: {metrics['total_tasks']}")

# Per-task accuracies
for key, value in metrics.items():
    if key.startswith('task_'):
        print(f"{key}: {value:.4f}")
```

## Advanced Features

### Custom Autoregressive Generation

You can customize the autoregressive generation process:

```python
def custom_autoregressive_generation(self, task_latent, task_output, input_grid, device):
    # Your custom generation logic here
    # Combine task patterns with input context
    # Apply specific transformations
    return generated_output
```

### Task-Specific Processing

The system supports task-specific processing:

```python
# Process individual tasks
for task in tasks:
    task_batch = EnhancedTaskBatch([task])
    results = model.process_task_batch_enhanced(task_batch, device)
    
    # Access task-specific results
    task_representations = results['task_representations']
    predictions = results['test_predictions']
```

## Performance Considerations

### Memory Usage

- Large grid sizes require significant memory
- Consider using smaller models for initial testing
- Use gradient checkpointing for very large models

### Training Time

- Task-based processing is more computationally intensive
- Consider using smaller batch sizes initially
- Use mixed precision training for faster training

### Optimization Tips

1. Start with small grid sizes and simple patterns
2. Gradually increase model complexity
3. Use learning rate scheduling
4. Monitor per-task performance
5. Use early stopping based on validation loss

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or grid size
2. **Poor Convergence**: Check learning rate and model size
3. **Incorrect Predictions**: Verify task data format
4. **Slow Training**: Use smaller models or fewer layers

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed processing information
```

## Future Enhancements

### Planned Features

1. **Multi-task Learning**: Learn across different types of patterns simultaneously
2. **Attention Visualization**: Visualize attention patterns for interpretability
3. **Dynamic Grid Sizing**: Support for variable grid sizes
4. **Advanced Patterns**: Support for more complex ARC patterns
5. **Ensemble Methods**: Combine multiple models for better performance

### Research Directions

1. **Meta-Learning**: Learn to learn new patterns quickly
2. **Few-Shot Learning**: Generalize from very few examples
3. **Transfer Learning**: Transfer knowledge between different pattern types
4. **Interpretability**: Understand how the model learns patterns

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@misc{task_based_autoencoder_2024,
  title={Task-Based Autoencoder for ARC Pattern Learning},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo/task-based-autoencoder}}
}
```