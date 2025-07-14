# ARC Transformer Implementation

This repository contains a complete transformer architecture implementation specifically designed for solving ARC (Abstraction and Reasoning Corpus) tasks. The transformer model learns to transform input grids to output grids by understanding the underlying patterns and rules.

## Overview

The transformer architecture consists of:

1. **Multi-Head Attention**: The core attention mechanism that allows the model to focus on different parts of the input
2. **Positional Encoding**: Adds position information to the input embeddings
3. **Feed-Forward Networks**: Non-linear transformations applied to each position
4. **Layer Normalization**: Stabilizes training by normalizing activations
5. **Residual Connections**: Helps with gradient flow during training

## Architecture Components

### Core Transformer Classes

- `Transformer`: Complete transformer with encoder and decoder
- `ARCGridTransformer`: Specialized transformer for ARC grid tasks
- `MultiHeadAttention`: Multi-head attention mechanism
- `PositionalEncoding`: Sinusoidal positional encoding
- `FeedForward`: Feed-forward network with ReLU activation

### Data Handling

- `ARCDataset`: PyTorch dataset for loading ARC training data
- `ARCTestDataset`: Dataset for test tasks
- Grid conversion utilities for 2D ↔ 1D transformations

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the ARC dataset in the `data/` directory:
```
data/
├── training/     # Training tasks (400 JSON files)
└── evaluation/   # Evaluation tasks (400 JSON files)
```

## Usage

### Basic Training

Train the transformer model on ARC tasks:

```bash
python train_transformer.py \
    --train_dir data/training \
    --val_dir data/evaluation \
    --batch_size 16 \
    --num_epochs 100 \
    --learning_rate 0.0001 \
    --d_model 256 \
    --num_heads 8 \
    --num_layers 6
```

### Training with Custom Parameters

```bash
python train_transformer.py \
    --train_dir data/training \
    --val_dir data/evaluation \
    --batch_size 32 \
    --num_epochs 200 \
    --learning_rate 0.0003 \
    --d_model 512 \
    --num_heads 16 \
    --num_layers 8 \
    --max_grid_size 30 \
    --checkpoint_dir my_checkpoints \
    --log_dir my_logs
```

### Resume Training

Resume training from a checkpoint:

```bash
python train_transformer.py \
    --train_dir data/training \
    --val_dir data/evaluation \
    --resume checkpoints/best_model.pt \
    --num_epochs 50
```

### Evaluation

Evaluate a trained model:

```bash
python train_transformer.py \
    --test_dir data/evaluation \
    --resume checkpoints/best_model.pt \
    --evaluate
```

### Generate Solutions

Generate solutions for test tasks:

```bash
python train_transformer.py \
    --test_dir data/evaluation \
    --resume checkpoints/best_model.pt \
    --generate
```

## Model Architecture Details

### ARCGridTransformer

The `ARCGridTransformer` is specifically designed for ARC tasks:

- **Input Processing**: Converts 2D grids to 1D sequences
- **Embedding**: Maps grid values (0-9) to high-dimensional vectors
- **Transformer Backbone**: Standard transformer with encoder-decoder
- **Output Generation**: Converts sequences back to 2D grids

### Key Features

1. **Grid-to-Sequence Conversion**: Flattens 2D grids into 1D sequences for transformer processing
2. **Color Embedding**: Embeds the 10 ARC colors (0-9) into continuous vectors
3. **Positional Encoding**: Adds position information to help the model understand spatial relationships
4. **Multi-Head Attention**: Allows the model to attend to different parts of the input grid
5. **Autoregressive Generation**: Generates output grids token by token

## Training Process

### Data Augmentation

The training process includes several data augmentation techniques:

- **Rotation**: Random 90°, 180°, 270° rotations
- **Flipping**: Random horizontal and vertical flips
- **Padding**: Pads grids to maximum size with zeros

### Loss Function

Uses Cross-Entropy Loss with padding token ignored:
- Predicts the next token in the sequence
- Ignores padding tokens (value 0) in loss calculation
- Optimizes for exact grid reconstruction

### Optimization

- **Optimizer**: AdamW with weight decay
- **Learning Rate**: Cosine annealing schedule
- **Gradient Clipping**: Prevents exploding gradients
- **Checkpointing**: Saves best model and regular checkpoints

## Monitoring Training

### Training Progress

Training progress is monitored through console output:
- Training loss per epoch
- Validation loss per epoch
- Validation accuracy per epoch
- Learning rate updates

### Checkpoints

The training script saves:
- `checkpoint_epoch_N.pt`: Regular checkpoints every 10 epochs
- `best_model.pt`: Best model based on validation loss

## Model Performance

### Expected Results

With proper training, the model should achieve:
- Training loss: < 0.5
- Validation loss: < 1.0
- Validation accuracy: > 60%

### Factors Affecting Performance

1. **Model Size**: Larger models (more parameters) generally perform better
2. **Training Data**: More diverse training data improves generalization
3. **Training Time**: Longer training with proper scheduling helps
4. **Data Augmentation**: Augmentation improves robustness

## Advanced Usage

### Custom Model Configuration

```python
from transformer import ARCGridTransformer

# Create custom model
model = ARCGridTransformer(
    grid_size=30,
    num_colors=10,
    d_model=512,      # Model dimension
    num_heads=16,     # Number of attention heads
    d_ff=2048,        # Feed-forward dimension
    num_layers=8,     # Number of transformer layers
    dropout=0.1       # Dropout rate
)
```

### Custom Training Loop

```python
from train_transformer import ARCTrainer
from arc_data_loader import create_arc_dataloaders

# Create data loaders
train_loader, val_loader = create_arc_dataloaders(
    train_dir='data/training',
    val_dir='data/evaluation',
    batch_size=16
)

# Create trainer
trainer = ARCTrainer(model, device='cuda', learning_rate=0.0001)

# Train
trainer.train(train_loader, val_loader, num_epochs=100)
```

### Grid Visualization

```python
from arc_data_loader import visualize_grid, grid_to_string

# Visualize a grid
visualize_grid(grid_tensor, title="ARC Grid")

# Convert to string
grid_str = grid_to_string(grid_tensor)
print(grid_str)
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or model size
2. **Slow Training**: Use GPU acceleration, reduce model complexity
3. **Poor Performance**: Increase training time, adjust learning rate
4. **Data Loading Errors**: Check data directory structure

### Performance Tips

1. **Use GPU**: Training is much faster on GPU
2. **Batch Size**: Larger batch sizes generally train faster
3. **Model Size**: Balance between performance and training time
4. **Data Augmentation**: Helps with generalization

## File Structure

```
├── transformer.py          # Core transformer implementation
├── arc_data_loader.py      # Data loading and preprocessing
├── train_transformer.py    # Training script
├── requirements.txt        # Dependencies
├── README_TRANSFORMER.md   # This file
├── checkpoints/           # Saved model checkpoints
└── solutions/             # Generated solutions
```

## Contributing

To improve the transformer implementation:

1. Experiment with different architectures
2. Try different attention mechanisms
3. Implement additional data augmentation
4. Optimize for specific ARC task types
5. Add ensemble methods

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [ARC: Abstraction and Reasoning Corpus](https://arxiv.org/abs/1911.01547) - ARC benchmark paper
- [PyTorch Documentation](https://pytorch.org/docs/) - PyTorch framework