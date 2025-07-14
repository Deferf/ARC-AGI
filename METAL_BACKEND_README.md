# Metal Backend Support and Evaluation Image Generation

This project now supports Apple's Metal Performance Shaders (MPS) backend for GPU acceleration on macOS devices, alongside CUDA support for NVIDIA GPUs. Additionally, the evaluation process can now generate visual outputs showing input/output/prediction pairs.

## Features

### 1. Metal Backend Support

The project automatically detects and uses the best available backend:
- **CUDA**: For NVIDIA GPUs (Linux/Windows)
- **MPS**: For Apple Silicon and AMD GPUs on macOS (Metal backend)
- **CPU**: Fallback when no GPU is available

#### Device Selection

You can specify the device using the `--device` flag:

```bash
# Automatic selection (recommended)
python train_autoencoder.py --device auto

# Force specific backend
python train_autoencoder.py --device cuda
python train_autoencoder.py --device mps
python train_autoencoder.py --device cpu
```

### 2. Evaluation Image Generation

During evaluation, the system can generate visualization images showing:
- Input grids
- Expected output grids
- Model predictions
- Accuracy indicators

#### Enabling Image Generation

Use the `--save_eval_images` flag during evaluation:

```bash
# Evaluate with image generation
python train_autoencoder.py --evaluate --checkpoint model.pt --save_eval_images

# Specify output directory
python train_autoencoder.py --evaluate --checkpoint model.pt --save_eval_images --eval_output_dir my_results
```

## Usage Examples

### 1. Training with Metal Backend

```bash
# Train on Apple Silicon Mac
python train_task_based_autoencoder.py --device auto --num_epochs 50

# The system will automatically detect and use MPS
```

### 2. Evaluation with Image Generation

```bash
# Task-based autoencoder evaluation
python train_task_based_autoencoder.py \
    --evaluate \
    --checkpoint checkpoints/best_model.pt \
    --save_eval_images \
    --eval_output_dir evaluation_results

# Standard autoencoder evaluation
python train_autoencoder.py \
    --evaluate \
    --checkpoint checkpoints/model.pt \
    --save_eval_images \
    --eval_output_dir eval_images
```

### 3. Testing Metal Backend Support

Run the test script to verify Metal backend detection and image generation:

```bash
python test_metal_backend.py
```

This will:
- Detect available backends
- Create a test model on the appropriate device
- Generate sample evaluation images

## Output Structure

When `--save_eval_images` is enabled, the following structure is created:

```
evaluation_images/
├── task_001/
│   ├── task_001_pair_0.png
│   ├── task_001_pair_1.png
│   └── ...
├── task_002/
│   └── ...
└── evaluation_summary_YYYYMMDD_HHMMSS.txt
```

Each image shows:
- **Left panel**: Input grid
- **Middle panel**: Expected output
- **Right panel**: Model prediction
- **Title**: Indicates if the prediction is correct

## API Usage

### Using the Evaluation Utils

```python
from evaluation_utils import get_device, save_input_output_pair

# Get appropriate device
device = get_device('auto')  # Returns 'cuda', 'mps', or 'cpu'

# Save visualization
save_input_output_pair(
    input_grid=input_tensor,
    output_grid=target_tensor,
    predicted_grid=prediction_tensor,
    task_id="example_task",
    pair_idx=0,
    output_dir="results"
)
```

### Modified Evaluation Functions

The evaluation functions now support image generation:

```python
# Task-based evaluation
results = evaluate_task_based_model(
    model=model,
    test_tasks=tasks,
    device=device,
    save_images=True,
    output_dir='evaluation_images'
)

# Standard autoencoder evaluation
results = evaluate_autoencoder(
    model=model,
    test_loader=loader,
    device=device,
    save_images=True,
    output_dir='evaluation_images',
    max_images=50  # Limit number of saved images
)
```

## Performance Considerations

### Metal Backend Performance

- Metal backend (MPS) provides significant speedup on Apple Silicon Macs
- Performance is generally comparable to CUDA for similar hardware tiers
- Some operations may have different optimization characteristics

### Image Generation Impact

- Saving images adds overhead to evaluation
- Use `max_images` parameter to limit the number of saved images
- Images are saved asynchronously when possible

## Troubleshooting

### Metal Backend Issues

If Metal backend is not detected:
1. Ensure you have macOS 12.3+ 
2. Update PyTorch: `pip install torch>=2.0.0`
3. Check availability: `python -c "import torch; print(torch.backends.mps.is_available())"`

### Image Generation Issues

If images are not generated:
1. Check write permissions for output directory
2. Ensure matplotlib is installed: `pip install matplotlib`
3. Verify PIL/Pillow for GIF generation: `pip install Pillow`

## Requirements

The following packages are required for full functionality:
- `torch>=2.0.0` (for MPS support)
- `matplotlib>=3.5.0` (for image generation)
- `Pillow>=8.3.0` (optional, for GIF creation)

Install all requirements:
```bash
pip install -r requirements.txt
```