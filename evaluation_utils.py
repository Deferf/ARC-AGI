import torch
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, List, Tuple, Dict
import numpy as np
from datetime import datetime


def get_device(device_preference: str = 'auto') -> str:
    """
    Get the appropriate device with Metal backend support.
    
    Args:
        device_preference: 'auto', 'cuda', 'mps', or 'cpu'
        
    Returns:
        String representing the device to use
    """
    if device_preference == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    elif device_preference == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    elif device_preference == 'mps' and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def save_grid_image(grid: torch.Tensor, filename: str, title: str = "", 
                   output_dir: str = "evaluation_images") -> None:
    """
    Save a grid as an image file.
    
    Args:
        grid: Grid tensor [height, width] or [batch_size, height, width]
        filename: Output filename (without path)
        title: Title for the plot
        output_dir: Directory to save images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle different grid dimensions
    if grid.dim() == 3:
        grid = grid[0]  # Take first batch
    
    # Move to CPU for plotting
    if grid.is_cuda or grid.device.type == 'mps':
        grid = grid.cpu()
    
    # Create color map for ARC colors (0-9)
    colors = ['black', 'blue', 'red', 'green', 'yellow', 
              'grey', 'magenta', 'orange', 'lightblue', 'brown']
    cmap = mcolors.ListedColormap(colors)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the grid
    im = ax.imshow(grid.numpy(), cmap=cmap, vmin=0, vmax=9)
    
    # Add title
    ax.set_title(title, fontsize=14, pad=10)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    
    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=range(10))
    cbar.ax.set_ylabel('Color Index', rotation=270, labelpad=15)
    
    # Save figure
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_input_output_pair(input_grid: torch.Tensor, output_grid: torch.Tensor,
                          predicted_grid: Optional[torch.Tensor] = None,
                          task_id: str = "", pair_idx: int = 0,
                          output_dir: str = "evaluation_images") -> None:
    """
    Save input/output pair (and optionally prediction) as a combined image.
    
    Args:
        input_grid: Input grid tensor
        output_grid: Target output grid tensor
        predicted_grid: Predicted output grid tensor (optional)
        task_id: Task identifier
        pair_idx: Index of the pair
        output_dir: Directory to save images
    """
    # Create output directory if it doesn't exist
    task_dir = os.path.join(output_dir, task_id) if task_id else output_dir
    os.makedirs(task_dir, exist_ok=True)
    
    # Move tensors to CPU
    if input_grid.is_cuda or input_grid.device.type == 'mps':
        input_grid = input_grid.cpu()
    if output_grid.is_cuda or output_grid.device.type == 'mps':
        output_grid = output_grid.cpu()
    if predicted_grid is not None and (predicted_grid.is_cuda or predicted_grid.device.type == 'mps'):
        predicted_grid = predicted_grid.cpu()
    
    # Handle different grid dimensions
    if input_grid.dim() == 3:
        input_grid = input_grid[0]
    if output_grid.dim() == 3:
        output_grid = output_grid[0]
    if predicted_grid is not None and predicted_grid.dim() == 3:
        predicted_grid = predicted_grid[0]
    
    # Create color map
    colors = ['black', 'blue', 'red', 'green', 'yellow', 
              'grey', 'magenta', 'orange', 'lightblue', 'brown']
    cmap = mcolors.ListedColormap(colors)
    
    # Create figure with subplots
    num_plots = 3 if predicted_grid is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 6))
    
    if num_plots == 2:
        axes = [axes[0], axes[1]]
    
    # Plot input
    im1 = axes[0].imshow(input_grid.numpy(), cmap=cmap, vmin=0, vmax=9)
    axes[0].set_title('Input', fontsize=14)
    add_grid_lines(axes[0], input_grid.shape)
    
    # Plot expected output
    im2 = axes[1].imshow(output_grid.numpy(), cmap=cmap, vmin=0, vmax=9)
    axes[1].set_title('Expected Output', fontsize=14)
    add_grid_lines(axes[1], output_grid.shape)
    
    # Plot predicted output if provided
    if predicted_grid is not None:
        im3 = axes[2].imshow(predicted_grid.numpy(), cmap=cmap, vmin=0, vmax=9)
        axes[2].set_title('Predicted Output', fontsize=14)
        add_grid_lines(axes[2], predicted_grid.shape)
        
        # Add accuracy to title if prediction is provided
        accuracy = torch.equal(output_grid, predicted_grid)
        fig.suptitle(f'Task {task_id} - Pair {pair_idx} - {"Correct" if accuracy else "Incorrect"}', 
                    fontsize=16)
    else:
        fig.suptitle(f'Task {task_id} - Pair {pair_idx}', fontsize=16)
    
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im1, cax=cbar_ax, ticks=range(10))
    cbar_ax.set_ylabel('Color Index', rotation=270, labelpad=15)
    
    # Save figure
    filename = f"task_{task_id}_pair_{pair_idx}.png" if task_id else f"pair_{pair_idx}.png"
    filepath = os.path.join(task_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def add_grid_lines(ax, shape):
    """Add grid lines to a matplotlib axis."""
    ax.set_xticks(np.arange(-0.5, shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def save_evaluation_summary(evaluation_results: Dict[str, float], 
                          output_dir: str = "evaluation_images") -> None:
    """
    Save a summary of evaluation results as a text file and optionally a plot.
    
    Args:
        evaluation_results: Dictionary containing evaluation metrics
        output_dir: Directory to save the summary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save text summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(output_dir, f"evaluation_summary_{timestamp}.txt")
    
    with open(summary_file, 'w') as f:
        f.write(f"Evaluation Summary - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        
        for metric, value in evaluation_results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"Evaluation summary saved to: {summary_file}")


def create_evaluation_gif(image_dir: str, output_filename: str = "evaluation_results.gif",
                         duration: int = 1000) -> None:
    """
    Create an animated GIF from evaluation images.
    
    Args:
        image_dir: Directory containing evaluation images
        output_filename: Output GIF filename
        duration: Duration per frame in milliseconds
    """
    try:
        from PIL import Image
        import glob
        
        # Get all PNG files
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        
        if not image_files:
            print("No images found to create GIF")
            return
        
        # Load images
        images = []
        for filename in image_files:
            images.append(Image.open(filename))
        
        # Save as GIF
        output_path = os.path.join(image_dir, output_filename)
        images[0].save(output_path, save_all=True, append_images=images[1:], 
                      duration=duration, loop=0)
        
        print(f"Created evaluation GIF: {output_path}")
        
    except ImportError:
        print("PIL not available. Cannot create GIF.")