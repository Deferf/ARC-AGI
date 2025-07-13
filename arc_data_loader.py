import json
import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import random


class ARCTask:
    """
    Represents a single ARC task with training and test pairs.
    """
    def __init__(self, task_id: str, train_pairs: List[Dict], test_pairs: List[Dict]):
        self.task_id = task_id
        self.train_pairs = train_pairs
        self.test_pairs = test_pairs
        
    @classmethod
    def from_json(cls, task_id: str, json_data: Dict) -> 'ARCTask':
        """Create ARCTask from JSON data."""
        return cls(
            task_id=task_id,
            train_pairs=json_data.get('train', []),
            test_pairs=json_data.get('test', [])
        )
    
    def get_grid_dimensions(self) -> Tuple[int, int]:
        """Get the maximum grid dimensions across all pairs."""
        max_height = 0
        max_width = 0
        
        for pair in self.train_pairs + self.test_pairs:
            for grid in [pair['input'], pair['output']]:
                max_height = max(max_height, len(grid))
                max_width = max(max_width, len(grid[0]) if grid else 0)
        
        return max_height, max_width


class ARCDataset(Dataset):
    """
    PyTorch Dataset for ARC tasks.
    """
    def __init__(self, data_dir: str, max_grid_size: int = 30, augment: bool = True):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.augment = augment
        self.tasks = []
        self.pairs = []
        
        self._load_tasks()
        self._create_pairs()
    
    def _load_tasks(self):
        """Load all ARC tasks from the data directory."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} not found")
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                task_id = filename.replace('.json', '')
                filepath = os.path.join(self.data_dir, filename)
                
                with open(filepath, 'r') as f:
                    json_data = json.load(f)
                
                task = ARCTask.from_json(task_id, json_data)
                self.tasks.append(task)
    
    def _create_pairs(self):
        """Create training pairs from all tasks."""
        for task in self.tasks:
            for pair in task.train_pairs:
                self.pairs.append({
                    'task_id': task.task_id,
                    'input': pair['input'],
                    'output': pair['output']
                })
    
    def _pad_grid(self, grid: List[List[int]], target_height: int, target_width: int) -> List[List[int]]:
        """Pad grid to target dimensions with zeros."""
        current_height = len(grid)
        current_width = len(grid[0]) if grid else 0
        
        # Create padded grid
        padded_grid = [[0] * target_width for _ in range(target_height)]
        
        # Copy original grid
        for i in range(min(current_height, target_height)):
            for j in range(min(current_width, target_width)):
                padded_grid[i][j] = grid[i][j]
        
        return padded_grid
    
    def _augment_grid(self, grid: List[List[int]]) -> List[List[int]]:
        """Apply data augmentation to grid."""
        if not self.augment:
            return grid
        
        # Random rotation (0, 90, 180, 270 degrees)
        rotation = random.choice([0, 1, 2, 3])
        for _ in range(rotation):
            grid = list(zip(*grid[::-1]))  # Rotate 90 degrees clockwise
            grid = [list(row) for row in grid]
        
        # Random horizontal flip
        if random.random() < 0.5:
            grid = [row[::-1] for row in grid]
        
        # Random vertical flip
        if random.random() < 0.5:
            grid = grid[::-1]
        
        return grid
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx]
        
        # Get input and output grids
        input_grid = pair['input']
        output_grid = pair['output']
        
        # Apply augmentation
        if self.augment:
            input_grid = self._augment_grid(input_grid)
            output_grid = self._augment_grid(output_grid)
        
        # Pad grids to maximum size
        input_grid = self._pad_grid(input_grid, self.max_grid_size, self.max_grid_size)
        output_grid = self._pad_grid(output_grid, self.max_grid_size, self.max_grid_size)
        
        # Convert to tensors
        input_tensor = torch.tensor(input_grid, dtype=torch.long)
        output_tensor = torch.tensor(output_grid, dtype=torch.long)
        
        return input_tensor, output_tensor


class ARCTestDataset(Dataset):
    """
    Dataset for ARC test tasks (without output labels).
    """
    def __init__(self, data_dir: str, max_grid_size: int = 30):
        self.data_dir = data_dir
        self.max_grid_size = max_grid_size
        self.test_pairs = []
        
        self._load_test_pairs()
    
    def _load_test_pairs(self):
        """Load test pairs from all tasks."""
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                task_id = filename.replace('.json', '')
                filepath = os.path.join(self.data_dir, filename)
                
                with open(filepath, 'r') as f:
                    json_data = json.load(f)
                
                for i, pair in enumerate(json_data.get('test', [])):
                    self.test_pairs.append({
                        'task_id': task_id,
                        'pair_index': i,
                        'input': pair['input'],
                        'output': pair.get('output')  # May be None for actual test
                    })
    
    def _pad_grid(self, grid: List[List[int]], target_height: int, target_width: int) -> List[List[int]]:
        """Pad grid to target dimensions with zeros."""
        current_height = len(grid)
        current_width = len(grid[0]) if grid else 0
        
        padded_grid = [[0] * target_width for _ in range(target_height)]
        
        for i in range(min(current_height, target_height)):
            for j in range(min(current_width, target_width)):
                padded_grid[i][j] = grid[i][j]
        
        return padded_grid
    
    def __len__(self) -> int:
        return len(self.test_pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        pair = self.test_pairs[idx]
        
        input_grid = self._pad_grid(pair['input'], self.max_grid_size, self.max_grid_size)
        input_tensor = torch.tensor(input_grid, dtype=torch.long)
        
        result = {
            'task_id': pair['task_id'],
            'pair_index': pair['pair_index'],
            'input': input_tensor
        }
        
        if pair['output'] is not None:
            output_grid = self._pad_grid(pair['output'], self.max_grid_size, self.max_grid_size)
            output_tensor = torch.tensor(output_grid, dtype=torch.long)
            result['output'] = output_tensor
        
        return result


def create_arc_dataloaders(train_dir: str, val_dir: Optional[str] = None, 
                          batch_size: int = 32, max_grid_size: int = 30,
                          num_workers: int = 4) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders for ARC tasks.
    
    Args:
        train_dir: Directory containing training task JSON files
        val_dir: Directory containing validation task JSON files (optional)
        batch_size: Batch size for training
        max_grid_size: Maximum grid size for padding
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    train_dataset = ARCDataset(train_dir, max_grid_size=max_grid_size, augment=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_dataloader = None
    if val_dir:
        val_dataset = ARCDataset(val_dir, max_grid_size=max_grid_size, augment=False)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_dataloader, val_dataloader


def load_arc_task(task_file: str) -> ARCTask:
    """
    Load a single ARC task from JSON file.
    
    Args:
        task_file: Path to the JSON task file
    
    Returns:
        ARCTask object
    """
    with open(task_file, 'r') as f:
        json_data = json.load(f)
    
    task_id = os.path.basename(task_file).replace('.json', '')
    return ARCTask.from_json(task_id, json_data)


def visualize_grid(grid: torch.Tensor, title: str = "Grid") -> None:
    """
    Visualize a grid using matplotlib.
    
    Args:
        grid: Grid tensor [height, width] or [batch_size, height, width]
        title: Title for the plot
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        if grid.dim() == 3:
            grid = grid[0]  # Take first batch
        
        # Create color map for ARC colors (0-9)
        colors = ['white', 'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray']
        cmap = mcolors.ListedColormap(colors)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(grid.numpy(), cmap=cmap, vmin=0, vmax=9)
        plt.title(title)
        plt.colorbar(ticks=range(10))
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Grid shape:", grid.shape)
        print("Grid content:")
        print(grid.numpy())


def grid_to_string(grid: torch.Tensor) -> str:
    """
    Convert grid tensor to string representation.
    
    Args:
        grid: Grid tensor [height, width]
    
    Returns:
        String representation of the grid
    """
    if grid.dim() == 3:
        grid = grid[0]
    
    lines = []
    for row in grid:
        line = ''.join(str(int(cell)) for cell in row)
        lines.append(line)
    
    return '\n'.join(lines)


def string_to_grid(grid_str: str) -> torch.Tensor:
    """
    Convert string representation back to grid tensor.
    
    Args:
        grid_str: String representation of grid
    
    Returns:
        Grid tensor [height, width]
    """
    lines = grid_str.strip().split('\n')
    height = len(lines)
    width = len(lines[0]) if lines else 0
    
    grid = torch.zeros(height, width, dtype=torch.long)
    
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            grid[i, j] = int(char)
    
    return grid


if __name__ == "__main__":
    # Example usage
    print("Testing ARC data loader...")
    
    # Check if data directory exists
    data_dir = "data/training"
    if os.path.exists(data_dir):
        print(f"Loading tasks from {data_dir}")
        
        # Load a few tasks
        tasks = []
        for filename in os.listdir(data_dir)[:3]:  # Load first 3 tasks
            if filename.endswith('.json'):
                task = load_arc_task(os.path.join(data_dir, filename))
                tasks.append(task)
                print(f"Loaded task {task.task_id} with {len(task.train_pairs)} training pairs")
        
        # Create dataset
        dataset = ARCDataset(data_dir, max_grid_size=30, augment=False)
        print(f"Created dataset with {len(dataset)} pairs")
        
        # Test data loading
        if len(dataset) > 0:
            input_grid, output_grid = dataset[0]
            print(f"Sample input grid shape: {input_grid.shape}")
            print(f"Sample output grid shape: {output_grid.shape}")
            
            # Visualize sample
            print("Sample input grid:")
            print(grid_to_string(input_grid))
            print("\nSample output grid:")
            print(grid_to_string(output_grid))
    
    else:
        print(f"Data directory {data_dir} not found. Please ensure ARC data is available.")