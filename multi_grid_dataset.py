import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class MultiGridDataset(Dataset):
    def __init__(self, npy_dir=None, normalize=True, grid_size=10,
                 dataset_length=1000, file_list=None):
        self.dataset_length = dataset_length
        if file_list is not None:
            self.files = [os.path.join(npy_dir, f) for f in file_list]
        else:
            self.files = [os.path.join(npy_dir, f) for f in os.listdir(npy_dir)
                         if f.endswith('.npy')]
        self.normalize = normalize
        self.grid_size = grid_size
        self.num_blocks = grid_size * grid_size
        self.mapped_data = [np.load(f, mmap_mode="r") for f in self.files]

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        data = random.choice(self.mapped_data)  # shape: (N, 50, 50, 2)
        total_blocks = data.shape[0]
        blocks_per_row = int(np.sqrt(total_blocks))  # Should be 100 for your 100x100 grid
        
        # Calculate valid starting positions that allow a complete grid_size x grid_size window
        valid_starts_row = blocks_per_row - self.grid_size + 1
        valid_starts_col = valid_starts_row
        
        # Randomly select a starting position for the top-left corner of our grid
        start_row = random.randint(0, valid_starts_row - 1)
        start_col = random.randint(0, valid_starts_col - 1)
        
        # Get indices for a contiguous grid_size x grid_size region
        indices = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                block_idx = (start_row + i) * blocks_per_row + (start_col + j)
                indices.append(block_idx)
        
        # Initialize tensor batch and normalize per block to save memory
        blocks = torch.empty((self.num_blocks, 2, 50, 50), dtype=torch.float32)
        for i, idx in enumerate(indices):
            block = data[idx].astype(np.float32)
            if self.normalize:
                block /= 255.0
            blocks[i] = torch.from_numpy(block).permute(2, 0, 1)

        return blocks
