import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class MultiGridDataset(Dataset):
    def __init__(self, npy_dir, normalize=True, num_blocks=10000):
        self.files = [os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.endswith('.npy')]
        self.normalize = normalize
        self.num_blocks = num_blocks
        self.mapped_data = [np.load(f, mmap_mode='r') for f in self.files]

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        data = random.choice(self.mapped_data)  # shape: (N, 50, 50, 2)
        total_blocks = data.shape[0]
        if self.num_blocks < total_blocks:
            indices = np.random.choice(total_blocks, size=self.num_blocks, replace=False)
        else:
            indices = np.arange(total_blocks)

        # Initialize tensor batch and normalize per block to save memory
        blocks = torch.empty((len(indices), 2, 50, 50), dtype=torch.float32)
        for i, idx in enumerate(indices):
            block = data[idx].astype(np.float32)
            if self.normalize:
                block /= 255.0
            blocks[i] = torch.from_numpy(block).permute(2, 0, 1)

        return blocks
