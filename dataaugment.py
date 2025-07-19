import numpy as np
import os
from PIL import Image

# Configuration
BLOCK_SIZE = 50
GRID_SIZE = 100
ROTATIONS = [0, 90, 180, 270]  # in degrees
offsets = [(0, 0), (10, 10), (25, 25), (40, 40)]  # example offsets

# Input paths (grayscale images with shape HxWx2 stacked)
spatial_path = r"C:\Users\jen\Desktop\Liouville\ulam_spiral_liouville.png"
fft_path = r"C:\Users\jen\Desktop\Liouville\ulam_spiral_fft.png"
output_dir = "augmented_grids"
os.makedirs(output_dir, exist_ok=True)

def load_image(path):
    return np.array(Image.open(path).convert("L"))

def rotate(img, angle):
    return np.array(Image.fromarray(img).rotate(angle, resample=Image.BICUBIC))

def extract_block_grid(image, offset):
    h_offset, w_offset = offset
    h, w, _ = image.shape
    max_rows = (h - h_offset) // BLOCK_SIZE
    max_cols = (w - w_offset) // BLOCK_SIZE
    grid_h = min(GRID_SIZE, max_rows)
    grid_w = min(GRID_SIZE, max_cols)
    if grid_h < GRID_SIZE or grid_w < GRID_SIZE:
        print(f"⚠️ Skipping offset {offset}: insufficient space for full grid ({grid_h}x{grid_w})")
        return None

    blocks = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            y = h_offset + i * BLOCK_SIZE
            x = w_offset + j * BLOCK_SIZE
            block = image[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE, :]
            blocks.append(block)
    return np.stack(blocks)  # (10000, 50, 50, 2)

# Load base grayscale and FFT images
spatial = load_image(spatial_path)
fft = load_image(fft_path)

# Stack spatial + fft to shape (H, W, 2)
full_img = np.stack([spatial, fft], axis=-1)

# Augment with rotations and offsets
for angle in ROTATIONS:
    rotated_img = rotate(full_img, angle)

    for offset in offsets:
        h_offset, w_offset = offset
        key = f"rot{angle}_offset{h_offset}_{w_offset}"
        blocks = extract_block_grid(rotated_img, offset)
        if blocks is None:
            continue
        output_path = os.path.join(output_dir, f"ulam_blocks_{key}.npy")
        np.save(output_path, blocks.astype(np.uint8))
        print(f"✅ Saved: {output_path} — shape: {blocks.shape}")
