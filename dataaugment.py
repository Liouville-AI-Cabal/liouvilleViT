import numpy as np
import os
from PIL import Image

# Configuration
BLOCK_SIZE = 50
GRID_SIZE = 100
ROTATIONS = [0, 90, 180, 270]  # in degrees

# Input paths (grayscale images with shape HxWx2 stacked)
spatial_path = r"/media/cc/2T/liouvilleViT/ulam_spiral_liouville.png"
fft_path = r"/media/cc/2T/liouvilleViT/ulam_spiral_fft_hi_contrast.png"
output_dir = "augmented_grids"
os.makedirs(output_dir, exist_ok=True)

def load_image(path):
    return np.array(Image.open(path).convert("L"))

def rotate(img, angle):
    return np.array(Image.fromarray(img).rotate(angle, resample=Image.BICUBIC))

def extract_block_grid(image):
    h, w, _ = image.shape
    max_rows = h // BLOCK_SIZE
    max_cols = w // BLOCK_SIZE
    grid_h = min(GRID_SIZE, max_rows)
    grid_w = min(GRID_SIZE, max_cols)
    if grid_h < GRID_SIZE or grid_w < GRID_SIZE:
        print(f"⚠️ Insufficient space for full grid ({grid_h}x{grid_w})")
        return None

    blocks = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            y = i * BLOCK_SIZE
            x = j * BLOCK_SIZE
            block = image[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE, :]
            blocks.append(block)
    return np.stack(blocks)  # (10000, 50, 50, 2)

# Load base grayscale and FFT images
spatial = load_image(spatial_path)
fft = load_image(fft_path)

# Stack spatial + fft to shape (H, W, 2)
full_img = np.stack([spatial, fft], axis=-1)

# Augment with rotations only
for angle in ROTATIONS:
    rotated_img = rotate(full_img, angle)
    key = f"rot{angle}"
    blocks = extract_block_grid(rotated_img)
    if blocks is None:
        continue
    # decide which split this file belongs to, e.g. put rot0 into test
    split = "test" if angle == 0 else "train"
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    output_path = os.path.join(split_dir, f"ulam_blocks_{key}.npy")
    np.save(output_path, blocks.astype(np.uint8))
    print(f"✅ Saved: {output_path} — shape: {blocks.shape}")
