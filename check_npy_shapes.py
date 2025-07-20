import os
import numpy as np

npy_dir = '/home/cc/Desktop/liouvilleViT/augmented_grids'
expected_shape = (10000, 50, 50, 2)

for fname in os.listdir(npy_dir):
    if fname.endswith('.npy'):
        path = os.path.join(npy_dir, fname)
        try:
            arr = np.load(path)
            if arr.shape != expected_shape:
                print(f"❌ {fname}: shape {arr.shape} (EXPECTED {expected_shape})")
            else:
                print(f"✅ {fname}: shape {arr.shape}")
        except Exception as e:
            print(f"⚠️  {fname}: could not load ({e})")

