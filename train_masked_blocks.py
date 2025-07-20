import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from masked_block_vit import MaskedBlockViT
from multi_grid_dataset import MultiGridDataset
import math
import numpy as np # Already imported
import matplotlib.pyplot as plt # New import
import os # New import
import wandb 


def save_block_comparison_images(epoch, target_blocks, predicted_blocks, ids_mask, num_samples=5, save_dir="block_comparisons"):
    """
    Saves comparison images of ground truth vs predicted blocks for a few masked samples.

    Args:
        epoch (int): Current epoch number.
        target_blocks (np.array): Ground truth blocks (N, 50, 50, 2).
        predicted_blocks (np.array): Predicted blocks (N, 50, 50, 2).
        ids_mask (torch.Tensor): Indices of the masked blocks (tensor of shape [B, num_masked_tokens]).
        num_samples (int): Number of masked blocks to visualize.
        save_dir (str): Directory to save the images.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # We need the actual 1D indices from the first batch item
    masked_indices = ids_mask[0].cpu().numpy()
    
    if len(masked_indices) > num_samples:
        selected_indices = np.random.choice(masked_indices, size=num_samples, replace=False)
    else:
        selected_indices = masked_indices

    for i, block_idx in enumerate(selected_indices):
        gt_block = target_blocks[block_idx] # (50, 50, 2)
        pred_block = predicted_blocks[block_idx] # (50, 50, 2)

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle(f"Epoch {epoch} - Masked Block {block_idx} Comparison", fontsize=16)

        # Channel 0
        axes[0, 0].imshow(gt_block[:, :, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title("GT Channel 0")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(pred_block[:, :, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title("Pred Channel 0")
        axes[0, 1].axis('off')

        # Channel 1
        axes[1, 0].imshow(gt_block[:, :, 1], cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title("GT Channel 1")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(pred_block[:, :, 1], cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title("Pred Channel 1")
        axes[1, 1].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch:03d}_block_{block_idx:05d}.png"))
        plt.close(fig) # Close the figure to free up memory
        print(f"    Saved comparison image: {os.path.join(save_dir, f'epoch_{epoch:03d}_block_{block_idx:05d}.png')}")


def evaluate(model, batch):
    model.eval()
    with torch.no_grad():
        pred, ids_mask = model(batch)
        target = batch.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()  # (10000, 50, 50, 2)
        pred_np = pred.squeeze(0).cpu().numpy()  # (10000, 50, 50, 2)

        mask = torch.zeros((10000,), dtype=torch.bool)
        mask[ids_mask[0]] = True # ids_mask[0] contains the indices of masked tokens for the first batch item

        pred_flat = pred_np[mask.numpy()].reshape(-1)
        target_flat = target[mask.numpy()].reshape(-1)

        mse = ((pred_flat - target_flat) ** 2).mean()
        mae = np.abs(pred_flat - target_flat).mean()
        psnr = 10 * math.log10(1.0 / (mse + 1e-8))

    return mse, mae, psnr, target, pred_np, ids_mask # Return target, pred_np, and ids_mask


def run_training():
    # Initialize WandB
    # Replace 'your_project_name' with a descriptive name for your project
    # You can also add 'entity="your_wandb_username"' if you want to log to a specific team/user
    wandb.init(project="masked-block-vit-training", config={
        "learning_rate": 2e-4,
        "epochs": 100,
        "batch_size": 1,
        "num_blocks_per_grid": 10000,
        # Add model specific hyperparameters from MaskedBlockViT if desired
        # e.g., "embed_dim": model.embed_dim,
        # "depth": model.transformer.num_layers,
        # "num_heads": model.transformer.encoder_layer.nhead,
        # "mask_ratio": model.mask_ratio,
        # "grid_size": model.grid_size
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedBlockViT().to(device)

    # Log model architecture (optional, but good for tracking)
    # wandb.watch(model) # This can be resource intensive for very large models

    dataset = MultiGridDataset(npy_dir="/home/cc/Desktop/liouvilleViT/augmented_grids", normalize=True, num_blocks=10000)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.learning_rate) # Use config for LR
    loss_fn = nn.MSELoss()
    best_mse = float('inf')

    for epoch in range(1, wandb.config.epochs + 1): # Use config for epochs
        print(f"\n--- Epoch {epoch}/{wandb.config.epochs} ---")
        model.train()
        total_loss = 0.0 # To track average loss per epoch
        num_batches = 0
        for batch in dataloader:
            batch = batch.to(device)  # shape: [1, 10000, 2, 50, 50]
            print("Batch shape:", batch.shape)
            pred, ids_mask = model(batch)  # pred: [1, 10000, 50, 50, 2]
            print("x_blocks.shape:", pred.shape) # Added print statement

            target = batch.squeeze(0).permute(0, 2, 3, 1)  # [10000, 50, 50, 2]
            pred = pred.squeeze(0)  # [10000, 50, 50, 2]

            mask = torch.zeros((10000,), dtype=torch.bool, device=device)
            mask[ids_mask[0]] = True

            loss = loss_fn(pred[mask], target[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            # print(f"  Batch {num_batches}: Loss = {loss.item():.6f}") # Optional: print batch loss
            print(f"  Batch {num_batches+1}: Loss = {loss.item():.6f}")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Get additional returns from evaluate for visualization
        mse, mae, psnr, target_blocks_eval, pred_blocks_eval, ids_mask_eval = evaluate(model, batch)
        print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | MSE: {mse:.6f} | MAE: {mae:.6f} | PSNR: {psnr:.2f} dB")

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "eval_mse": mse,
            "eval_mae": mae,
            "eval_psnr": psnr
        })

        # Save comparison images after evaluation
        save_block_comparison_images(epoch, target_blocks_eval, pred_blocks_eval, ids_mask_eval)

        if mse < best_mse:
            best_mse = mse
            torch.save(model.state_dict(), "best_masked_block_vit.pt")
            print(f"✅ Saved new best model at epoch {epoch} (MSE: {mse:.6f})")
    
    wandb.finish() # Finish the WandB run
    print("Training complete! Best MSE achieved:", best_mse)


if __name__ == "__main__":
    run_training()
