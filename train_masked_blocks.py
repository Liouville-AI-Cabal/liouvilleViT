import datetime
import traceback
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
from pytorch_msssim import SSIM
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.patches import Rectangle
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from masked_block_vit import MaskedBlockViT
from multi_grid_dataset import MultiGridDataset

def save_block_comparison_images(epoch, target_blocks, predicted_blocks, ids_mask, num_samples=5, save_dir="block_comparisons"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Move indices to CPU for matplotlib
    ids_mask = [ids.cpu() for ids in ids_mask]
    
    # Get grid size from the number of blocks
    grid_size = int(np.sqrt(target_blocks.shape[1]))
    block_size = target_blocks.shape[-1]
    
    for sample_idx in range(min(num_samples, target_blocks.shape[0])):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
        
        # Reshape blocks into full images
        target = target_blocks[sample_idx].detach().cpu()
        target = target.reshape(grid_size, grid_size, 2, block_size, block_size)
        target = target.permute(0, 3, 1, 4, 2).reshape(grid_size*block_size, grid_size*block_size, 2)
        
        pred = predicted_blocks[sample_idx].detach().cpu()
        # Create a copy of the target for visualization
        pred_viz = target.clone()
        # Only replace the masked blocks with predictions
        mask = torch.zeros(pred.shape[0], dtype=torch.bool, device=pred.device)
        mask[ids_mask[sample_idx]] = True
        pred_viz[mask] = pred[mask]
        
        # Reshape for visualization
        pred = pred_viz.reshape(grid_size, grid_size, block_size, block_size, 2)
        pred = pred.permute(0, 2, 1, 3, 4).reshape(grid_size*block_size, grid_size*block_size, 2)
        
        # Plot ground truth - spatial channel
        im1 = ax1.imshow(target[..., 0].numpy(), cmap='gray')
        ax1.set_title("Ground Truth (Spatial)")
        plt.colorbar(im1, ax=ax1)
        
        # Plot ground truth - FFT channel
        im2 = ax2.imshow(target[..., 1].numpy(), cmap='gray')
        ax2.set_title("Ground Truth (FFT)")
        plt.colorbar(im2, ax=ax2)
        
        # Plot prediction - spatial channel
        im3 = ax3.imshow(pred[..., 0].numpy(), cmap='gray')
        ax3.set_title("Prediction (Spatial)")
        plt.colorbar(im3, ax=ax3)
        
        # Plot prediction - FFT channel
        im4 = ax4.imshow(pred[..., 1].numpy(), cmap='gray')
        ax4.set_title("Prediction (FFT)")
        plt.colorbar(im4, ax=ax4)
        
        # Highlight masked blocks in red on all plots
        for idx in ids_mask[sample_idx]:
            row = int(idx.item()) // grid_size  # Convert tensor to int
            col = int(idx.item()) % grid_size
            for ax in [ax1, ax2, ax3, ax4]:
                rect = Rectangle((col*block_size, row*block_size), block_size, block_size, 
                               fill=False, color='red', linewidth=2)
                ax.add_patch(rect)
        
        # Remove axes for cleaner visualization
        for ax in [ax1, ax2, ax3, ax4]:
            ax.axis('off')
        
        # Add super title
        plt.suptitle(f'Epoch {epoch} - Sample {sample_idx}', fontsize=16, y=0.95)
        
        # Save the figure
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_sample_{sample_idx}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        # Also log to wandb
        wandb.log({
            f"reconstructions/sample_{sample_idx}": wandb.Image(save_path),
            "epoch": epoch
        })

def calculate_pixel_accuracy(pred, target, threshold=0.05):
    """Calculate percentage of pixels reconstructed within threshold."""
    diff = torch.abs(pred - target)
    correct_pixels = (diff <= threshold).float().mean() * 100
    return correct_pixels.item()

def calculate_gradient_norm(model):
    """Calculate the L2 norm of gradients."""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def evaluate(model, batch, ring_mask=None):
    model.eval()
    device = next(model.parameters()).device
    ssim_module = SSIM(data_range=1.0, size_average=True, channel=2)
    
    with torch.no_grad():
        pred, ids_mask = model(batch)
        target = batch[0].permute(0, 2, 3, 1)
        pred = pred[0]
        
        # Get masked tokens
        mask = torch.zeros(pred.shape[0], dtype=torch.bool, device=device)
        mask[ids_mask[0]] = True
        
        # Calculate metrics only on masked tokens
        pred_masked = pred[mask]
        target_masked = target[mask]
        
        mse = nn.MSELoss()(pred_masked, target_masked).item()
        mae = nn.L1Loss()(pred_masked, target_masked).item()
        psnr = -10 * torch.log10(torch.tensor(mse)).item()
        
        # Calculate SSIM on masked regions
        # Reshape for SSIM calculation (expects [B, C, H, W])
        pred_ssim = pred_masked.permute(0, 3, 1, 2)
        target_ssim = target_masked.permute(0, 3, 1, 2)
        ssim = ssim_module(pred_ssim, target_ssim).item()
        
        # Calculate pixel accuracy
        pixel_accuracy = calculate_pixel_accuracy(pred_masked, target_masked)
        
    return mse, mae, psnr, ssim, pixel_accuracy

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wandb.init(
        project="masked-block-vit-training",
        config={
            "learning_rate": 5e-4,
            "epochs": 100,
            "batch_size": 8,
            "grid_size": 10,
            "embed_dim": 256,
            "depth": 12,
            "num_heads": 8,
            "mask_ratio": 0.3,
        },
    )

    cfg = wandb.config
    
    # Print configuration
    print(f"\n📊 Training Configuration:")
    print(f"  Learning Rate: {cfg.learning_rate}")
    print(f"  Grid Size: {cfg.grid_size}x{cfg.grid_size} ({cfg.grid_size**2} blocks)")
    print(f"  Batch Size: {cfg.batch_size}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Device: {device}")
    print(f"  Embedding Dim: {cfg.embed_dim}")
    print(f"  Transformer Depth: {cfg.depth}")
    print(f"  Attention Heads: {cfg.num_heads}")
    print(f"  Mask Ratio: {cfg.mask_ratio:.1%}")

    # Model setup
    model = MaskedBlockViT(
        grid_size=cfg.grid_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mask_ratio=cfg.mask_ratio,
    ).to(device)

    # Dataset and dataloader
    dataset = MultiGridDataset(
        npy_dir="/media/cc/2T/liouvilleViT/augmented_grids",
        normalize=True,
        grid_size=cfg.grid_size,
        dataset_length=1000,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.05,
        betas=(0.9, 0.95)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=1e-6
    )

    loss_fn = nn.MSELoss()
    scaler = GradScaler()
    best_mse = float("inf")
    
    ssim_module = SSIM(data_range=1.0, size_average=True, channel=2)
    
    print(f"\n📊 Training Configuration:")
    print(f"  Learning Rate: {cfg.learning_rate}")
    print(f"  Grid Size: {cfg.grid_size}x{cfg.grid_size} ({cfg.grid_size**2} blocks)")
    print(f"  Mask Ratio: {model.mask_ratio:.1%}")  # Simplified to just access mask_ratio directly
    print(f"  Batch Size: {cfg.batch_size}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Device: {device}")
    print(f"  Dataset Length: {len(dataset)}")
    
    for epoch in range(1, cfg.epochs + 1):
        # sampler is not None:
        #     sampler.set_epoch(epoch)  # Ensure different ordering each epoch
            
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{cfg.epochs}")
        print(f"{'='*80}")
        
        model.train()
        total_loss = 0.0
        total_train_mae = 0.0
        total_train_psnr = 0.0
        total_train_ssim = 0.0
        total_train_pixel_acc = 0.0
        num_batches = 0
        
        # Add progress tracking
        print("\nTraining:")
        total_batches = len(dataloader)
        for batch_idx, batch in enumerate(dataloader, 1):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            with autocast():
                pred, ids_mask = model(batch)
                target = batch[0].permute(0, 2, 3, 1)
                pred = pred[0]
                
                mask = torch.zeros(pred.shape[0], dtype=torch.bool, device=device)
                mask[ids_mask[0]] = True
                
                # Calculate all metrics for training
                loss = loss_fn(pred[mask], target[mask])
                with torch.no_grad():
                    mae = nn.L1Loss()(pred[mask], target[mask]).item()
                    mse = nn.MSELoss()(pred[mask], target[mask]).item()
                    psnr = -10 * torch.log10(torch.tensor(mse)).item()
                    ssim = ssim_module(
                        pred[mask].permute(0, 3, 1, 2),
                        target[mask].permute(0, 3, 1, 2)
                    ).item()
                    pixel_acc = calculate_pixel_accuracy(pred[mask], target[mask])
                
            scaler.scale(loss).backward()
            grad_norm = calculate_gradient_norm(model)
            scaler.step(optimizer)
            scaler.update()
            
            # Accumulate training metrics
            total_loss += loss.item()
            total_train_mae += mae
            total_train_psnr += psnr
            total_train_ssim += ssim
            total_train_pixel_acc += pixel_acc
            num_batches += 1
            
            # Print progress every 10% of epoch
            if batch_idx % max(1, total_batches//10) == 0:
                progress = batch_idx/total_batches * 100
                print(f"  ▶ {progress:3.0f}% | Batch {batch_idx}/{total_batches} | "
                      f"Loss: {loss.item():.4f} | Grad Norm: {grad_norm:.2f}")
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_train_mae = total_train_mae / num_batches
        avg_train_psnr = total_train_psnr / num_batches
        avg_train_ssim = total_train_ssim / num_batches
        avg_train_pixel_acc = total_train_pixel_acc / num_batches
        
        # Print epoch summary with all metrics
        print("\n📈 Epoch Summary:")
        print("  Training Metrics:")
        print(f"    Loss:     {avg_loss:.6f}")
        print(f"    MAE:      {avg_train_mae:.6f}")
        print(f"    PSNR:     {avg_train_psnr:.2f} dB")
        print(f"    SSIM:     {avg_train_ssim:.4f}")
        print(f"    Pixel Acc: {avg_train_pixel_acc:.2f}%")
        
        # Evaluate
        print("\n  Evaluation Metrics:")
        eval_mse, eval_mae, eval_psnr, eval_ssim, eval_pixel_acc = evaluate(model, batch)
        print(f"    MSE:      {eval_mse:.6f}")
        print(f"    MAE:      {eval_mae:.6f}")
        print(f"    PSNR:     {eval_psnr:.2f} dB")
        print(f"    SSIM:     {eval_ssim:.4f}")
        print(f"    Pixel Acc: {eval_pixel_acc:.2f}%")
        
        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "gradient_norm": grad_norm,
            # Training metrics
            "train_loss": avg_loss,
            "train_mae": avg_train_mae,
            "train_psnr": avg_train_psnr,
            "train_ssim": avg_train_ssim,
            "train_pixel_accuracy": avg_train_pixel_acc,
            # Evaluation metrics
            "eval_mse": eval_mse,
            "eval_mae": eval_mae,
            "eval_psnr": eval_psnr,
            "eval_ssim": eval_ssim,
            "eval_pixel_accuracy": eval_pixel_acc,
        })

        # Step scheduler at end of epoch
        scheduler.step()

        # Print model saving info with more detail
        if eval_mse < best_mse:
            improvement = (best_mse - eval_mse) / best_mse * 100
            print(f"\n✨ New Best Model!")
            print(f"  Previous MSE: {best_mse:.6f}")
            print(f"  New MSE:      {eval_mse:.6f}")
            print(f"  Improvement:   {improvement:.2f}%")
            best_mse = eval_mse
            torch.save(model.state_dict(), "best_masked_block_vit.pt")

        # Only save visualizations after the first epoch
        if epoch > 1:
            print("\n📸 Saving prediction visualizations...")
            save_block_comparison_images(
                epoch=epoch,
                target_blocks=batch,  # Original input
                predicted_blocks=pred.unsqueeze(0),  # Add batch dimension back
                ids_mask=ids_mask,
                num_samples=1,  # Only show one sample
                save_dir="block_comparisons"
            )

    print("\n🎉 Training Complete!")
    print(f"  Best MSE: {best_mse:.6f}")
    wandb.finish()

if __name__ == "__main__":
    # Launch with torch.distributed.launch
    run_training()
