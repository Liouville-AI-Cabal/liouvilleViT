import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from masked_block_vit import MaskedBlockViT
from multi_grid_dataset import MultiGridDataset
import math


def evaluate(model, batch):
    model.eval()
    with torch.no_grad():
        pred, ids_mask = model(batch)
        target = batch.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()  # (10000, 50, 50, 2)
        pred_np = pred.squeeze(0).cpu().numpy()  # (10000, 50, 50, 2)

        mask = torch.zeros((10000,), dtype=torch.bool)
        mask[ids_mask[0]] = True

        pred_flat = pred_np[mask.numpy()].reshape(-1)
        target_flat = target[mask.numpy()].reshape(-1)

        mse = ((pred_flat - target_flat) ** 2).mean()
        mae = np.abs(pred_flat - target_flat).mean()
        psnr = 10 * math.log10(1.0 / (mse + 1e-8))

    return mse, mae, psnr


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedBlockViT().to(device)

    dataset = MultiGridDataset(npy_dir="augmented_grids", normalize=True, num_blocks=10000)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.MSELoss()
    best_mse = float('inf')

    for epoch in range(1, 101):
        model.train()
        for batch in dataloader:
            batch = batch.to(device)  # shape: [1, 10000, 2, 50, 50]
            pred, ids_mask = model(batch)  # pred: [1, 10000, 50, 50, 2]

            target = batch.squeeze(0).permute(0, 2, 3, 1)  # [10000, 50, 50, 2]
            pred = pred.squeeze(0)  # [10000, 50, 50, 2]

            mask = torch.zeros((10000,), dtype=torch.bool, device=device)
            mask[ids_mask[0]] = True

            loss = loss_fn(pred[mask], target[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mse, mae, psnr = evaluate(model, batch)
        print(f"Epoch {epoch} | Loss: {loss.item():.6f} | MSE: {mse:.6f} | MAE: {mae:.6f} | PSNR: {psnr:.2f} dB")

        if mse < best_mse:
            best_mse = mse
            torch.save(model.state_dict(), "best_masked_block_vit.pt")
            print(f"✅ Saved new best model at epoch {epoch} (MSE: {mse:.6f})")


if __name__ == "__main__":
    import numpy as np
    run_training()
