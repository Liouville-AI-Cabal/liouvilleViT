import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlockEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),  # [2, 50, 50] -> [32, 50, 50]
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> [64, 25, 25]
            nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # -> [64, 13, 13]
            nn.GELU(),
            nn.Flatten(),  # -> [64*13*13]
            nn.Linear(64 * 13 * 13, out_dim)
        )

    def forward(self, x):
        print("  CNNBlockEncoder input shape:", x.shape)
        out = self.encoder(x)
        print("  CNNBlockEncoder output shape:", out.shape)
        return out  # x: [B, 2, 50, 50] -> [B, out_dim]

class MaskedBlockViT(nn.Module):
    def __init__(self, grid_size=100, embed_dim=128, depth=8, num_heads=8, mask_ratio=0.3):
        super().__init__()
        self.grid_size = grid_size # Store grid_size for 2D positional embeddings
        self.num_tokens = grid_size ** 2
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        # CNN block encoder
        self.block_encoder = CNNBlockEncoder(out_dim=embed_dim)

        # 2D Learnable Positional encoding
        self.row_embed = nn.Parameter(torch.randn(grid_size, embed_dim))
        self.col_embed = nn.Parameter(torch.randn(grid_size, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Decoder: map embedding back to full block using convolutions
        # The CNNBlockEncoder's last conv layer outputs 64 channels at 13x13 resolution.
        self.linear_decoder_projection = nn.Sequential(
            nn.Linear(embed_dim, 64 * 13 * 13),
            nn.GELU()
        )
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # -> [64, 25, 25]
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # -> [32, 50, 50]
            nn.GELU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1) # -> [2, 50, 50] (final output channels)
        )

    def random_mask(self, x, mask_ratio):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        return x_masked, ids_keep, ids_mask, ids_restore

    def forward(self, x_blocks):
        print("MaskedBlockViT.forward called")
        print("  x_blocks shape:", x_blocks.shape)
        # x_blocks: [B, 10000, 2, 50, 50]
        B, N, C, H, W = x_blocks.shape

        x = x_blocks.view(-1, C, H, W)  # (B*N, 2, 50, 50)
        x_emb = self.block_encoder(x).view(B, N, -1)  # (B, 10000, embed_dim)
        print("  x_emb shape after encoder and view:", x_emb.shape)

        # Generate 2D positional embeddings by combining row and column embeddings
        row_embed_expanded = self.row_embed.unsqueeze(1)
        col_embed_expanded = self.col_embed.unsqueeze(0)
        pos_embed_2d = (row_embed_expanded + col_embed_expanded).view(1, self.num_tokens, self.embed_dim)

        x_emb = x_emb + pos_embed_2d[:, :N, :] # Add 2D positional embeddings
        print("  x_emb shape after adding pos_embed_2d:", x_emb.shape)

        x_masked, ids_keep, ids_mask, ids_restore = self.random_mask(x_emb, self.mask_ratio)
        print("  x_masked shape:", x_masked.shape)
        print("  ids_keep shape:", ids_keep.shape)
        print("  ids_mask shape:", ids_mask.shape)
        print("  ids_restore shape:", ids_restore.shape)

        encoded = self.transformer(x_masked)
        print("  encoded shape after transformer:", encoded.shape)

        # Prepare full sequence to decode
        B, L, D_encoded = encoded.shape # D_encoded is the embed_dim
        decoder_input = torch.zeros(B, N, D_encoded, device=x.device)
        decoder_input.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, D_encoded), encoded)
        print("  decoder_input shape:", decoder_input.shape)

        # Apply linear projection for each block's embedding
        linear_decoded = self.linear_decoder_projection(decoder_input) # (B, N, 64 * 13 * 13)
        print("  linear_decoded shape:", linear_decoded.shape)

        # Reshape for convolutional decoder input: (B*N, 64, 13, 13)
        conv_input = linear_decoded.view(B * N, 64, 13, 13)
        print("  conv_input shape:", conv_input.shape)

        # Apply convolutional decoder to reconstruct blocks
        pred_flat_conv = self.conv_decoder(conv_input) # (B*N, 2, 50, 50)
        print("  pred_flat_conv shape:", pred_flat_conv.shape)
        # Crop to (2, 50, 50) if necessary
        if pred_flat_conv.shape[2] > 50 or pred_flat_conv.shape[3] > 50:
            print(f"  Cropping pred_flat_conv from {pred_flat_conv.shape} to (2, 50, 50)")
            pred_flat_conv = pred_flat_conv[:, :, :50, :50]

        # Reshape back to original (B, N, 50, 50, 2)
        pred = pred_flat_conv.permute(0, 2, 3, 1).view(B, N, 50, 50, 2)
        return pred, ids_mask
