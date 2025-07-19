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
        return self.encoder(x)  # x: [B, 2, 50, 50] -> [B, out_dim]


class MaskedBlockViT(nn.Module):
    def __init__(self, grid_size=100, embed_dim=128, depth=8, num_heads=8, mask_ratio=0.3):
        super().__init__()
        self.num_tokens = grid_size ** 2
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        # CNN block encoder
        self.block_encoder = CNNBlockEncoder(out_dim=embed_dim)

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Decoder: map embedding back to full block
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, 50 * 50 * 2)
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
        # x_blocks: [B, 10000, 2, 50, 50]
        B, N, C, H, W = x_blocks.shape

        x = x_blocks.view(-1, C, H, W)  # (B*N, 2, 50, 50)
        x_emb = self.block_encoder(x).view(B, N, -1)  # (B, 10000, embed_dim)

        x_emb = x_emb + self.pos_embed[:, :N, :]

        x_masked, ids_keep, ids_mask, ids_restore = self.random_mask(x_emb, self.mask_ratio)

        encoded = self.transformer(x_masked)

        # Prepare full sequence to decode
        B, L, D = encoded.shape
        decoder_input = torch.zeros(B, N, D, device=x.device)
        decoder_input.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, D), encoded)

        # Decode all tokens
        pred = self.decoder(decoder_input)  # (B, N, 5000)
        return pred.view(B, N, 50, 50, 2), ids_mask
