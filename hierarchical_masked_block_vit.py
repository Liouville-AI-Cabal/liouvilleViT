import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Existing CNNBlockEncoder ---
class CNNBlockEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 13 * 13, out_dim)
        )

    def forward(self, x):
        return self.encoder(x)

# --- NEW: Patch Merging Module (Downsampling) ---
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution # (H, W) of blocks grid
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False) # Merges 2x2 blocks, so 4 times the dim, reduces to 2*dim
        self.norm = norm_layer(4 * dim) # Norm before linear reduction

    def forward(self, x):
        # x: (B, H*W, D)
        H, W = self.input_resolution
        B, L, D = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) is not even."

        x = x.view(B, H, W, D)

        # Rearrange to merge 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 0::2, 1::2, :]  # Top-right
        x2 = x[:, 1::2, 0::2, :]  # Bottom-left
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4*D)
        x = x.view(B, -1, 4 * D)  # (B, H/2 * W/2, 4*D)

        x = self.norm(x)
        x = self.reduction(x) # (B, H/2 * W/2, 2*D)

        return x

# --- NEW: Patch Expanding Module (Upsampling for Decoder) ---
class PatchExpanding(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution # (H, W) of blocks grid
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) # Expands dim to 2*dim for upsampling
        self.norm = norm_layer(dim // 2) # Norm after linear expansion and reshaping
        
    def forward(self, x):
        # x: (B, H*W, D)
        H, W = self.input_resolution
        B, L, D = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.expand(x) # (B, H*W, 2*D)
        x = x.view(B, H, W, 2*D)

        # Rearrange to expand patches
        x = x.view(B, H, W, 2, D // 2).permute(0, 1, 3, 2, 4).reshape(B, H * 2, W * 2, D // 2)
        x = x.view(B, -1, D // 2) # (B, H*2 * W*2, D/2)
        x = self.norm(x) # Apply norm after expanding and before returning
        return x

# --- MaskedBlockViT with Hierarchical Architecture ---
class MaskedBlockViT(nn.Module):
    def __init__(self, grid_size=100, embed_dim=128, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32], mask_ratio=0.3):
        super().__init__()
        self.grid_size = grid_size
        self.num_tokens = grid_size ** 2
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.num_stages = len(depths)

        # CNN block encoder
        self.block_encoder = CNNBlockEncoder(out_dim=embed_dim)

        # Learnable mask token for MAE
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional encoding for initial patch embeddings (before any merging)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim)) # Could be 2D learnable pos if preferred

        # --- Hierarchical Encoder Stages ---
        self.encoder_stages = nn.ModuleList()
        current_resolution = (grid_size, grid_size)
        current_dim = embed_dim

        for i in range(self.num_stages):
            # Patch Merging (except for the first stage)
            if i > 0:
                self.encoder_stages.append(PatchMerging(current_resolution, current_dim))
                current_resolution = (current_resolution[0] // 2, current_resolution[1] // 2)
                current_dim *= 2

            # Transformer Encoder layers for the current stage
            encoder_blocks = []
            for _ in range(depths[i]):
                # Use standard TransformerEncoderLayer or your CustomTransformerEncoderLayer if desired
                # If using CustomTransformerEncoderLayer, remember to pass grid_size, etc.
                encoder_blocks.append(
                    nn.TransformerEncoderLayer(d_model=current_dim, nhead=num_heads[i], batch_first=True)
                )
            self.encoder_stages.append(nn.Sequential(*encoder_blocks))

        # --- Decoder ---
        # The decoder now needs to upsample from the final encoder stage's resolution and dim
        # This example assumes a symmetric upsampling path with PatchExpanding
        self.decoder_stages = nn.ModuleList()
        # Start from the last encoder stage's output resolution and dim
        decoder_current_resolution = current_resolution
        decoder_current_dim = current_dim

        for i in range(self.num_stages - 1, -1, -1): # Iterate backwards
            # Patch Expanding (except for the last stage where we just project to pixel space)
            if i < self.num_stages - 1: # Only expand if not the very last stage
                self.decoder_stages.append(PatchExpanding(decoder_current_resolution, decoder_current_dim))
                decoder_current_resolution = (decoder_current_resolution[0] * 2, decoder_current_resolution[1] * 2)
                decoder_current_dim //= 2

            # Linear layer to project the embeddings to the original pixel space (50*50*2) for each block
            # This is simplified; often you'd have more complex upsampling here, e.g., ConvTranspose2d
            # If using ConvTranspose2d, the `linear_decoder_projection` and `conv_decoder` from previous step
            # would need to be adapted to work across multiple PatchExpanding outputs or as the final stage.
            self.decoder_stages.append(
                nn.Sequential(
                    nn.Linear(decoder_current_dim, 512),
                    nn.GELU(),
                    nn.Linear(512, 50 * 50 * 2)
                )
            )
        
    def random_mask(self, x, mask_ratio):
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]

        return ids_keep, ids_mask, ids_restore

    def forward(self, x_blocks):
        # x_blocks: [B, 10000, 2, 50, 50]
        B, N, C, H, W = x_blocks.shape

        x = x_blocks.view(-1, C, H, W)  # (B*N, 2, 50, 50)
        x_emb = self.block_encoder(x).view(B, N, -1)  # (B, 10000, embed_dim)

        # Add initial positional embeddings
        x_emb = x_emb + self.pos_embed[:, :N, :]

        # Generate mask indices
        ids_keep, ids_mask, ids_restore = self.random_mask(x_emb, self.mask_ratio)

        # --- MAE Masking for Encoder Input ---
        # Create full sequence with mask tokens
        masked_x_emb = x_emb.clone()
        # Replace masked tokens with the learnable mask token
        # Expand mask_token to match batch size and sequence length for masked positions
        mask_token_expanded = self.mask_token.expand(B, N, -1)
        # Scatter the mask tokens at the masked positions
        masked_x_emb.scatter_(1, ids_mask.unsqueeze(-1).expand(-1, -1, self.embed_dim), mask_token_expanded.gather(1, ids_mask.unsqueeze(-1).expand(-1, -1, self.embed_dim)))


        # --- Hierarchical Encoder Forward Pass ---
        current_x = masked_x_emb
        encoder_features = [] # Store features from each stage if needed
        stage_idx = 0
        for module in self.encoder_stages:
            if isinstance(module, PatchMerging):
                current_x = module(current_x)
            elif isinstance(module, nn.Sequential): # Transformer blocks
                current_x = module(current_x)
                encoder_features.append(current_x) # Store output of transformer blocks
            stage_idx += 1
        
        # --- Decoder Forward Pass ---
        # The decoder now starts from the final encoder output
        decoded_features = current_x # Output of the last encoder stage
        
        for module in self.decoder_stages:
            if isinstance(module, PatchExpanding):
                decoded_features = module(decoded_features)
            elif isinstance(module, nn.Sequential): # Linear projection/upsampling to pixel space
                decoded_features = module(decoded_features)

        # Final prediction reshape: (B, N, 50, 50, 2)
        pred = decoded_features.view(B, N, 50, 50, 2)

        return pred, ids_mask

