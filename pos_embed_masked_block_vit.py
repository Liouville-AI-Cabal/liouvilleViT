import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


# --- Custom Multihead Attention with Relative Positional Biases ---
# Simplified T5-style relative positional encoding

def _get_rel_pos_bias_bucket(relative_position, num_buckets, max_distance):
    """
    Adapted from T5 relative positional encoding.
    Maps relative positions to a bucket index.
    """
    is_negative = (relative_position < 0).to(torch.int32)
    relative_position = torch.abs(relative_position)
    # Clamp to max_distance to avoid issues with large relative positions
    relative_position = torch.clamp(relative_position, 0, max_distance - 1)

    # These constants (32, max_distance / 2) are from T5 implementation
    # You might need to tune them based on your grid size and desired bucket distribution.
    max_exact = num_buckets // 2
    is_power2 = (relative_position >= max_exact).to(torch.int32)
    bucket = is_negative * num_buckets  # Shift for negative values
    bucket += torch.where(
        relative_position < max_exact,
        relative_position,
        (max_exact + (torch.log(relative_position / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).to(torch.int32))
    )
    return bucket

class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, grid_size=100,
                 num_buckets=32, max_distance=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.scaling = self.head_dim ** -0.5

        # Relative positional bias parameters
        self.grid_size = grid_size
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads) # Bias per head

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        B, N, D = query.shape # N is sequence length (num_tokens)

        # In-projection for Q, K, V
        q, k, v = self.in_proj(query).chunk(3, dim=-1)
        q = q.contiguous().view(B, N, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, N, head_dim)
        k = k.contiguous().view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.contiguous().view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling # (B, num_heads, N, N)

        # --- Add Relative Positional Biases ---
        # Calculate relative positions
        # Create a tensor of indices (0 to N-1) for rows and columns
        indices = torch.arange(N, device=query.device)
        row_indices = indices // self.grid_size
        col_indices = indices % self.grid_size

        # Compute relative row and column distances
        # (N, 1) - (1, N) -> (N, N)
        relative_row_dist = row_indices.unsqueeze(1) - row_indices.unsqueeze(0)
        relative_col_dist = col_indices.unsqueeze(1) - col_indices.unsqueeze(0)

        # Map to buckets
        row_bucket = _get_rel_pos_bias_bucket(relative_row_dist, self.num_buckets, self.max_distance)
        col_bucket = _get_rel_pos_bias_bucket(relative_col_dist, self.num_buckets, self.max_distance)

        # Retrieve biases
        row_bias = self.relative_attention_bias(row_bucket) # (N, N, num_heads)
        col_bias = self.relative_attention_bias(col_bucket) # (N, N, num_heads)

        # Combine biases and add to attention weights
        # (N, N, num_heads) -> (num_heads, N, N) for broadcasting
        relative_bias = (row_bias + col_bias).permute(2, 0, 1).unsqueeze(0) # (1, num_heads, N, N)
        attn_weights = attn_weights + relative_bias

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))

        if key_padding_mask is not None:
            # key_padding_mask: (B, N) -> (B, 1, 1, N)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, v) # (B, num_heads, N, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights # Also return weights if needed elsewhere


# --- Custom Transformer Encoder Layer ---
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 batch_first=False, norm_first=False, grid_size=100):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout,
                                                  grid_size=grid_size)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            # Pre-norm architecture
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            # Post-norm architecture
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        attn_output, _ = self.self_attn(x, x, x,
                                       attn_mask=attn_mask,
                                       key_padding_mask=key_padding_mask,
                                       need_weights=False)
        return self.dropout1(attn_output)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


# --- MaskedBlockViT with Custom Transformer Layer ---
class MaskedBlockViT(nn.Module):
    def __init__(self, grid_size=100, embed_dim=128, depth=8, num_heads=8, mask_ratio=0.3):
        super().__init__()
        self.grid_size = grid_size
        self.num_tokens = grid_size ** 2
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        # CNN block encoder
        self.block_encoder = CNNBlockEncoder(out_dim=embed_dim)

        # 2D Learnable Positional encoding
        self.row_embed = nn.Parameter(torch.randn(grid_size, embed_dim))
        self.col_embed = nn.Parameter(torch.randn(grid_size, embed_dim))

        # Transformer encoder (using custom layer with relative attention)
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            grid_size=grid_size # Pass grid_size to the custom layer
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Decoder: map embedding back to full block using convolutions
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
        # x_blocks: [B, 10000, 2, 50, 50]
        B, N, C, H, W = x_blocks.shape

        x = x_blocks.view(-1, C, H, W)  # (B*N, 2, 50, 50)
        x_emb = self.block_encoder(x).view(B, N, -1)  # (B, 10000, embed_dim)

        # Generate 2D positional embeddings by combining row and column embeddings
        row_embed_expanded = self.row_embed.unsqueeze(1)
        col_embed_expanded = self.col_embed.unsqueeze(0)
        pos_embed_2d = (row_embed_expanded + col_embed_expanded).view(1, self.num_tokens, self.embed_dim)

        x_emb = x_emb + pos_embed_2d[:, :N, :] # Add 2D positional embeddings

        x_masked, ids_keep, ids_mask, ids_restore = self.random_mask(x_emb, self.mask_ratio)

        encoded = self.transformer(x_masked)

        # Prepare full sequence to decode
        B, L, D_encoded = encoded.shape
        decoder_input = torch.zeros(B, N, D_encoded, device=x.device)
        decoder_input.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, D_encoded), encoded)

        # Apply linear projection for each block's embedding
        linear_decoded = self.linear_decoder_projection(decoder_input) # (B, N, 64 * 13 * 13)

        # Reshape for convolutional decoder input: (B*N, 64, 13, 13)
        conv_input = linear_decoded.view(B * N, 64, 13, 13)

        # Apply convolutional decoder to reconstruct blocks
        pred_flat_conv = self.conv_decoder(conv_input) # (B*N, 2, 50, 50)

        # Reshape back to original (B, N, 50, 50, 2)
        pred = pred_flat_conv.permute(0, 2, 3, 1).view(B, N, 50, 50, 2)
        return pred, ids_mask

