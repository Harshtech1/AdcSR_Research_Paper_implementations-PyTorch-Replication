import torch
import torch.nn as nn

class AdcSRSelfAttention(nn.Module):
    """
    Sleek Self-Attention layer tracking Section 3.2.1.
    Allows image pixels to look at other image pixels for context,
    with channels pruned to 75% of Stable Diffusion's base dimensions.
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # 75% channel dimension projection layers
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj_out = nn.Linear(channels, channels)
        
    def forward(self, x):
        # x shape: [Batch, Tokens, Channels]
        b, t, c = x.shape
        
        # FIXED: Pass axis indices (0, 1, 2, 3, 4) instead of the actual dimension sizes
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Core Self-Attention Matrix calculations
        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_weights = attn_scores.softmax(dim=-1)
        
        out = (attn_weights @ v).permute(0, 2, 1, 3).reshape(b, t, c)
        return self.proj_out(out)

class AdcSRPrunedUNet(nn.Module):
    # in_channels is 12 (3 RGB channels * 2 * 2) because we dropped the VAE Encoder
    def __init__(self, in_channels=12, base_channels=240): 
        super().__init__()
        # Pixel Unshuffle increases input channels
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Pruned Denoising Blocks containing only convolutional features and self-attention
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        )
        
        # Positional spatial processing wrapper
        self.spatial_attn = AdcSRSelfAttention(channels=base_channels)
        
    def forward(self, x):
        # Expecting input shape from Pixel Unshuffle: [Batch, 12, 64, 64]
        h = self.init_conv(x)
        h = self.block1(h)
        
        # Reshape to token matrix sequences for self-attention block run
        b, c, w, h_spatial = h.shape
        tokens = h.flatten(2).transpose(1, 2) # [Batch, Tokens, Channels]
        
        attn_tokens = self.spatial_attn(tokens)
        
        # Recover structural visual matrix dimensions
        out_feat = attn_tokens.transpose(1, 2).reshape(b, c, w, h_spatial)
        return out_feat
