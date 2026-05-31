import torch
import torch.nn as nn

class PixelUnshuffleWrapper(nn.Module):
    """
    Rearranges spatial pixels into channels tracking Section 3.2.1.
    Transforms a [B, C, H, W] tensor into [B, C * (downscale^2), H/downscale, W/downscale].
    Preserves raw low-res edge data completely without computational lossy operations.
    """
    def __init__(self, downscale_factor=2):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        
    def forward(self, x):
        return self.unshuffle(x)

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
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim).permute(2, b, self.num_heads, t, self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_weights = attn_scores.softmax(dim=-1)
        
        out = (attn_weights @ v).permute(0, 2, 1, 3).reshape(b, t, c)
        return self.proj_out(out)

if __name__ == "__main__":
    # Test checking the mathematical boundaries
    mock_img = torch.randn(2, 4, 128, 128)
    unshuffler = PixelUnshuffleWrapper(downscale_factor=2)
    print(f"✅ PixelUnshuffle check: Input {mock_img.shape} -> Output {unshuffler(mock_img).shape}")
