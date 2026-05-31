import torch
import torch.nn as nn

class PrunedDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.norm(self.conv2(h))
        return self.act(h + self.shortcut(x))

class AdcSRPrunedDecoder(nn.Module):
    def __init__(self, unet_feat_channels=240, base_channels=128):
        super().__init__()
        self.unet_bridge = nn.Conv2d(unet_feat_channels, base_channels * 2, kernel_size=3, padding=1)
        self.block1 = PrunedDecoderBlock(base_channels * 2, base_channels * 2)
        # CHANGED: Upsample scale_factor is now 8 so 64x64 -> 512x512
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False) 
        self.block2 = PrunedDecoderBlock(base_channels * 2, base_channels)
        self.to_rgb = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

    def forward(self, unet_features):
        h = self.unet_bridge(unet_features)
        h = self.block1(h)
        h = self.upsample(h)
        h = self.block2(h)
        return self.to_rgb(h)
