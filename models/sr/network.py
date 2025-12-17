import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSR(nn.Module):
    """
    A simple super-resolution backbone (SRCNN / ESPCN style).

    This is intentionally lightweight and meant to be a placeholder that you
    can replace with your own SR architecture (EDSR, RCAN, SwinIR, etc.).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor

        num_feats = 64
        self.conv1 = nn.Conv2d(in_channels, num_feats, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1)

        # upsample using PixelShuffle
        self.conv_up = nn.Conv2d(num_feats, num_feats * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

        self.conv_out = nn.Conv2d(num_feats, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if input LR resolution is not the desired one, you can still pass it;
        # model will upsample by scale_factor.
        feat = F.relu(self.conv1(x))
        feat = F.relu(self.conv2(feat))
        feat = self.conv_up(feat)
        feat = self.pixel_shuffle(feat)
        out = self.conv_out(feat)
        return out


