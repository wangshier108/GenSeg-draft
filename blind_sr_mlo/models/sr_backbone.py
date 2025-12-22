import math

import torch
import torch.nn as nn


class EDSRResBlock(nn.Module):
    """EDSR 中的残差块：Conv-ReLU-Conv + res_scale, 无 BN。"""

    def __init__(self, channels: int, res_scale: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)
        self.res_scale = res_scale

    def forward(self, x):
        identity = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return identity + out * self.res_scale


class Upsampler(nn.Sequential):
    """来自 EDSR 的上采样模块，支持 2^n 或 3 倍放大。"""

    def __init__(self, scale, n_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m += [
                    nn.Conv2d(n_feat, 4 * n_feat, 3, 1, 1),
                    nn.PixelShuffle(2),
                ]
        elif scale == 3:
            m += [
                nn.Conv2d(n_feat, 9 * n_feat, 3, 1, 1),
                nn.PixelShuffle(3),
            ]
        else:
            raise NotImplementedError(f"Unsupported scale {scale}")
        super().__init__(*m)


class EDSR(nn.Module):
    """
    EDSR 超分网络实现（简化版），默认配置接近 EDSR-baseline。
    输入: LR (B,3,H,W)
    输出: SR (B,3,scale*H,scale*W)
    """

    def __init__(
        self,
        scale: int = 4,
        n_resblocks: int = 16,
        n_feats: int = 64,
        res_scale: float = 0.1,
    ):
        super().__init__()
        self.scale = scale

        # head
        self.head = nn.Conv2d(3, n_feats, 3, 1, 1)

        # body
        body = [EDSRResBlock(n_feats, res_scale=res_scale) for _ in range(n_resblocks)]
        body.append(nn.Conv2d(n_feats, n_feats, 3, 1, 1))
        self.body = nn.Sequential(*body)

        # upsampler + tail
        self.upsampler = Upsampler(scale, n_feats)
        self.tail = nn.Conv2d(n_feats, 3, 3, 1, 1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        # print("why sr   00000")
        x = x + res
        x = self.upsampler(x)
        # print("why sr   11111")
        x = self.tail(x)
        # print("why sr   22222")
        return x
