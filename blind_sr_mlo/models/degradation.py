import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixed_ops import MixedOp1D, MixedOpChannelWise


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = conv3x3(channels, channels)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = conv3x3(channels, channels)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.in2(out)
        return self.act(out + identity)


class SharedEncoder(nn.Module):
    """
    HR(3x256x256) -> feature(256x64x64)
    conv1(7x7,64,s=1) + IN + ReLU
    ResBlock x3 (64)
    conv2(3x3,128,s=2) + IN + ReLU
    ResBlock x3 (128)
    conv3(3x3,256,s=2) + IN + ReLU
    ResBlock x3 (256)
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, base_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.in1 = nn.InstanceNorm2d(base_ch, affine=True)
        self.rb1 = nn.Sequential(*[ResBlock(base_ch) for _ in range(3)])

        self.conv2 = conv3x3(base_ch, base_ch * 2, stride=2)
        self.in2 = nn.InstanceNorm2d(base_ch * 2, affine=True)
        self.rb2 = nn.Sequential(*[ResBlock(base_ch * 2) for _ in range(3)])

        self.conv3 = conv3x3(base_ch * 2, base_ch * 4, stride=2)
        self.in3 = nn.InstanceNorm2d(base_ch * 4, affine=True)
        self.rb3 = nn.Sequential(*[ResBlock(base_ch * 4) for _ in range(3)])

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.in1(x)
        x = self.act(x)
        x = self.rb1(x)

        x = self.conv2(x)
        x = self.in2(x)
        x = self.act(x)
        x = self.rb2(x)

        x = self.conv3(x)
        x = self.in3(x)
        x = self.act(x)
        x = self.rb3(x)
        return x  # [B, 256, 64, 64]


class BlurHead(nn.Module):
    """
    feature(256x64x64) -> blur kernel k (max 25x25)
    使用 GlobalAvgPool + FC 生成 base kernel，再通过 3 个不同 K 的 MixedOp 组合。
    这里实现为一个固定 25x25 kernel，并通过 mixed-ops 选择实际的有效尺寸。
    """

    def __init__(self, feat_ch: int = 256, ks_choices: List[int] = [7, 15, 25]):
        super().__init__()
        self.ks_choices = ks_choices
        self.max_k = max(ks_choices)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(feat_ch, 128)
        self.fc2 = nn.Linear(128, self.max_k * self.max_k)
        self.act = nn.ReLU(inplace=True)

        # 对 kernel 尺度做可微选择
        self.mixed_ks = MixedOp1D(num_choices=len(ks_choices))

    def forward(self, feat):
        b, c, _, _ = feat.shape
        x = self.gap(feat).view(b, c)
        x = self.act(self.fc1(x))
        base = self.fc2(x)  # [B, max_k*max_k]
        base = base.view(b, 1, self.max_k, self.max_k)

        # 为每个 ks 生成一个 zero-padded kernel，并通过 mixed_ks 组合
        kernels = []
        for ks in self.ks_choices:
            if ks == self.max_k:
                k = base
            else:
                # 中心 crop 成 ks，再 pad 回 max_k
                off = (self.max_k - ks) // 2
                k_center = base[:, :, off:off + ks, off:off + ks]
                pad = (off, self.max_k - ks - off, off, self.max_k - ks - off)
                k = F.pad(k_center, pad)
            kernels.append(k)

        # [num_choices, B, 1, max_k, max_k] -> [num_choices, B, ...]
        stack = torch.stack(kernels, dim=0)
        # MixedOp1D 目前按 [num_choices, ...] 聚合成 [...]
        # 这里对 batch 维一视同仁，简化实现
        stack_flat = stack.view(len(self.ks_choices), -1)
        mixed = self.mixed_ks(stack_flat).view(b, 1, self.max_k, self.max_k)

        # softmax 归一化，确保非负且和为1
        mixed = F.softmax(mixed.view(b, -1), dim=-1).view_as(mixed)
        return mixed  # [B, 1, max_k, max_k]


class NoiseHead(nn.Module):
    """
    feature(256x64x64) -> σ(x), μ(x) 两张图，通过像素级回归 + 通道级 MixedOp 控制强度刻度.
    """

    def __init__(self, feat_ch: int = 256, num_scales: int = 8):
        super().__init__()
        # backbone 生成 2-channel map
        self.conv1 = nn.Conv2d(feat_ch, 64, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.softplus = nn.Softplus()

        # 每个 channel 上的强度刻度 MixedOp（σ，μ 两个通道）
        self.mixed_scale = MixedOpChannelWise(num_channels=2, num_choices=num_scales)
        # 预定义离散刻度（0, 0.01, ..., 0.07）
        scales = torch.linspace(0.0, 0.07, steps=num_scales)
        self.register_buffer("scale_values", scales)

    def forward(self, feat, hr_size):
        b, _, _, _ = feat.shape
        h, w = hr_size

        x = self.conv1(feat)
        x = self.act(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = self.conv2(x)
        x = self.act(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = self.conv_out(x)  # [B, 2, H, W] with H,W~256
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = self.softplus(x)  # 保证非负

        # 通道级刻度选择
        # values: [C, K]
        values = torch.stack([self.scale_values, self.scale_values], dim=0)
        scales = self.mixed_scale(values)  # [2]
        # σ, μ 两个通道分别缩放
        sigma = torch.clamp(x[:, 0:1] * scales[0], 0.0, 0.06)
        mu = torch.clamp(x[:, 1:2] * scales[1], 0.0, 0.12)
        return sigma, mu


class CompressionHead(nn.Module):
    """
    feature -> JPEG QF (离散表的 convex combination).
    """

    def __init__(self, feat_ch: int = 256, qf_table: List[int] = None):
        super().__init__()
        if qf_table is None:
            qf_table = list(range(50, 100, 10))  # [50, 60, ..., 90]
        self.qf_table = qf_table
        self.num_qf = len(qf_table)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feat_ch, self.num_qf)
        self.mixed_qf = MixedOp1D(num_choices=self.num_qf)

        # 注册成 buffer，便于在 GPU 上使用
        self.register_buffer("qf_values", torch.tensor(qf_table, dtype=torch.float32))

    def forward(self, feat):
        b, c, _, _ = feat.shape
        x = self.gap(feat).view(b, c)
        logits = self.fc(x)  # [B, num_qf]
        weights = torch.softmax(logits, dim=-1)  # [B, num_qf]
        # 用 batch 平均的权重近似，简化 MixedOp 使用（你也可以改成逐样本 MixedOp）
        avg_w = weights.mean(dim=0)
        qf = (avg_w * self.qf_values).sum()
        return qf


class DifferentiableDegradation(nn.Module):
    """
    接收 HR 和 (k, sigma, mu, QF) 组合，输出 LR 图像.
    这里提供一个可微近似版本：
      - blur: conv2d
      - noise: Poisson + Gaussian 近似
      - JPEG: 用 smooth quantization 模拟（若安装了 diffjpeg，可在此替换为真实 Differentiable JPEG）
    """

    def __init__(self, max_kernel_size: int):
        super().__init__()
        self.max_kernel_size = max_kernel_size

    def forward(self, hr, kernel, sigma, mu, qf):
        """
        hr: [B,3,H,W], 0-1
        kernel: [B,1,K,K]
        sigma, mu: [B,1,H,W]
        qf: scalar tensor
        """
        b, c, h, w = hr.shape

        # 1) blur with reflection padding
        pad = self.max_kernel_size // 2
        hr_pad = F.pad(hr, (pad, pad, pad, pad), mode="reflect")
        # 将 kernel 应用于每个通道
        kernel = kernel.repeat(1, c, 1, 1)  # [B,C,K,K]
        kernel = kernel.view(b * c, 1, self.max_kernel_size, self.max_kernel_size)
        hr_reshaped = hr_pad.view(1, b * c, h + 2 * pad, w + 2 * pad)
        hr_blur = F.conv2d(hr_reshaped, kernel, groups=b * c)
        hr_blur = hr_blur.view(b, c, h, w)

        # 2) noise: Poisson + Gaussian 近似（用 reparameterization trick 保留梯度）
        # 这里只做一个平滑近似：hr_blur + mu * hr_blur + sigma * eps
        eps = torch.randn_like(hr_blur)
        hr_noisy = hr_blur + mu * hr_blur + sigma * eps
        hr_noisy = torch.clamp(hr_noisy, 0.0, 1.0)

        # 3) JPEG: 简化为基于 qf 的平滑量化（可替换为 diffjpeg.DiffJPEG）
        # qf in [50,95] 近似到 [0,1] 量化强度
        qf_norm = 1.0 - (qf - 50.0) / (95.0 - 50.0 + 1e-6)
        step = 1.0 / (1.0 + 50.0 * qf_norm)  # [大步长 -> 更强压缩]
        hr_scaled = hr_noisy / step
        # smooth rounding
        hr_rounded = hr_scaled - torch.sin(2 * math.pi * hr_scaled) / (2 * math.pi)
        hr_compressed = torch.clamp(hr_rounded * step, 0.0, 1.0)

        return hr_compressed


class DegradationGenerator(nn.Module):
    """
    整体退化生成器 G: HR -> (k, n, QF) -> LR
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 64, ks_choices: List[int] = None, qf_table: List[int] = None):
        super().__init__()
        if ks_choices is None:
            ks_choices = [7, 15, 25]
        self.encoder = SharedEncoder(in_ch=in_ch, base_ch=base_ch)
        self.blur_head = BlurHead(feat_ch=base_ch * 4, ks_choices=ks_choices)
        self.noise_head = NoiseHead(feat_ch=base_ch * 4)
        self.comp_head = CompressionHead(feat_ch=base_ch * 4, qf_table=qf_table)
        self.degradation = DifferentiableDegradation(max_kernel_size=max(ks_choices))

    def forward(self, hr):
        feat = self.encoder(hr)
        kernel = self.blur_head(feat)
        # print("why degra   00000")
        sigma, mu = self.noise_head(feat, hr_size=hr.shape[-2:])
        qf = self.comp_head(feat)
        # print("why degra   11111")
        lr = self.degradation(hr, kernel, sigma, mu, qf)
        # print("why degra   22222")
        return lr, kernel, sigma, mu, qf


