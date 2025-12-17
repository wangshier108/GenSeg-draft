from typing import Tuple

import torch
import torch.nn.functional as F


class ConditionedDegradation:
    """
    Differentiable degradation pipeline parameterized by continuous variables.

    Given an HR image and degradation parameters, produces a synthetic LR image.
    This corresponds to GenSeg 的“第一级”——构建 LR-HR 对作为第二级超分训练的补充数据。

    Current implementation:
        - Gaussian blur (controlled by blur_sigma)
        - Bicubic downsampling by a fixed scale
        - Additive Gaussian noise (noise_std)
    """

    def __init__(self, scale_factor: int = 4):
        self.scale_factor = scale_factor

    def _gaussian_kernel(self, sigma: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
        # sigma: (B,)  -> build per-sample kernel then stack
        device = sigma.device
        radius = kernel_size // 2
        x = torch.arange(-radius, radius + 1, device=device).float()
        gauss_1d = []
        for s in sigma:
            s = torch.clamp(s, min=1e-3)
            g = torch.exp(-0.5 * (x / s) ** 2)
            g = g / g.sum()
            gauss_1d.append(g)
        gauss_1d = torch.stack(gauss_1d, dim=0)  # (B, K)

        # separable 2D kernel
        kernel_2d = gauss_1d.unsqueeze(2) * gauss_1d.unsqueeze(1)  # (B, K, K)
        return kernel_2d

    def _apply_blur(self, x: torch.Tensor, blur_sigma: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        blur_sigma: (B,)
        """
        if torch.all(blur_sigma < 1e-3):
            return x

        B, C, H, W = x.shape
        kernel_2d = self._gaussian_kernel(blur_sigma)  # (B, K, K)
        K = kernel_2d.shape[-1]
        kernel_2d = kernel_2d.view(B, 1, K, K)
        kernel_2d = kernel_2d.repeat(1, C, 1, 1)

        x = x.view(1, B * C, H, W)
        kernel_2d = kernel_2d.view(B * C, 1, K, K)
        padding = K // 2
        x = F.conv2d(x, kernel_2d, padding=padding, groups=B * C)
        x = x.view(B, C, H, W)
        return x

    def __call__(self, hr: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hr: (B, 3, H, W) high-resolution tensor in [0, 1] or [-1, 1]
            params: (B, 2) with (blur_sigma, noise_std)
        Returns:
            lr: (B, 3, H/scale, W/scale)
        """
        blur_sigma, noise_std = params[:, 0], params[:, 1]

        x = hr
        x = self._apply_blur(x, blur_sigma)

        # Bicubic downsample
        x = F.interpolate(
            x,
            scale_factor=1.0 / float(self.scale_factor),
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=True,
        )

        # Additive Gaussian noise
        if torch.any(noise_std > 1e-6):
            B, C, H, W = x.shape
            noise = torch.randn_like(x)
            noise = noise * noise_std.view(B, 1, 1, 1)
            x = x + noise

        return x


