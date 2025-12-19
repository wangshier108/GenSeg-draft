import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedOp1D(nn.Module):
    """可微的离散选择（比如 JPEG QF 或噪声强度刻度），使用 softmax 权重做 convex combination."""

    def __init__(self, num_choices: int):
        super().__init__()
        self.num_choices = num_choices
        # 架构参数 alpha，会在 NAS 阶段被优化
        self.alpha = nn.Parameter(torch.zeros(num_choices))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """
        values: shape [num_choices] 或 [num_choices, ...]，对应每个离散选项的数值或张量.
        返回 softmax(alpha) 加权的 convex combination.
        """
        weights = F.softmax(self.alpha, dim=0)
        return torch.tensordot(weights, values, dims=1)


class MixedOpChannelWise(nn.Module):
    """对每个channel做可微选择，用在噪声强度刻度（σ/μ）。"""

    def __init__(self, num_channels: int, num_choices: int):
        super().__init__()
        self.num_channels = num_channels
        self.num_choices = num_choices
        # [C, K]
        self.alpha = nn.Parameter(torch.zeros(num_channels, num_choices))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """
        values: shape [C, K]，每个通道的 K 个候选标量.
        返回 shape [C] 的 convex combination.
        """
        assert values.shape == (self.num_channels, self.num_choices)
        weights = F.softmax(self.alpha, dim=-1)  # [C, K]
        out = (weights * values).sum(dim=-1)
        return out


