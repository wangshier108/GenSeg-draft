import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixer import MixedOp


class DegradationPriorNet(nn.Module):
    """
    由多个 mixerop 组成的退化先验网络，用于预测退化参数（供 ConditionedDegradation 使用）。

    Example parameterization (continuous):
        - blur_sigma: strength of Gaussian blur
        - noise_std: standard deviation of additive Gaussian noise

    这里的 NAS 通过 MixedOp 中的 alpha 实现，ArchProblem 将只更新这些 alpha。
    """

    def __init__(self, in_channels: int = 3, num_feats: int = 32, num_layers: int = 3):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, num_feats, kernel_size=3, padding=1, bias=False)

        # 一串 MixedOp，用于做神经架构搜索
        self.mixer_layers = nn.ModuleList(
            [MixedOp(num_feats, stride=1) for _ in range(num_layers)]
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        # 2 parameters: blur_sigma (>=0), noise_std (>=0)
        self.fc = nn.Linear(num_feats, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = F.relu(self.stem(x))
        for mixer in self.mixer_layers:
            feat = F.relu(mixer(feat))
        feat = self.pool(feat).flatten(1)
        params = self.fc(feat)
        # enforce non-negativity with softplus
        params = F.softplus(params)
        return params

    def arch_parameters(self):
        """
        返回 NAS/MixedOp 的架构参数 alpha，用于 ArchProblem 的优化器。
        """
        arch_params = []
        for mixer in self.mixer_layers:
            # 每个 MixedOp 里只有一个 alpha 参数
            arch_params.append(mixer.alpha)
        return arch_params
