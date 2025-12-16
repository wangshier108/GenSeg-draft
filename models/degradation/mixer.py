import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedOp(nn.Module):
    """
    一个简单的 DARTS 风格 mixed operation，用于退化先验网络中的 NAS。

    候选算子（可按需扩展/修改）：
        - 3x3 conv
        - 5x5 conv
        - 3x3 depthwise conv + 1x1 pointwise conv
        - 3x3 dilated conv
        - identity（1x1 conv 近似残差）
    """

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        C = channels
        self.ops = nn.ModuleList()

        # 3x3 conv
        self.ops.append(
            nn.Conv2d(C, C, kernel_size=3, stride=stride, padding=1, bias=False)
        )

        # 5x5 conv
        self.ops.append(
            nn.Conv2d(C, C, kernel_size=5, stride=stride, padding=2, bias=False)
        )

        # depthwise 3x3 + pointwise 1x1
        depthwise = nn.Conv2d(
            C, C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False
        )
        pointwise = nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0, bias=False)
        self.ops.append(nn.Sequential(depthwise, pointwise))

        # 3x3 dilated conv
        self.ops.append(
            nn.Conv2d(
                C,
                C,
                kernel_size=3,
                stride=stride,
                padding=2,
                dilation=2,
                bias=False,
            )
        )

        # identity（用 1x1 conv 近似，便于维度对齐）
        self.ops.append(
            nn.Conv2d(C, C, kernel_size=1, stride=stride, padding=0, bias=False)
        )

        # NAS 的架构参数（logits），一维向量
        self.alpha = nn.Parameter(torch.zeros(len(self.ops)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.alpha, dim=0)
        out = 0.0
        for w, op in zip(weights, self.ops):
            out = out + w * op(x)
        return out


