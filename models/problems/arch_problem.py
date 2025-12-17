from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from betty.problems import ImplicitProblem


class ArchProblem(ImplicitProblem):
    """
    第三级问题：根据验证集上的 SR 结果（合成 LR -> SR -> 对比 HR）
    来优化退化先验（deg_prior）的参数。

    Data loader:
        ValSRDataset -> (lr_img, name), (hr_img, name)
    """

    def __init__(
        self,
        name: str,
        sr_net: nn.Module,
        deg_prior: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_data_loader: Iterable,
        config,
        device: torch.device,
    ):
        # 注意：这里 module 设为 sr_net 以满足 Betty 的接口，
        # 但优化器可以绑定在退化先验的参数上，从而实现“通过验证损失更新第一级”。
        super().__init__(
            name=name,
            module=sr_net,
            optimizer=optimizer,
            train_data_loader=train_data_loader,
            config=config,
            device=device,
        )
        self.sr_net = sr_net
        self.deg_prior = deg_prior
        self.device = device

    def training_step(
        self,
        batch: Tuple[Tuple[torch.Tensor, str], Tuple[torch.Tensor, str]],
    ) -> torch.Tensor:
        (lr, _), (hr, _) = batch
        lr = lr.to(self.device)
        hr = hr.to(self.device)

        with torch.no_grad():
            sr_pred = self.sr_net(lr)

        # outer objective: validation reconstruction error
        loss_arch = F.l1_loss(sr_pred, hr)
        return loss_arch


