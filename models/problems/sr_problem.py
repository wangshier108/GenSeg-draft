from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from betty.problems import ImplicitProblem


class SRProblem(ImplicitProblem):
    """
    第二级问题：使用（真实 HR, 合成 LR）对训练 SR 网络。

    Data loader:
        hr_loader -> (hr_img, name)
    """

    def __init__(
        self,
        name: str,
        sr_net: nn.Module,
        deg_prior: nn.Module,
        deg_pipeline,
        optimizer: torch.optim.Optimizer,
        train_data_loader: Iterable,
        config,
        device: torch.device,
    ):
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
        self.deg_pipeline = deg_pipeline
        self.device = device

    def training_step(self, batch: Tuple[torch.Tensor, str]) -> torch.Tensor:
        hr, _ = batch
        hr = hr.to(self.device)

        with torch.no_grad():
            params = self.deg_prior(hr)
            lr = self.deg_pipeline(hr, params)

        sr_pred = self.sr_net(lr)

        # assume HR and SR have the same spatial size
        loss = F.l1_loss(sr_pred, hr)
        return loss


