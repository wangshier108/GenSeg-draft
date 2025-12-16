from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from betty.problems import ImplicitProblem


class DegradationProblem(ImplicitProblem):
    """
    第一级问题：学习退化先验，使得从 HR 合成的 LR 能够逼近真实 LR。

    Data loader:
        zip(hr_loader, lr_loader)
        -> (hr_img, hr_name), (real_lr_img, lr_name)
    """

    def __init__(
        self,
        name: str,
        prior: nn.Module,
        pipeline,
        optimizer: torch.optim.Optimizer,
        train_data_loader: Iterable,
        config,
        device: torch.device,
    ):
        super().__init__(
            name=name,
            module=prior,
            optimizer=optimizer,
            train_data_loader=train_data_loader,
            config=config,
            device=device,
        )
        self.prior = prior
        self.pipeline = pipeline
        self.device = device

    def training_step(
        self,
        batch: Tuple[Tuple[torch.Tensor, str], Tuple[torch.Tensor, str]],
    ) -> torch.Tensor:
        (hr, _), (real_lr, _) = batch
        hr = hr.to(self.device)
        real_lr = real_lr.to(self.device)

        # predict degradation parameters from HR
        params = self.prior(hr)
        fake_lr = self.pipeline(hr, params)

        loss = F.l1_loss(fake_lr, real_lr)
        return loss


