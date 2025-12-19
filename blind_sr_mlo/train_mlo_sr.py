"""
三层嵌套优化的盲超分训练脚本（简化版）

Stage ①: 退化生成器 G 预训练 (HR -> LR_syn)，对齐真实 LR_real 分布
Stage ②: 超分网络 S 训练 (LR_syn/LR_real -> SR)，使用 L1+LPIPS+GAN
Stage ③: 使用 Betty 做 MLO/NAS，在验证集上最小化 L_val，引导架构参数 alpha 与 G/S 协同优化

本文件给出 Betty 集成的 Stage ③ 主体结构，Stage ①/② 可在此脚本中封装为不同模式，也可以单独脚本实现。
"""

from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem

from blind_sr_mlo.models import DegradationGenerator, EDSR


class PatchDiscriminator(nn.Module):
    """简单 PatchGAN 判别器，用于退化与 SR 的 GAN loss."""

    def __init__(self, in_ch: int = 3, base_ch: int = 64, num_layers: int = 3):
        super().__init__()
        layers = []
        ch = base_ch
        layers += [
            nn.Conv2d(in_ch, ch, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(1, num_layers):
            layers += [
                nn.Conv2d(ch, ch * 2, 5, 2, 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch *= 2
        layers += [nn.Conv2d(ch, 1, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VGGPerceptualLoss(nn.Module):
    """用 VGG 特征近似 LPIPS，计算感知损失."""

    def __init__(self, layer: str = "relu3_3"):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        # 截取到 relu3_3（第 16 层左右），具体索引按 vgg16 结构确定
        self.features = nn.Sequential(*list(vgg.children())[:16])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        # x,y: [B,3,H,W], 0-1
        def norm(im):
            mean = torch.tensor([0.485, 0.456, 0.406], device=im.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=im.device).view(1, 3, 1, 1)
            return (im - mean) / std

        x_f = self.features(norm(x))
        y_f = self.features(norm(y))
        return torch.mean((x_f - y_f) ** 2)


class DegProblem(ImplicitProblem):
    """退化生成器 G 的 inner problem（可选 PatchGAN 判别器，可按需扩展）"""

    def __init__(self, *args, discriminator: nn.Module = None, optim_D: torch.optim.Optimizer = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.discriminator = discriminator
        self.optim_D = optim_D
        self.l1 = nn.L1Loss()
        self.adv_loss = nn.BCEWithLogitsLoss()

    def training_step(self, batch: Dict[str, Any]):
        hr = batch["hr"].to(self.device)
        lr_real = batch["lr"].to(self.device)
        lr_syn, k, sigma, mu, qf = self.module(hr)
        loss_l1 = self.l1(lr_syn, lr_real)
        # kernel 正则（近似稀疏+光滑）
        k_grad_h = k[:, :, 1:, :] - k[:, :, :-1, :]
        k_grad_w = k[:, :, :, 1:] - k[:, :, :, :-1]
        loss_k = torch.norm(k, p=2) + 0.1 * (torch.norm(k_grad_h, p=2) + torch.norm(k_grad_w, p=2))
        # 噪声 TV 正则
        n = sigma + mu
        n_grad_h = n[:, :, 1:, :] - n[:, :, :-1, :]
        n_grad_w = n[:, :, :, 1:] - n[:, :, :, :-1]
        loss_tv = 0.05 * (torch.mean(torch.abs(n_grad_h)) + torch.mean(torch.abs(n_grad_w)))

        # PatchGAN: 更新判别器
        if self.discriminator is not None and self.optim_D is not None:
            self.discriminator.train()
            self.optim_D.zero_grad()
            with torch.no_grad():
                lr_syn_detach = lr_syn.detach()
            pred_real = self.discriminator(lr_real)
            pred_fake = self.discriminator(lr_syn_detach)
            real_labels = torch.ones_like(pred_real)
            fake_labels = torch.zeros_like(pred_fake)
            loss_D = 0.5 * (
                self.adv_loss(pred_real, real_labels) + self.adv_loss(pred_fake, fake_labels)
            )
            loss_D.backward(retain_graph=True)
            self.optim_D.step()

            # 对 G 的对抗损失
            pred_fake_for_G = self.discriminator(lr_syn)
            loss_G_adv = self.adv_loss(pred_fake_for_G, real_labels)
        else:
            loss_G_adv = 0.0

        loss = loss_l1 + loss_k + loss_tv + 0.01 * loss_G_adv
        return loss


class SRProblem(ImplicitProblem):
    """超分网络 S 的 inner problem."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = nn.L1Loss()
        self.perc = VGGPerceptualLoss()
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.discriminator = None
        self.optim_D = None

    def set_discriminator(self, discriminator: nn.Module, optim_D: torch.optim.Optimizer):
        self.discriminator = discriminator
        self.optim_D = optim_D

    def training_step(self, batch: Dict[str, Any]):
        lr = batch["lr"].to(self.device)
        hr = batch["hr"].to(self.device)
        sr = self.module(lr)

        # L1
        l_l1 = self.l1(sr, hr)
        # 感知损失（近似 LPIPS）
        l_perc = self.perc(sr, hr)

        # PatchGAN 判别器更新
        if self.discriminator is not None and self.optim_D is not None:
            self.discriminator.train()
            self.optim_D.zero_grad()
            with torch.no_grad():
                sr_detach = sr.detach()
            pred_real = self.discriminator(hr)
            pred_fake = self.discriminator(sr_detach)
            real_labels = torch.ones_like(pred_real)
            fake_labels = torch.zeros_like(pred_fake)
            loss_D = 0.5 * (
                self.adv_loss(pred_real, real_labels) + self.adv_loss(pred_fake, fake_labels)
            )
            loss_D.backward(retain_graph=True)
            self.optim_D.step()

            pred_fake_for_G = self.discriminator(sr)
            l_gan = self.adv_loss(pred_fake_for_G, real_labels)
        else:
            l_gan = 0.0

        # L② = λ1 L1 + λ2 LPIPS + λ3 LGAN
        loss = 1.0 * l_l1 + 0.1 * l_perc + 0.01 * l_gan
        return loss


class ArchProblem(ImplicitProblem):
    """外层 NAS 问题：在验证集上最小化 L_val，对架构参数（G 中的 MixedOp 等）做优化。"""

    def __init__(self, *args, sr_model: nn.Module = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sr_model = sr_model
        self.l1 = nn.L1Loss()
        self.perc = VGGPerceptualLoss()

    def training_step(self, batch: Dict[str, Any]):
        hr = batch["hr"].to(self.device)
        # 使用当前退化生成器产生 LR_syn
        lr_syn, _, _, _, _ = self.module(hr)
        with torch.no_grad():
            sr = self.sr_model(lr_syn)
        l_l1 = self.l1(sr, hr)
        l_perc = self.perc(sr, hr)
        # L_val = L1 + 0.1 LPIPS（不含 GAN）
        loss = l_l1 + 0.1 * l_perc
        return loss


class MLOEngine(Engine):
    pass


def build_mlo_engine(train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
    # 退化生成器 G
    G = DegradationGenerator().to(device)
    optim_G = torch.optim.Adam(G.parameters(), lr=1e-4)

    # 退化 PatchGAN 判别器
    D_deg = PatchDiscriminator(in_ch=3).to(device)
    optim_D_deg = torch.optim.Adam(D_deg.parameters(), lr=1e-4)

    # 超分网络 S
    S = EDSR(scale=4).to(device)
    optim_S = torch.optim.Adam(S.parameters(), lr=1e-4)

    # 超分 PatchGAN 判别器
    D_sr = PatchDiscriminator(in_ch=3).to(device)
    optim_D_sr = torch.optim.Adam(D_sr.parameters(), lr=1e-4)

    # 架构参数优化器：这里只简单地对 G 中所有 MixedOp 参数做优化
    arch_params = []
    for m in G.modules():
        if hasattr(m, "alpha"):
            arch_params.append(m.alpha)
    optim_arch = torch.optim.Adam(arch_params, lr=3e-4)

    inner_cfg = Config(type="darts", unroll_steps=1)
    outer_cfg = Config(retain_graph=True)
    engine_cfg = EngineConfig(
        train_iters=10000,
        valid_step=200,
        roll_back=True,
    )

    deg_problem = DegProblem(
        name="deg",
        module=G,
        optimizer=optim_G,
        train_data_loader=train_loader,
        config=inner_cfg,
        device=device,
        discriminator=D_deg,
        optim_D=optim_D_deg,
    )
    sr_problem = SRProblem(
        name="sr",
        module=S,
        optimizer=optim_S,
        train_data_loader=train_loader,
        config=inner_cfg,
        device=device,
    )
    sr_problem.set_discriminator(D_sr, optim_D_sr)
    arch_problem = ArchProblem(
        name="arch",
        module=G,
        optimizer=optim_arch,
        train_data_loader=val_loader,
        config=outer_cfg,
        device=device,
        sr_model=S,
    )

    problems = [deg_problem, sr_problem, arch_problem]
    # 依赖关系：G/S 作为 lower，Arch 作为 upper
    l2u = {deg_problem: [arch_problem], sr_problem: [arch_problem]}
    u2l = {arch_problem: [deg_problem, sr_problem]}
    dependencies = {"l2u": l2u, "u2l": u2l}

    engine = MLOEngine(config=engine_cfg, problems=problems, dependencies=dependencies)
    return engine


def main():
    # 这里只放占位的数据加载逻辑，实际使用时请替换为 DIV2K/Flickr2K/RealSR 等数据集的 DataLoader.
    # 为了保证脚本可运行，我们使用随机张量的 DummyDataset.
    from torch.utils.data import Dataset


    class DummySRDataset(Dataset):
        def __init__(self, length=1000, hr_size=(256, 256)):
            self.length = length
            self.hr_size = hr_size

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            h, w = self.hr_size
            hr = torch.rand(3, h, w)
            # 这里先简单生成一个下采样 LR，真实场景中会用真实 LR_real
            lr = nn.functional.interpolate(hr.unsqueeze(0), scale_factor=0.25, mode="bicubic", align_corners=False).squeeze(0)
            return {"hr": hr, "lr": lr}


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = DummySRDataset(length=500)
    val_set = DummySRDataset(length=50)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    engine = build_mlo_engine(train_loader, val_loader, device)
    engine.run()


if __name__ == "__main__":
    main()


