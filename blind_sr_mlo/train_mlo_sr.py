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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import math

def _gaussian_window(window_size: int, sigma: float, device: torch.device):
    gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)], device=device)
    gauss = gauss / gauss.sum()
    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    return window_2d


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5, eps: float = 1e-8) -> torch.Tensor:
    """Compute SSIM for 3-channel images in range [0,1]."""
    device = img1.device
    window = _gaussian_window(window_size, sigma, device=device).expand(3, 1, window_size, window_size)
    padding = window_size // 2

    mu1 = F.conv2d(img1, window, padding=padding, groups=3)
    mu2 = F.conv2d(img2, window, padding=padding, groups=3)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=3) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=3) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps)
    return ssim_map.mean()

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
        # 尝试加载预训练权重；若环境离线或下载失败，则回退为随机初始化，避免阻塞
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        except Exception:
            vgg = models.vgg16(weights=None).features
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
        # Ensure VGG feature extractor is on the same device as inputs
        if next(self.features.parameters()).device != x.device:
            self.features = self.features.to(x.device)

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
        # Ensure lr_syn matches lr_real spatial size for loss/discriminator
        if lr_syn.shape[-2:] != lr_real.shape[-2:]:
            lr_syn = nn.functional.interpolate(
                lr_syn, size=lr_real.shape[-2:], mode="bicubic", align_corners=False
            )
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
            # 仅用于更新判别器，无需保留计算图，避免内存增长导致速度变慢
            loss_D.backward(retain_graph=False)
            self.optim_D.step()

            # 对 G 的对抗损失
            pred_fake_for_G = self.discriminator(lr_syn)
            loss_G_adv = self.adv_loss(pred_fake_for_G, real_labels)
        else:
            loss_G_adv = 0.0

        loss = loss_l1 + loss_k + loss_tv + 0.01 * loss_G_adv
        # 记录最近的损失用于进度打印
        try:
            gan_val = loss_G_adv.item() if hasattr(loss_G_adv, "item") else float(loss_G_adv)
        except Exception:
            gan_val = 0.0
        self._last_metrics = {
            "l1": float(loss_l1.item()),
            "k_reg": float(loss_k.item()),
            "tv": float(loss_tv.item()),
            "gan": gan_val,
            "total": float(loss.item()),
        }
        return loss


class SRProblem(ImplicitProblem):
    """超分网络 S 的 inner problem."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = nn.L1Loss()
        self.perc = VGGPerceptualLoss().to(self.device)
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

        # PSNR / SSIM 计算（数据范围假定 0-1）
        with torch.no_grad():
            mse = torch.mean((sr.clamp(0, 1) - hr.clamp(0, 1)) ** 2)
            psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
            # SSIM 使用固定窗口，避免额外依赖
            ssim = _ssim(sr.clamp(0, 1), hr.clamp(0, 1))

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
        # 记录最近的损失用于进度打印
        try:
            gan_val = l_gan.item() if hasattr(l_gan, "item") else float(l_gan)
        except Exception:
            gan_val = 0.0
        self._last_metrics = {
            "l1": float(l_l1.item()),
            "perc": float(l_perc.item()),
            "gan": gan_val,
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "total": float(loss.item()),
        }
        return loss


class ArchProblem(ImplicitProblem):
    """外层 NAS 问题：在验证集上最小化 L_val，对架构参数（G 中的 MixedOp 等）做优化。"""

    def __init__(self, *args, sr_model: nn.Module = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sr_model = sr_model
        self.l1 = nn.L1Loss()
        self.perc = VGGPerceptualLoss().to(self.device)

    def training_step(self, batch: Dict[str, Any]):
        hr = batch["hr"].to(self.device)
        # 使用当前退化生成器产生 LR_syn
        lr_syn, _, _, _, _ = self.module(hr)
        # Try to match LR_syn to the real LR size if available; otherwise infer by x4 downscale
        target_size = None
        if "lr" in batch:
            lr_real = batch["lr"].to(self.device)
            target_size = lr_real.shape[-2:]
        else:
            h, w = hr.shape[-2], hr.shape[-1]
            # Fallback assumes scale=4 as used by EDSR in this script
            target_size = (h // 4, w // 4)
        if lr_syn.shape[-2:] != target_size:
            lr_syn = nn.functional.interpolate(
                lr_syn, size=target_size, mode="bicubic", align_corners=False
            )
        # 不使用 no_grad，避免打断梯度从验证损失回传到退化生成器的架构参数
        self.sr_model.eval()
        sr = self.sr_model(lr_syn)
        l_l1 = self.l1(sr, hr)
        l_perc = self.perc(sr, hr)
        # L_val = L1 + 0.1 LPIPS（不含 GAN）
        loss = l_l1 + 0.1 * l_perc
        # 记录最近的验证损失用于进度打印
        self._last_metrics = {
            "val_l1": float(l_l1.item()),
            "val_perc": float(l_perc.item()),
            "val_total": float(loss.item()),
        }
        return loss


import os
from collections import deque


class MLOEngine(Engine):
    """Engine 扩展：周期性保存 G/S 及优化器的检查点。"""

    def set_checkpoint(self, dir_path: str, save_every: int = 500, keep_last: int = 5):
        self._ckpt_dir = dir_path
        self._ckpt_every = int(save_every)
        self._ckpt_keep = int(keep_last)
        os.makedirs(self._ckpt_dir, exist_ok=True)
        self._recent_ckpts = deque(maxlen=self._ckpt_keep)

    def _state_dict(self):
        state = {
            "global_step": getattr(self, "global_step", 0),
            "problems": {},
        }
        for p in self.problems:
            # 仅保存必要内容，避免不兼容对象
            state["problems"][p.name] = {
                "module": p.module.state_dict(),
                "optimizer": p.optimizer.state_dict() if p.optimizer is not None else None,
            }
        return state

    def _save_checkpoint(self, suffix: str):
        fname = f"ckpt_{suffix}.pt"
        path = os.path.join(self._ckpt_dir, fname)
        torch.save(self._state_dict(), path)
        self._recent_ckpts.append(path)
        # 维护最近 N 个
        while len(self._recent_ckpts) > self._recent_ckpts.maxlen:
            old = self._recent_ckpts.popleft()
            try:
                os.remove(old)
            except Exception:
                pass

    def train_step(self):
        # 调用基类执行一次训练步
        super().train_step()
        # 周期性保存
        if hasattr(self, "_ckpt_every") and self._ckpt_every > 0:
            gs = getattr(self, "global_step", 0)
            if gs > 0 and gs % self._ckpt_every == 0:
                self._save_checkpoint(suffix=f"step{gs}")
        # 进度打印（心跳）
        if hasattr(self, "_print_every") and self._print_every > 0:
            gs = getattr(self, "global_step", 0)
            if gs > 0 and gs % self._print_every == 0:
                # 聚合最近的指标
                metrics = {}
                for p in self.problems:
                    if hasattr(p, "_last_metrics"):
                        metrics[p.name] = p._last_metrics
                # 估算 epoch（若可得）
                if getattr(self, "_steps_per_epoch", None):
                    epoch = gs // max(1, self._steps_per_epoch)
                    print(f"[Progress] step={gs} epoch={epoch} metrics={metrics}")
                else:
                    print(f"[Progress] step={gs} metrics={metrics}")
        # SR 细粒度指标打印（PSNR/SSIM/损失），默认每步
        if hasattr(self, "_sr_detail_every") and self._sr_detail_every > 0:
            gs = getattr(self, "global_step", 0)
            if gs > 0 and gs % self._sr_detail_every == 0:
                sr_metrics = None
                for p in self.problems:
                    if getattr(p, "name", "") == "sr" and hasattr(p, "_last_metrics"):
                        sr_metrics = p._last_metrics
                        break
                if sr_metrics is not None:
                    psnr = sr_metrics.get("psnr")
                    ssim = sr_metrics.get("ssim")
                    l_total = sr_metrics.get("total")
                    l_l1 = sr_metrics.get("l1")
                    l_perc = sr_metrics.get("perc")
                    l_gan = sr_metrics.get("gan")
                    print(
                        f"[SR] step={gs} psnr={psnr:.4f} ssim={ssim:.4f} "
                        f"loss_total={l_total:.4f} l1={l_l1:.4f} perc={l_perc:.4f} gan={l_gan:.4f}"
                    )

    def set_progress(self, print_every: int = 100, sr_detail_every: int = 1):
        self._print_every = int(print_every)
        self._sr_detail_every = int(sr_detail_every)
        # 估算每轮步数（用于打印 epoch），默认取第一个有 train_data_loader 的 problem
        self._steps_per_epoch = None
        for p in getattr(self, "problems", []):
            if hasattr(p, "train_data_loader") and p.train_data_loader is not None:
                try:
                    self._steps_per_epoch = len(p.train_data_loader)
                    break
                except Exception:
                    continue


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
        valid_step=10,
        roll_back=True,
    )

    deg_problem = DegProblem(
        name="deg",
        module=G,
        optimizer=optim_G,
        train_data_loader=train_loader,
        config=inner_cfg,
        discriminator=D_deg,
        optim_D=optim_D_deg,
    )
    sr_problem = SRProblem(
        name="sr",
        module=S,
        optimizer=optim_S,
        train_data_loader=train_loader,
        config=inner_cfg,
    )
    sr_problem.set_discriminator(D_sr, optim_D_sr)
    arch_problem = ArchProblem(
        name="arch",
        module=G,
        optimizer=optim_arch,
        train_data_loader=val_loader,
        config=outer_cfg,
        sr_model=S,
    )

    problems = [deg_problem, sr_problem, arch_problem]
    # 依赖关系：G/S 作为 lower，Arch 作为 upper
    l2u = {deg_problem: [arch_problem], sr_problem: [arch_problem]}
    u2l = {arch_problem: [deg_problem, sr_problem]}
    dependencies = {"l2u": l2u, "u2l": u2l}

    engine = MLOEngine(config=engine_cfg, problems=problems, dependencies=dependencies)
    # 启用周期性保存与进度打印
    engine.set_checkpoint(dir_path=os.path.join("running_files", "mlo_ckpt"), save_every=200, keep_last=5)
    # print_every: 50 步输出全量 metrics；sr_detail_every: 默认每步输出 SR 的 PSNR/SSIM
    engine.set_progress(print_every=50, sr_detail_every=1)
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
    from datasets.sr_datasets import PairedImageFolders
    stage1_hr_folder = "/data/hanying/DIV2K_train_HR/"
    stage1_lr_folder = ""
    train_set = PairedImageFolders()
    train_set = DummySRDataset(length=500)
    val_set = DummySRDataset(length=50)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    engine = build_mlo_engine(train_loader, val_loader, device)
    engine.run()


if __name__ == "__main__":
    main()


