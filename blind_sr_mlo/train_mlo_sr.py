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
import argparse

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
        self.deg_module = None  # optional: generate synthetic LR for extra supervision
        self.real_branch_weight = 1.0
        self.fake_branch_weight_max = 0.6
        self.fake_branch_ramp = 2000  # steps to reach max

    def set_discriminator(self, discriminator: nn.Module, optim_D: torch.optim.Optimizer):
        self.discriminator = discriminator
        self.optim_D = optim_D

    def set_degradation(self, deg_module: nn.Module):
        # Used to generate lr_syn from hr for additional fake branch supervision
        self.deg_module = deg_module

    def set_branch_weights(self, real_w: float = 1.0, fake_w_max: float = 0.6, ramp_steps: int = 2000):
        # Weight real vs fake branch; fake ramp-up over ramp_steps
        self.real_branch_weight = float(real_w)
        self.fake_branch_weight_max = float(fake_w_max)
        self.fake_branch_ramp = max(1, int(ramp_steps))

    def training_step(self, batch: Dict[str, Any]):
        lr_real = batch["lr"].to(self.device)
        hr = batch["hr"].to(self.device)
        sr_real = self.module(lr_real)

        sr_list = [sr_real]
        lr_tags = ["real"]
        # 动态 fake 权重：随 global_step 线性上升到上限
        gs = getattr(self, "global_step", 0)
        fake_w = self.fake_branch_weight_max * min(1.0, gs / float(self.fake_branch_ramp))
        w_list = [self.real_branch_weight]

        # 可选：使用退化生成器产生 lr_syn，形成 fake 分支，增强监督
        if self.deg_module is not None:
            with torch.no_grad():
                lr_syn, _, _, _, _ = self.deg_module(hr)
                if lr_syn.shape[-2:] != lr_real.shape[-2:]:
                    lr_syn = nn.functional.interpolate(
                        lr_syn, size=lr_real.shape[-2:], mode="bicubic", align_corners=False
                    )
            sr_fake = self.module(lr_syn)
            sr_list.append(sr_fake)
            lr_tags.append("fake")
            w_list.append(fake_w)

        # PSNR / SSIM 计算（数据范围假定 0-1）
        with torch.no_grad():
            mse = torch.mean((sr_real.clamp(0, 1) - hr.clamp(0, 1)) ** 2)
            psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
            # SSIM 使用固定窗口，避免额外依赖（统计 real 分支）
            ssim = _ssim(sr_real.clamp(0, 1), hr.clamp(0, 1))

        l1_list, perc_list, gan_list = [], [], []

        # PatchGAN 判别器更新
        if self.discriminator is not None and self.optim_D is not None:
            self.discriminator.train()
            self.optim_D.zero_grad()
            pred_real_img = self.discriminator(hr)
            real_labels = torch.ones_like(pred_real_img)
            fake_labels = torch.zeros_like(pred_real_img)

            loss_D_terms = []
            fake_weights = []
            for sr_item, w in zip(sr_list, w_list):
                with torch.no_grad():
                    sr_detach = sr_item.detach()
                pred_fake = self.discriminator(sr_detach)
                loss_D_terms.append(self.adv_loss(pred_fake, fake_labels) * w)
                fake_weights.append(w)
            sum_w_fake = max(1e-8, torch.tensor(fake_weights, device=self.device).sum())
            loss_D_fake = torch.stack(loss_D_terms).sum() / sum_w_fake
            loss_D = 0.5 * (self.adv_loss(pred_real_img, real_labels) + loss_D_fake)
            loss_D.backward(retain_graph=True)
            self.optim_D.step()

            # GAN loss for generator on all branches
            pred_fake_for_G_terms = []
            gan_weights = []
            for sr_item, w in zip(sr_list, w_list):
                pred_fake_for_G_terms.append(self.discriminator(sr_item))
                gan_weights.append(w)
            sum_w_gan = max(1e-8, torch.tensor(gan_weights, device=self.device).sum())
            l_gan = torch.stack([self.adv_loss(p, real_labels) * w for p, w in zip(pred_fake_for_G_terms, gan_weights)]).sum() / sum_w_gan
        else:
            l_gan = torch.tensor(0.0, device=self.device)

        # 汇总各分支的 L1/LPIPS
        for sr_item, w in zip(sr_list, w_list):
            l1_i = self.l1(sr_item, hr) * w
            perc_i = self.perc(sr_item, hr) * w
            l1_list.append(l1_i)
            perc_list.append(perc_i)
        sum_w = max(1e-8, torch.tensor(w_list, device=self.device).sum())
        total_l1 = torch.stack(l1_list).sum() / sum_w
        total_perc = torch.stack(perc_list).sum() / sum_w

        # L② = λ1 L1 + λ2 LPIPS + λ3 LGAN
        loss = 1.0 * total_l1 + 0.1 * total_perc + 0.01 * l_gan

        self._last_metrics = {
            "l1": float(total_l1.item()),
            "perc": float(total_perc.item()),
            "gan": float(l_gan.item()) if hasattr(l_gan, "item") else 0.0,
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "total": float(loss.item()),
            "branches": lr_tags,
        }
        return loss


class ArchProblem(ImplicitProblem):
    """外层 NAS 问题：在验证集上最小化 L_val，对架构参数（G 中的 MixedOp 等）做优化。"""

    def __init__(self, *args, sr_model: nn.Module = None, scale: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.sr_model = sr_model
        self.l1 = nn.L1Loss()
        self.perc = VGGPerceptualLoss().to(self.device)
        self.scale = scale

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
            # Fallback assumes provided scale
            target_size = (max(1, h // getattr(self, "scale", 4)), max(1, w // getattr(self, "scale", 4)))
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


def build_mlo_engine(
    train_loader_deg: DataLoader,
    train_loader_sr: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    pretrained_g: str | None = None,
    scale: int = 2,
):
    # 退化生成器 G
    G = DegradationGenerator().to(device)
    # 可选加载 Stage1 预训练权重
    if pretrained_g and os.path.exists(pretrained_g):
        state = torch.load(pretrained_g, map_location=device)
        g_sd = None
        if isinstance(state, dict):
            # 兼容 stage1 保存 {"G": ...} 或完整 ckpt
            if "G" in state:
                g_sd = state["G"]
            elif "problems" in state and "deg" in state["problems"]:
                g_sd = state["problems"]["deg"].get("module") or state["problems"]["deg"].get("module_state")
            elif "module_state" in state:
                g_sd = state["module_state"]
        if g_sd:
            missing, unexpected = G.load_state_dict(g_sd, strict=False)
            print(f"[Init] Loaded pretrained G from {pretrained_g}, missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            print(f"[Init] Found checkpoint {pretrained_g} but no G weights were loaded.")

    optim_G = torch.optim.Adam(G.parameters(), lr=1e-4)

    # 退化 PatchGAN 判别器
    D_deg = PatchDiscriminator(in_ch=3).to(device)
    optim_D_deg = torch.optim.Adam(D_deg.parameters(), lr=1e-4)

    # 超分网络 S
    S = EDSR(scale=scale).to(device)
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
        train_data_loader=train_loader_deg,
        config=inner_cfg,
        discriminator=D_deg,
        optim_D=optim_D_deg,
    )
    sr_problem = SRProblem(
        name="sr",
        module=S,
        optimizer=optim_S,
        train_data_loader=train_loader_sr,
        config=inner_cfg,
    )
    sr_problem.set_discriminator(D_sr, optim_D_sr)
    sr_problem.set_degradation(G)
    # fake 分支权重随步数线性上升至 0.6，前 2000 步完成
    sr_problem.set_branch_weights(real_w=1.0, fake_w_max=0.6, ramp_steps=2000)
    arch_problem = ArchProblem(
        name="arch",
        module=G,
        optimizer=optim_arch,
        train_data_loader=val_loader,
        config=outer_cfg,
        sr_model=S,
        scale=scale,
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
    parser = argparse.ArgumentParser(description="Stage2: joint training with pretrained degradation")
    parser.add_argument("--hr_dir", type=str, required=False, help="SR train HR (Track2 train HR, paired with --lr_dir)")
    parser.add_argument("--lr_dir", type=str, required=False, help="SR train LR (Track2 train LR, paired with --hr_dir)")
    parser.add_argument("--hr_val_dir", type=str, required=False, help="Arch/val HR (Track2 val HR, paired with --lr_val_dir)")
    parser.add_argument("--lr_val_dir", type=str, required=False, help="Arch/val LR (Track2 val LR, paired with --hr_val_dir)")
    parser.add_argument("--lr_bank_dir", type=str, required=False, help="Deg LR bank (unpaired, e.g., NTIRE 2018/2017 Track2 LR train)")
    parser.add_argument("--scale", type=int, default=2, help="SR upscale factor (e.g., 2 for Track2 X2)")
    parser.add_argument("--hr_patch_size", type=int, default=256, help="HR patch size for SR/val aligned cropping")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--pretrained_g", type=str, default="running_files/stage1_deg/deg_final.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 数据集选择：
    # deg: 继续使用 LR bank 的非配对分布（保持 Stage1 退化分布）；若无 bank，则退化到配对数据
    # sr: 使用配对的 Track2 train (HR/LR)
    # arch/val: 使用 Track2 val (HR/LR)
    from blind_sr_mlo.datasets.sr_datasets import PairedImageFolders, UnpairedHRLRDataset

    # 必须提供真实数据路径：SR/Arch 用配对 Track2，Deg 继续用 Stage1 的 HR + LR bank 分布
    if not (args.hr_dir and args.lr_dir):
        raise ValueError("SR train set requires --hr_dir and --lr_dir (paired Track2 train).")
    if not (args.hr_val_dir and args.lr_val_dir):
        raise ValueError("Arch/val set requires --hr_val_dir and --lr_val_dir (paired Track2 val).")
    if not (args.hr_dir and args.lr_bank_dir):
        raise ValueError("Deg set requires --hr_dir and --lr_bank_dir (Stage1 LR bank).")

    train_set_sr = PairedImageFolders(args.hr_dir, args.lr_dir, hr_patch_size=args.hr_patch_size, scale=args.scale, train=True)
    val_set = PairedImageFolders(args.hr_val_dir, args.lr_val_dir, hr_patch_size=args.hr_patch_size, scale=args.scale, train=False)
    print("[Stage2] Deg uses unpaired HR (--hr_dir) + LR bank (--lr_bank_dir). SR uses paired Track2 train (--hr_dir/--lr_dir); val uses Track2 val (--hr_val_dir/--lr_val_dir).")
    train_set_deg = UnpairedHRLRDataset(hr_root=args.hr_dir, lr_root=args.lr_bank_dir, scale=args.scale, hr_patch_size=256, augment=True)
    train_loader_sr = DataLoader(train_set_sr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_loader_deg = DataLoader(train_set_deg, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=args.num_workers)

    engine = build_mlo_engine(train_loader_deg, train_loader_sr, val_loader, device, pretrained_g=args.pretrained_g, scale=args.scale)
    engine.config.train_iters = args.iters
    print(f"[Stage2] Joint training with pretrained G from {args.pretrained_g}, iters={args.iters}")
    engine.run()


if __name__ == "__main__":
    import argparse
    main()


