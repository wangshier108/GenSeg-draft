import argparse
import os
import math
from typing import Dict, Any
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from blind_sr_mlo.models import DegradationGenerator
from blind_sr_mlo.train_mlo_sr import PatchDiscriminator  # reuse same small discriminator
from blind_sr_mlo.datasets.sr_datasets import UnpairedHRLRDataset


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5, eps: float = 1e-8) -> torch.Tensor:
    """Compute SSIM for 3-channel images in range [0,1]."""
    device = img1.device
    gauss = torch.tensor(
        [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)],
        device=device,
    )
    gauss = gauss / gauss.sum()
    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window = window_2d.expand(3, 1, window_size, window_size)
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

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps
    )
    return ssim_map.mean()


def build_dataloader(
    hr_dir: str,
    lr_bank_dir: str,
    batch_size: int,
    num_workers: int,
    hr_patch_size: int = 256,
    scale: int = 4,
):
    ds = UnpairedHRLRDataset(
        hr_root=hr_dir,
        lr_root=lr_bank_dir,
        scale=scale,
        hr_patch_size=hr_patch_size,
        augment=True,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


def train_stage1(
    hr_dir: str,
    lr_bank_dir: str,
    iters: int = 5000,
    batch_size: int = 8,
    num_workers: int = 4,
    lr_g: float = 1e-4,
    lr_d: float = 1e-4,
    ckpt_dir: str = "running_files/stage1_deg",
    ckpt_every: int = 500,
    print_every: int = 50,
    device: str = "cuda",
    hr_patch_size: int = 256,
    scale: int = 4,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    loader = build_dataloader(hr_dir, lr_bank_dir, batch_size, num_workers, hr_patch_size=hr_patch_size, scale=scale)

    G = DegradationGenerator().to(device)
    D = PatchDiscriminator(in_ch=3).to(device)
    optim_G = torch.optim.Adam(G.parameters(), lr=lr_g)
    optim_D = torch.optim.Adam(D.parameters(), lr=lr_d)

    l1 = nn.L1Loss()
    adv = nn.BCEWithLogitsLoss()

    os.makedirs(ckpt_dir, exist_ok=True)

    step = 0
    epoch = 0
    while step < iters:
        epoch += 1
        epoch_loss = 0.0
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        epoch_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False, dynamic_ncols=True)
        for batch in pbar:
            # move to device
            for key in batch:
                batch[key] = batch[key].to(device, non_blocking=True)
            hr = batch["hr"]
            lr_real = batch["lr"]

            # forward
            lr_syn, k, sigma, mu, qf = G(hr)
            if lr_syn.shape[-2:] != lr_real.shape[-2:]:
                lr_syn = F.interpolate(lr_syn, size=lr_real.shape[-2:], mode="bicubic", align_corners=False)

            loss_l1 = l1(lr_syn, lr_real)
            k_grad_h = k[:, :, 1:, :] - k[:, :, :-1, :]
            k_grad_w = k[:, :, :, 1:] - k[:, :, :, :-1]
            loss_k = torch.norm(k, p=2) + 0.1 * (torch.norm(k_grad_h, p=2) + torch.norm(k_grad_w, p=2))
            n = sigma + mu
            n_grad_h = n[:, :, 1:, :] - n[:, :, :-1, :]
            n_grad_w = n[:, :, :, 1:] - n[:, :, :, :-1]
            loss_tv = 0.05 * (torch.mean(torch.abs(n_grad_h)) + torch.mean(torch.abs(n_grad_w)))

            # D update
            D.train()
            optim_D.zero_grad()
            with torch.no_grad():
                lr_syn_detach = lr_syn.detach()
            pred_real = D(lr_real)
            pred_fake = D(lr_syn_detach)
            real_labels = torch.ones_like(pred_real)
            fake_labels = torch.zeros_like(pred_fake)
            loss_D = 0.5 * (adv(pred_real, real_labels) + adv(pred_fake, fake_labels))
            loss_D.backward()
            optim_D.step()

            # G update (recompute preds to keep graph)
            pred_fake_for_G = D(lr_syn)
            loss_G_adv = adv(pred_fake_for_G, real_labels)
            loss = loss_l1 + loss_k + loss_tv + 0.01 * loss_G_adv

            optim_G.zero_grad()
            loss.backward()
            optim_G.step()

            step += 1
            epoch_batches += 1
            epoch_loss += loss.item()
            epoch_loss_G += loss.item()
            epoch_loss_D += loss_D.item()
            # progress update
            if step % print_every == 0:
                with torch.no_grad():
                    mse = torch.mean((lr_syn.clamp(0, 1) - lr_real.clamp(0, 1)) ** 2)
                    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
                    ssim = _ssim(lr_syn.clamp(0, 1), lr_real.clamp(0, 1))
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "l1": f"{loss_l1.item():.4f}",
                    "k_reg": f"{loss_k.item():.4f}",
                    "tv": f"{loss_tv.item():.4f}",
                    "adv": f"{loss_G_adv.item():.4f}",
                    "psnr": f"{psnr.item():.2f}",
                    "ssim": f"{ssim.item():.4f}",
                })

            if step % ckpt_every == 0:
                torch.save({
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "optim_G": optim_G.state_dict(),
                    "optim_D": optim_D.state_dict(),
                    "step": step,
                }, os.path.join(ckpt_dir, f"ckpt_step{step}.pt"))

            if step >= iters:
                break
        pbar.close()
        if epoch_batches > 0:
            avg_loss = epoch_loss / epoch_batches
            avg_g = epoch_loss_G / epoch_batches
            avg_d = epoch_loss_D / epoch_batches
            print(f"[Stage1] epoch={epoch} step={step} avg_loss={avg_loss:.4f} avg_G={avg_g:.4f} avg_D={avg_d:.4f}")

    final_path = os.path.join(ckpt_dir, "deg_final.pt")
    torch.save({"G": G.state_dict()}, final_path)
    print(f"[Stage1] finished {step} steps, saved G to {final_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage1: pretrain degradation generator")
    parser.add_argument("--hr_dir", type=str, required=True, help="Path to HR images")
    parser.add_argument("--lr_bank_dir", type=str, required=True, help="Path to LR bank (NTIRE 2017 Track2)")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--ckpt_dir", type=str, default="running_files/stage1_deg")
    parser.add_argument("--ckpt_every", type=int, default=500)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hr_patch_size", type=int, default=256, help="HR patch size for cropping")
    parser.add_argument("--scale", type=int, default=4, help="Downscale factor between HR and LR")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_stage1(
        hr_dir=args.hr_dir,
        lr_bank_dir=args.lr_bank_dir,
        iters=args.iters,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        ckpt_dir=args.ckpt_dir,
        ckpt_every=args.ckpt_every,
        print_every=args.print_every,
        device=args.device,
        hr_patch_size=args.hr_patch_size,
        scale=args.scale,
    )
