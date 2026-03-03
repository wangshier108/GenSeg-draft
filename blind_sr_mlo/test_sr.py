"""
SR evaluation script for blind_sr_mlo.

Computes PSNR / SSIM / LPIPS on paired LR/HR folders using a trained SR model.
Assumes the SR backbone is EDSR with a fixed upscale factor.
Supports checkpoints saved by MLOEngine (problems['sr']['module']).
"""

import argparse
import math
import os
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from blind_sr_mlo.models import EDSR


def _gaussian_window(window_size: int, sigma: float, device: torch.device):
    gauss = torch.tensor(
        [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma**2)) for x in range(window_size)],
        device=device,
    )
    gauss = gauss / gauss.sum()
    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    return window_2d


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, sigma: float = 1.5, eps: float = 1e-8) -> torch.Tensor:
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
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps)
    return ssim_map.mean()


class PairedEvalDataset(Dataset):
    """Loads paired HR/LR images without cropping; assumes ordering matches after sorting."""

    def __init__(self, hr_dir: str, lr_dir: str):
        self.hr_paths = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir)])
        self.lr_paths = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir)])
        if len(self.hr_paths) != len(self.lr_paths):
            raise RuntimeError(f"HR/LR count mismatch: {len(self.hr_paths)} vs {len(self.lr_paths)}")
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr = self.to_tensor(Image.open(self.hr_paths[idx]).convert("RGB"))
        lr = self.to_tensor(Image.open(self.lr_paths[idx]).convert("RGB"))
        return {"hr": hr, "lr": lr}


def load_sr_weights(model: torch.nn.Module, ckpt: str):
    state = torch.load(ckpt, map_location="cpu")
    sd = None
    if isinstance(state, dict):
        if "problems" in state and "sr" in state["problems"]:
            sd = state["problems"]["sr"].get("module") or state["problems"]["sr"].get("module_state")
        elif "module_state" in state:
            sd = state["module_state"]
    if sd is None:
        raise RuntimeError(f"No SR weights found in checkpoint: {ckpt}")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[Load] SR weights from {ckpt}, missing={len(missing)}, unexpected={len(unexpected)}")


def try_lpips(device: torch.device):
    try:
        import lpips  # type: ignore

        net = lpips.LPIPS(net="alex").to(device)
        net.eval()
        return net
    except Exception:
        return None


def lpips_score(lpips_net, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # expects tensors in [-1,1]
    return lpips_net(x * 2 - 1, y * 2 - 1).mean()


def evaluate(model, loader, device: torch.device):
    model.eval()
    lpips_net = try_lpips(device)
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            # If sizes mismatch, resize LR up by model scale before forward
            sr = model(lr)
            sr = sr.clamp(0, 1)
            hr = hr.clamp(0, 1)

            mse = torch.mean((sr - hr) ** 2)
            psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
            ssim_val = ssim(sr, hr)
            if lpips_net is not None:
                lpips_val = lpips_score(lpips_net, sr, hr)
            else:
                # fallback: simple VGG-like perceptual proxy using MSE in pixel space
                lpips_val = mse

            bsz = lr.shape[0]
            count += bsz
            psnr_sum += psnr.item() * bsz
            ssim_sum += ssim_val.item() * bsz
            lpips_sum += lpips_val.item() * bsz

    return {
        "psnr": psnr_sum / count,
        "ssim": ssim_sum / count,
        "lpips": lpips_sum / count,
        "count": count,
    }


def main():
    parser = argparse.ArgumentParser("Evaluate SR model (PSNR/SSIM/LPIPS)")
    parser.add_argument("--hr_dir", required=True, help="Directory of HR ground-truth images")
    parser.add_argument("--lr_dir", required=True, help="Directory of corresponding LR images")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path (mlo_ckpt ckpt_*.pt)")
    parser.add_argument("--scale", type=int, default=2, help="Upscale factor used by SR model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = PairedEvalDataset(args.hr_dir, args.lr_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = EDSR(scale=args.scale).to(device)
    load_sr_weights(model, args.ckpt)

    metrics = evaluate(model, loader, device)
    print(
        f"Eval on {metrics['count']} images — PSNR: {metrics['psnr']:.4f}, SSIM: {metrics['ssim']:.4f}, LPIPS: {metrics['lpips']:.4f}"
    )


if __name__ == "__main__":
    main()