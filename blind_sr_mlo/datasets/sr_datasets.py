import os
from glob import glob
from typing import Tuple, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F


class SRImagePairDataset(Dataset):
    """
    通用 SR 数据集封装:
      - HR 根目录: hr_root，支持子目录
      - 可选 LR 根目录: lr_root，若缺省则运行时用 bicubic from HR 生成 LR

    用法示例:
      train_set = SRImagePairDataset(
          hr_root="/path/to/DIV2K_train_HR",
          lr_root=None,  # 或 "/path/to/DIV2K_train_LR_bicubic/X4"
          scale=4,
          patch_size=256,
          augment=True,
      )
    """

    def __init__(
        self,
        hr_root: str,
        lr_root: Optional[str] = None,
        scale: int = 4,
        patch_size: int = 256,
        augment: bool = True,
        file_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ):
        super().__init__()
        self.hr_root = hr_root
        self.lr_root = lr_root
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment

        self.hr_paths: List[str] = []
        for ext in file_exts:
            self.hr_paths.extend(glob(os.path.join(hr_root, f"**/*{ext}"), recursive=True))
        self.hr_paths = sorted(self.hr_paths)
        if not self.hr_paths:
            raise RuntimeError(f"No HR images found in {hr_root}")

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.hr_paths)

    def _load_hr(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.to_tensor(img)  # [3,H,W], 0-1

    def _random_crop_pair(self, hr: torch.Tensor) -> torch.Tensor:
        _, H, W = hr.shape
        ps = self.patch_size
        if H < ps or W < ps:
            # 如果图太小，先放大一点
            scale = max(ps / H, ps / W)
            new_h, new_w = int(H * scale) + 2, int(W * scale) + 2
            hr = F.interpolate(hr.unsqueeze(0), size=(new_h, new_w), mode="bicubic", align_corners=False).squeeze(0)
            _, H, W = hr.shape
        top = torch.randint(0, H - ps + 1, (1,)).item()
        left = torch.randint(0, W - ps + 1, (1,)).item()
        hr = hr[:, top : top + ps, left : left + ps]
        return hr

    def _augment(self, hr: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return hr
        if torch.rand(1) < 0.5:
            hr = torch.flip(hr, dims=[2])  # 水平翻转
        if torch.rand(1) < 0.5:
            hr = torch.flip(hr, dims=[1])  # 垂直翻转
        return hr

    def _get_lr_from_hr(self, hr: torch.Tensor) -> torch.Tensor:
        # 下采样到 1/scale 尺寸
        _, H, W = hr.shape
        lr_h, lr_w = H // self.scale, W // self.scale
        lr = F.interpolate(hr.unsqueeze(0), size=(lr_h, lr_w), mode="bicubic", align_corners=False).squeeze(0)
        return lr

    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]
        hr = self._load_hr(hr_path)
        hr = self._random_crop_pair(hr)
        hr = self._augment(hr)

        if self.lr_root is not None:
            # 通过文件名匹配 LR，简单假设同名/平行目录
            rel = os.path.relpath(hr_path, self.hr_root)
            lr_path = os.path.join(self.lr_root, rel)
            if not os.path.exists(lr_path):
                # 回退到 bicubic 生成
                lr = self._get_lr_from_hr(hr)
            else:
                lr_img = Image.open(lr_path).convert("RGB")
                lr = self.to_tensor(lr_img)
        else:
            lr = self._get_lr_from_hr(hr)

        return {"hr": hr, "lr": lr}


