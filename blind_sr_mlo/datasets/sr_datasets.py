import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import torch.nn.functional as F


class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x


class PairedImageFolders(Dataset):

    def __init__(
        self,
        root_path_1,
        root_path_2,
        hr_patch_size: int | None = None,
        scale: int = 4,
        train: bool = True,
        **kwargs,
    ):
        # Underlying folders still load full images; optional aligned crop/pad happens in __getitem__
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)
        self.hr_patch_size = hr_patch_size
        self.scale = scale
        self.train = train

    def __len__(self):
        return len(self.dataset_1)

    def _aligned_crop(self, hr: torch.Tensor, lr: torch.Tensor):
        if self.hr_patch_size is None:
            return hr, lr

        hr_patch = self.hr_patch_size
        lr_patch = max(1, hr_patch // self.scale)
        hr_patch = lr_patch * self.scale  # enforce divisibility

        def _pad_to(x: torch.Tensor, target_h: int, target_w: int):
            _, h, w = x.shape
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            if pad_h == 0 and pad_w == 0:
                return x
            # symmetric reflect padding
            pad = (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            )
            return F.pad(x, pad, mode="reflect")

        hr = _pad_to(hr, hr_patch, hr_patch)
        lr = _pad_to(lr, lr_patch, lr_patch)

        _, Hh, Wh = hr.shape
        _, Hl, Wl = lr.shape

        # Sample on the LR grid to keep HR/LR spatially aligned by `scale`
        max_lr_top = max(0, min(Hl - lr_patch, Hh // self.scale - lr_patch))
        max_lr_left = max(0, min(Wl - lr_patch, Wh // self.scale - lr_patch))

        if self.train:
            lr_top = random.randint(0, max_lr_top)
            lr_left = random.randint(0, max_lr_left)
        else:
            lr_top = max_lr_top // 2
            lr_left = max_lr_left // 2

        hr_top = lr_top * self.scale
        hr_left = lr_left * self.scale

        hr_crop = hr[:, hr_top : hr_top + hr_patch, hr_left : hr_left + hr_patch]
        lr_crop = lr[:, lr_top : lr_top + lr_patch, lr_left : lr_left + lr_patch]

        # safety: if any dim is short (shouldn't after padding), pad to exact patch
        if hr_crop.shape[-2:] != (hr_patch, hr_patch):
            hr_crop = F.pad(
                hr_crop,
                (0, max(0, hr_patch - hr_crop.shape[2]), 0, max(0, hr_patch - hr_crop.shape[1])),
                mode="reflect",
            )
        if lr_crop.shape[-2:] != (lr_patch, lr_patch):
            lr_crop = F.pad(
                lr_crop,
                (0, max(0, lr_patch - lr_crop.shape[2]), 0, max(0, lr_patch - lr_crop.shape[1])),
                mode="reflect",
            )

        return hr_crop, lr_crop

    def __getitem__(self, idx):
        hr = self.dataset_1[idx]
        lr = self.dataset_2[idx]
        hr, lr = self._aligned_crop(hr, lr)
        return {"hr": hr, "lr": lr}


class UnpairedHRLRDataset(Dataset):
    """
    非配对数据集：
      - HR 来自一个文件夹（尺寸不一致）
      - LR 来自 NTIRE 2017 Track2 的 LR-only 分布（尺寸不一致）

    用于 Stage1（只训退化生成器）：
      - 每次从 HR 取一个 patch（hr_patch_size）
      - 独立从 LR 银行取一个样本，并调整到与 HR patch 的缩放匹配（target_lr_size = hr_patch_size // scale）
      - 返回的是分布匹配用的 lr_real（非配对），可用于判别器与分布对齐；L1 等配对损失不适用

    注意：为保证 batch 内维度一致，这里会对 LR 样本进行 bicubic resize 到 target_lr_size。
    """

    def __init__(
        self,
        hr_root: str,
        lr_root: str,
        scale: int = 4,
        hr_patch_size: int = 256,
        augment: bool = True,
        file_exts: tuple = (".png", ".jpg", ".jpeg"),
    ):
        super().__init__()
        self.hr_root = hr_root
        self.lr_root = lr_root
        self.scale = scale
        self.hr_patch_size = hr_patch_size
        self.augment = augment

        self.hr_paths = []
        self.lr_paths = []
        for ext in file_exts:
            self.hr_paths.extend(sorted(
                [os.path.join(dp, f) for dp, dn, fn in os.walk(hr_root) for f in fn if f.lower().endswith(ext)]
            ))
            self.lr_paths.extend(sorted(
                [os.path.join(dp, f) for dp, dn, fn in os.walk(lr_root) for f in fn if f.lower().endswith(ext)]
            ))
        if not self.hr_paths:
            raise RuntimeError(f"No HR images found in {hr_root}")
        if not self.lr_paths:
            raise RuntimeError(f"No LR images found in {lr_root}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_paths)

    def _load_tensor(self, path: str) -> torch.Tensor:
        return self.to_tensor(Image.open(path).convert("RGB"))

    def _random_crop(self, img: torch.Tensor, patch: int) -> torch.Tensor:
        _, H, W = img.shape
        if H < patch or W < patch:
            # 先等比放大到不小于 patch，再随机裁剪
            scale = max(patch / H, patch / W)
            new_h, new_w = int(H * scale) + 2, int(W * scale) + 2
            img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode="bicubic", align_corners=False).squeeze(0)
            _, H, W = img.shape
        top = random.randint(0, H - patch)
        left = random.randint(0, W - patch)
        return img[:, top : top + patch, left : left + patch]

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return img
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])
        if random.random() < 0.5:
            img = torch.flip(img, dims=[1])
        return img

    def __getitem__(self, idx):
        # HR: 取固定尺寸 patch
        hr = self._load_tensor(self.hr_paths[idx])
        hr = self._random_crop(hr, self.hr_patch_size)
        hr = self._augment(hr)

        # LR bank: 随机抽取一个样本，然后调整到目标尺寸（hr_patch_size // scale）
        lr_path = random.choice(self.lr_paths)
        lr_img = self._load_tensor(lr_path)
        target_h = max(1, self.hr_patch_size // self.scale)
        target_w = max(1, self.hr_patch_size // self.scale)
        # 先随机裁一块接近目标大小的区域，再精确 resize 到目标，保留一定的分布多样性
        # 若原图很小，直接 resize
        _, H, W = lr_img.shape
        crop_h = min(H, target_h * 2)
        crop_w = min(W, target_w * 2)
        if H >= crop_h and W >= crop_w:
            top = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)
            lr_img = lr_img[:, top : top + crop_h, left : left + crop_w]
        lr = F.interpolate(lr_img.unsqueeze(0), size=(target_h, target_w), mode="bicubic", align_corners=False).squeeze(0)

        return {"hr": hr, "lr": lr}