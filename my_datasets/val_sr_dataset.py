import os
from typing import Tuple

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class ValSRDataset(Dataset):
    """
    Validation dataset providing paired (LR, HR) images.

    Folder structure:
        lr_root/
            img1.png
        hr_root/
            img1.png
    """

    def __init__(
        self,
        lr_root: str,
        hr_root: str,
        lr_size: Tuple[int, int] = None,
        hr_size: Tuple[int, int] = None,
    ):
        self.lr_root = lr_root
        self.hr_root = hr_root

        # assume filenames are aligned between LR and HR folders
        self.files = sorted(
            [
                f
                for f in os.listdir(lr_root)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
                and os.path.exists(os.path.join(hr_root, f))
            ]
        )

        lr_transforms = [T.ToTensor()]
        hr_transforms = [T.ToTensor()]
        if lr_size is not None:
            lr_transforms.insert(
                0, T.Resize(lr_size, interpolation=T.InterpolationMode.BICUBIC)
            )
        if hr_size is not None:
            hr_transforms.insert(
                0, T.Resize(hr_size, interpolation=T.InterpolationMode.BICUBIC)
            )

        self.lr_transform = T.Compose(lr_transforms)
        self.hr_transform = T.Compose(hr_transforms)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        lr_path = os.path.join(self.lr_root, fname)
        hr_path = os.path.join(self.hr_root, fname)

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        lr_img = self.lr_transform(lr_img)
        hr_img = self.hr_transform(hr_img)

        return (lr_img, fname), (hr_img, fname)


