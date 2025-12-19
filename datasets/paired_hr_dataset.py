import os
from typing import Tuple

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class HRDataset(Dataset):
    """
    Simple dataset of high-resolution images used as ground-truth HR.

    Folder structure:
        root/
            img1.png
            img2.png
            ...
    """

    def __init__(self, root: str, image_size: Tuple[int, int] = None):
        self.root = root
        self.files = sorted(
            [
                f
                for f in os.listdir(root)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ]
        )

        transforms = [T.ToTensor()]
        if image_size is not None:
            transforms.insert(0, T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC))
        self.transform = T.Compose(transforms)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.root, fname)
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, fname


