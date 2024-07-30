import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Tuple

class YESeg(Dataset):

    CLASSES = ['background', 'bare ground', 'vegetation', 'trees', 'house', 'water', 'roads', 'other']

    def __init__(self, root: str, split: str = 'train', transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        print(f"Loading {split} data...")
        self.images, self.sar, self.labels = self.get_files(root, split)
        print(f"Found {len(self.images)} {split} images.")

    def get_files(self, root: str, split: str):
        root = Path(root)
        all_labels = list((root / 'label').rglob('*.png'))  # Assuming labels are in PNG format
        images, sar, labels = [], [], []

        file_txt = f"{split}.txt"
        with open(root / file_txt) as f:
            all_files = f.read().splitlines()

        for f in all_files:
            images.append(root / 'OPT' / f)
            sar.append(root / 'SAR' / f)
            img_name = f.split('.')[0]
            labels_per_images = list(filter(lambda x: x.stem.startswith(img_name), all_labels))
            assert labels_per_images != []
            labels.append(labels_per_images)

        assert len(images) == len(labels) and len(sar) == len(labels)
        return images, sar, labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.images[index])
        sar_path = str(self.sar[index])
        lbl_paths = self.labels[index]

        opt = self.read_image(img_path)
        sar = self.read_image(sar_path)
        label = self.read_label(lbl_paths)

        return opt,sar, label.squeeze().long()

    def read_image(self, img_path: str) -> Tensor:
        with Image.open(img_path) as img:
            image = np.array(img).astype(np.float32)
            image = torch.from_numpy(image).float()

            if image.dim() == 2:
                image = image.unsqueeze(0)

            if image.size(0) == 1:
                image = image.repeat(3, 1, 1)

            if image.dim() == 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1)

        return image

    def read_label(self, lbl_paths: list) -> Tensor:
        labels = None
        label_idx = None

        for lbl_path in lbl_paths:
            with Image.open(lbl_path) as img:
                label = np.array(img).astype(np.uint8)
                if label_idx is None:
                    label_idx = np.zeros(label.shape, dtype=np.uint8)
                label = np.ma.masked_array(label, mask=label_idx)
                label_idx += np.minimum(label, 1)
                if labels is None:
                    labels = label
                else:
                    labels += label
        return torch.from_numpy(labels.data).unsqueeze(0).to(torch.uint8)

def band_normalization(data):
    """ Normalize the matrix to (0,1), r.s.t A axis (Default=0)
        return normalized matrix and a record matrix for normalize back
    """
    size = data.shape
    for i in range(size[0]):
        _range = np.max(data[i, :, :]) - np.min(data[i, :, :])
        if _range != 0:
            data[i, :, :] = (data[i, :, :] - np.min(data[i, :, :])) / (_range + 1e-8)
    return data
