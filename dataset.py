"""HAM10000 PyTorch Dataset.

Usage:
    from dataset import HAMDataset
    from transforms import get_train_transform

    ds = HAMDataset("splits/train.csv", "data/", transform=get_train_transform())
    image_tensor, label = ds[0]   # torch.Size([3, 224, 224]), int

The Dataset:
  - Reads a split CSV produced by scripts/make_splits.py (columns: image_id, dx, lesion_id)
  - Resolves image paths across HAM10000_images_part_1/ and HAM10000_images_part_2/ at
    init time — O(1) lookup in __getitem__, not os.path.exists per call
  - Maps dx label strings to integer indices via CLASS_TO_IDX from model.py
  - Applies the supplied albumentations transform (PIL → numpy → albumentations → tensor)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from model import CLASS_TO_IDX


class HAMDataset(Dataset):
    """HAM10000 dermoscopy image dataset.

    Args:
        csv_path:   Path to one of splits/{train,val,test}.csv
        data_dir:   Root data directory containing HAM10000_images_part_1/
                    and HAM10000_images_part_2/
        transform:  An albumentations Compose pipeline (or None for raw PIL arrays)
    """

    IMAGE_DIRS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]

    def __init__(
        self,
        csv_path: str | Path,
        data_dir: str | Path,
        transform: object | None = None,
    ) -> None:
        self.transform = transform
        self.data_dir = Path(data_dir)

        df = pd.read_csv(csv_path)
        self.image_ids: list[str] = df["image_id"].tolist()
        self.labels: list[int] = [CLASS_TO_IDX[dx] for dx in df["dx"]]

        # Build {image_id: full_path} once — avoids repeated filesystem lookups
        self._path_map: dict[str, Path] = self._build_path_map()

        missing = [iid for iid in self.image_ids if iid not in self._path_map]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} image(s) not found in {self.data_dir}. "
                f"First missing: {missing[0]}"
            )

    def _build_path_map(self) -> dict[str, Path]:
        path_map: dict[str, Path] = {}
        for subdir in self.IMAGE_DIRS:
            img_dir = self.data_dir / subdir
            if not img_dir.exists():
                continue
            for p in img_dir.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    path_map[p.stem] = p
        return path_map

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        image_id = self.image_ids[idx]
        label = self.labels[idx]

        img = Image.open(self._path_map[image_id]).convert("RGB")
        img_np = np.array(img)

        if self.transform is not None:
            img_np = self.transform(image=img_np)["image"]

        return img_np, label

    @property
    def class_counts(self) -> dict[int, int]:
        """Return {class_index: count} — used to build WeightedRandomSampler."""
        counts: dict[int, int] = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return counts


if __name__ == "__main__":
    from pathlib import Path
    from transforms import get_train_transform, get_eval_transform

    root = Path(__file__).parent
    for split, tfm in [("train", get_train_transform()), ("val", get_eval_transform())]:
        ds = HAMDataset(root / "splits" / f"{split}.csv", root / "data", transform=tfm)
        img, lbl = ds[0]
        print(f"{split}: {len(ds)} samples | image {tuple(img.shape)} dtype={img.dtype} | label {lbl}")
