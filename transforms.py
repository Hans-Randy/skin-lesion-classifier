"""Albumentations-based augmentation pipelines.

Two pipelines are exported:
  - get_train_transform()  — used for the training DataLoader
  - get_eval_transform()   — used for val, test, and inference (no augmentation)

Design notes:
  - HSV jitter (HueSaturationValue) is deliberately kept — it handles skin-tone
    and imaging-condition variation across the HAM10000 dataset.
  - CoarseDropout simulates occlusion (ruler marks, hair) — the known failure mode.
  - Resize is applied last so rotations/crops operate at native resolution first.
  - ImageNet mean/std normalization matches the EfficientNet-B2 pretrained weights.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_SIZE = 224

# ImageNet statistics — must match timm's EfficientNet-B2 pretrained weights
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform() -> A.Compose:
    """Augmentation pipeline for training. Returns an albumentations Compose."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.7),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.7,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0,
            p=0.3,
        ),
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transform() -> A.Compose:
    """Deterministic pipeline for val/test/inference — no augmentation."""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


if __name__ == "__main__":
    import numpy as np

    dummy = np.zeros((450, 600, 3), dtype=np.uint8)
    for name, t in [("train", get_train_transform()), ("eval", get_eval_transform())]:
        out = t(image=dummy)["image"]
        print(f"{name}: {tuple(out.shape)}  dtype={out.dtype}")
