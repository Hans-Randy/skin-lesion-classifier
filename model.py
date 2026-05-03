"""EfficientNet-B2 model factory and the canonical class list.

Single source of truth for class ordering — imported by dataset.py, train.py,
evaluate.py, gradcam.py, and app.py so that index <-> label is identical
everywhere.
"""

from __future__ import annotations

CLASSES: list[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES: int = len(CLASSES)


if __name__ == "__main__":
    print(f"{NUM_CLASSES} classes: {CLASSES}")
