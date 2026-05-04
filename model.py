"""EfficientNet-B2 model factory and the canonical class list.

Single source of truth for class ordering — imported by dataset.py, train.py,
evaluate.py, gradcam.py, and app.py so that index <-> label is identical
everywhere.

Architecture (per README + Day-3 Opus review):
  - Backbone: timm tf_efficientnet_b2, pretrained=True, num_classes=0, global_pool='avg'
  - Head:     nn.Sequential(Dropout(0.3), Linear(num_features → 7))
  - num_features is read from backbone.num_features (1408 for B2) — never hardcoded.
  - Two-stage training uses freeze_backbone(model, freeze=True/False).
    When freeze=True, backbone.eval() is ALSO called after model.train() so that
    BatchNorm running mean/var are not updated during the warm-up stage.

Checkpoint contract (used by train.py, evaluate.py, app.py):
  {
    "state_dict":       model.state_dict(),
    "classes":          CLASSES,
    "epoch":            int,
    "val_balanced_acc": float,
    "model_name":       "tf_efficientnet_b2",
  }
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm

# ── Canonical class list ─────────────────────────────────────────────────────
# ORDER MATTERS — dataset.py, evaluate.py, app.py all rely on this mapping.
CLASSES: list[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES: int = len(CLASSES)

_MODEL_NAME = "tf_efficientnet_b2"


# ── Model ────────────────────────────────────────────────────────────────────

class HAMClassifier(nn.Module):
    """EfficientNet-B2 backbone + classification head for HAM10000.

    Attributes
    ----------
    backbone : nn.Module
        timm tf_efficientnet_b2 with global average pooling, no classifier head.
    head : nn.Sequential
        Dropout(0.3) → Linear(num_features, num_classes).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            _MODEL_NAME,
            pretrained=True,
            num_classes=0,       # strip timm's classifier; we attach our own
            global_pool="avg",
        )
        feat_dim: int = self.backbone.num_features  # 1408 for B2 — read dynamically
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W) → (B, num_classes)
        features = self.backbone(x)   # (B, feat_dim)
        return self.head(features)    # (B, num_classes) — raw logits


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> HAMClassifier:
    """Instantiate and return a HAMClassifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes.  Default matches HAM10000 (7).
    dropout : float
        Dropout probability applied before the linear head.
    """
    return HAMClassifier(num_classes=num_classes, dropout=dropout)


def freeze_backbone(model: HAMClassifier, freeze: bool) -> None:
    """Freeze or unfreeze the backbone for two-stage training.

    Parameters
    ----------
    model : HAMClassifier
    freeze : bool
        True  → freeze backbone (stage 1 warm-up: train head only).
        False → unfreeze backbone (stage 2 fine-tune: train everything).

    Notes
    -----
    When *freezing*, ``model.backbone.eval()`` must be called every time after
    ``model.train()`` to prevent BatchNorm running mean/var from drifting.
    ``freeze_backbone`` sets ``requires_grad`` on all backbone parameters and
    also calls eval/train on the backbone itself, but the caller is responsible
    for re-calling ``model.backbone.eval()`` after each ``model.train()`` call
    during stage 1 (the training loop does this explicitly).
    """
    for param in model.backbone.parameters():
        param.requires_grad = not freeze

    if freeze:
        model.backbone.eval()   # stop BN running stats from updating
    else:
        model.backbone.train()  # re-enable BN updates for fine-tuning


def save_checkpoint(
    model: HAMClassifier,
    epoch: int,
    val_balanced_acc: float,
    path,
) -> None:
    """Save a checkpoint in the standard contract format used by train/evaluate/app."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": CLASSES,
            "epoch": epoch,
            "val_balanced_acc": val_balanced_acc,
            "model_name": _MODEL_NAME,
        },
        path,
    )


def load_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
    num_classes: int = NUM_CLASSES,
) -> tuple[HAMClassifier, dict]:
    """Load a checkpoint written by train.py.

    Returns
    -------
    model : HAMClassifier
        Model with weights loaded, moved to *device*, set to eval mode.
    meta : dict
        Everything in the checkpoint except 'state_dict' (epoch, metrics, …).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Validate class list for forward-compat
    if ckpt.get("classes") != CLASSES:
        raise ValueError(
            f"Checkpoint class list {ckpt.get('classes')} doesn't match "
            f"current CLASSES {CLASSES}. Re-train or use the correct checkpoint."
        )

    model = build_model(num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta


if __name__ == "__main__":
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {_MODEL_NAME}")
    print(f"  Total params:     {n_params:,}")
    print(f"  Trainable params: {n_trainable:,}")
    print(f"  Feature dim:      {model.backbone.num_features}")
    print(f"  Classes ({NUM_CLASSES}):   {CLASSES}")

    # Smoke-forward
    dummy = torch.zeros(2, 3, 224, 224)
    out = model(dummy)
    print(f"  Output shape:     {tuple(out.shape)}  OK")

    # Freeze test
    freeze_backbone(model, freeze=True)
    frozen = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
    total_bb = sum(1 for _ in model.backbone.parameters())
    print(f"  Backbone frozen:  {frozen}/{total_bb} params  OK")

    freeze_backbone(model, freeze=False)
    unfrozen = sum(1 for p in model.backbone.parameters() if p.requires_grad)
    print(f"  Backbone unfrozen:{unfrozen}/{total_bb} params  OK")
