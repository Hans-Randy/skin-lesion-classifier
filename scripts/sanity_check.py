"""Regression guard — load best_model.pth and assert basic output properties.

Checks:
  1. Checkpoint loads without error; class list matches CLASSES in model.py.
  2. Forward pass on a synthetic image produces shape (1, 7).
  3. Softmax probabilities sum to 1.0 (±1e-5).
  4. Single real image (if provided) produces a non-uniform distribution
     (not all-equal logits, i.e. the model is not collapsed to a prior).

Usage:
    uv run python scripts/sanity_check.py --checkpoint checkpoints/best_model.pth
    uv run python scripts/sanity_check.py --checkpoint checkpoints/best_model.pth \\
                                          --image data/HAM10000_images_part_1/ISIC_0024306.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import CLASSES, load_checkpoint


def run_checks(checkpoint: str, image_path: str | None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint}\n")

    # ── 1. Load checkpoint ────────────────────────────────────────────────────
    model, meta = load_checkpoint(checkpoint, device=device)
    model.eval()
    assert meta["classes"] == CLASSES, \
        f"FAIL: class list mismatch\n  got:      {meta['classes']}\n  expected: {CLASSES}"
    print(f"[1] Checkpoint loaded — epoch {meta['epoch']}, "
          f"val_bal_acc={meta['val_balanced_acc']:.4f}  PASS")

    # ── 2. Output shape ───────────────────────────────────────────────────────
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        logits = model(dummy)
    assert logits.shape == (1, 7), f"FAIL: output shape {logits.shape} != (1, 7)"
    print(f"[2] Output shape (1, 7)  PASS")

    # ── 3. Probability sum ────────────────────────────────────────────────────
    probs = logits.softmax(dim=1)
    prob_sum = probs.sum().item()
    assert abs(prob_sum - 1.0) < 1e-5, f"FAIL: probabilities sum to {prob_sum:.8f}"
    print(f"[3] Probabilities sum to 1.0  PASS")

    # ── 4. Real image (optional) ──────────────────────────────────────────────
    if image_path is not None:
        from PIL import Image
        import numpy as np
        from transforms import get_eval_transform

        img_np = np.array(Image.open(image_path).convert("RGB"))
        transform = get_eval_transform()
        tensor = transform(image=img_np)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits_real = model(tensor)
        probs_real = logits_real.softmax(dim=1).squeeze()
        pred_idx = int(probs_real.argmax().item())
        pred_class = CLASSES[pred_idx]
        max_prob = probs_real.max().item()
        std_probs = probs_real.std().item()

        # A collapsed model would output nearly uniform probs — std < 0.01 is suspicious
        assert std_probs > 0.01, \
            f"FAIL: probability std={std_probs:.4f} is suspiciously low (model may be collapsed)"

        print(f"[4] Real image inference  PASS")
        print(f"    Predicted: {pred_class}  ({max_prob:.3f})")
        print("    All class probs:")
        for cls, p in zip(CLASSES, probs_real.tolist()):
            bar = "#" * int(p * 40)
            print(f"      {cls:8s}  {p:.4f}  {bar}")

    print("\nAll checks PASSED.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity-check a trained checkpoint")
    p.add_argument("--checkpoint", required=True,
                   help="Path to best_model.pth or any checkpoint file")
    p.add_argument("--image", default=None,
                   help="Optional: path to a real dermoscopy image for check 4")
    return p.parse_args()


if __name__ == "__main__":
    run_checks(**vars(parse_args()))
