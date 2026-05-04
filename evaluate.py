"""Test-set evaluation: balanced accuracy, macro/weighted F1, confusion matrix.

CLI (per README):
    python evaluate.py --checkpoint checkpoints/best_model.pth

Full options:
    python evaluate.py --checkpoint checkpoints/best_model.pth --split test
    python evaluate.py --checkpoint checkpoints/best_model.pth --split val

Outputs
-------
- Prints balanced accuracy, macro F1, weighted F1, and a per-class
  classification_report to stdout.
- Writes assets/confusion_matrix.png — row-normalised, cmap='Blues'.
- Writes assets/per_class_metrics.csv — per-class precision, recall, F1, support.

Notes
-----
Raw accuracy is intentionally excluded as the primary metric.  The HAM10000
training set is ~67% nv; a model that always predicts nv would score ~0.67
raw accuracy but 0.143 balanced accuracy.  All reported numbers use
balanced_accuracy_score, macro-averaged F1, and weighted F1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in headless envs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HAMDataset
from model import CLASSES, load_checkpoint
from transforms import get_eval_transform

ROOT = Path(__file__).resolve().parent
SPLITS_DIR = ROOT / "splits"
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    checkpoint_path: str,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Load model and run a full pass over *split*.

    Returns
    -------
    all_labels : list[int]   ground-truth class indices
    all_preds  : list[int]   predicted class indices
    """
    model, meta = load_checkpoint(checkpoint_path, device=device)
    model.eval()

    print(f"Checkpoint: {checkpoint_path}")
    print(f"  Trained for {meta.get('epoch', '?')} epochs")
    print(f"  Best val balanced acc: {meta.get('val_balanced_acc', '?'):.4f}")

    ds = HAMDataset(
        SPLITS_DIR / f"{split}.csv",
        DATA_DIR,
        transform=get_eval_transform(),
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_labels: list[int] = []
    all_preds: list[int] = []

    for images, labels in tqdm(loader, desc=f"Evaluating ({split})", unit="batch"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    return all_labels, all_preds


# ── Metrics ──────────────────────────────────────────────────────────────────

def print_metrics(labels: list[int], preds: list[int]) -> None:
    bal_acc = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print("Summary metrics")
    print("=" * 60)
    print(f"  Balanced accuracy : {bal_acc:.4f}")
    print(f"  Macro F1          : {macro_f1:.4f}")
    print(f"  Weighted F1       : {weighted_f1:.4f}")
    print()
    print(classification_report(
        labels, preds,
        target_names=CLASSES,
        digits=3,
        zero_division=0,
    ))


# ── Confusion matrix ─────────────────────────────────────────────────────────

def save_confusion_matrix(labels: list[int], preds: list[int], out_path: Path) -> None:
    """Save a row-normalised confusion matrix PNG."""
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(9, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=13, pad=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved -> {out_path}")


# ── Per-class CSV ─────────────────────────────────────────────────────────────

def save_per_class_csv(labels: list[int], preds: list[int], out_path: Path) -> None:
    """Save per-class precision, recall, F1, and support to a CSV."""
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(CLASSES))), zero_division=0
    )
    df = pd.DataFrame({
        "class": CLASSES,
        "precision": precision.round(4),
        "recall": recall.round(4),
        "f1": f1.round(4),
        "support": support.astype(int),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Per-class metrics saved -> {out_path}")
    print(df.to_string(index=False))


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate HAM10000 classifier on a split")
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint .pth file")
    p.add_argument("--split", choices=["train", "val", "test"], default="test",
                   help="Which split to evaluate (default: test)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Inference batch size (default: 64)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker processes (default: 4)")
    p.add_argument("--confusion-matrix", type=str,
                   default=str(ASSETS_DIR / "confusion_matrix.png"),
                   help="Output path for confusion matrix PNG")
    p.add_argument("--metrics-csv", type=str,
                   default=str(ASSETS_DIR / "per_class_metrics.csv"),
                   help="Output path for per-class metrics CSV")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    labels, preds = run_inference(
        args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    print_metrics(labels, preds)
    save_confusion_matrix(labels, preds, Path(args.confusion_matrix))
    save_per_class_csv(labels, preds, Path(args.metrics_csv))
