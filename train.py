"""Two-stage training loop for the HAM10000 classifier.

CLI (per README):
    python train.py --epochs 30 --batch-size 32 --lr 1e-5

Full options:
    python train.py \\
        --epochs 30 --batch-size 32 --lr 1e-5 \\
        --warmup-epochs 5 --warmup-lr 1e-3 \\
        --seed 42 --num-workers 4 \\
        --checkpoint-dir checkpoints/

Two-stage strategy
------------------
Stage 1  (epochs 1 … warmup_epochs):
    Backbone frozen, head trained at warmup_lr.
    model.backbone.eval() is called *after* model.train() every step so that
    BatchNorm running mean/var don't drift while parameters are frozen.

Stage 2  (epochs warmup_epochs+1 … epochs):
    Full fine-tune at lr.  Cosine scheduler resets at stage boundary.

Checkpointing
-------------
best_model.pth  — best val balanced accuracy ever seen
last_model.pth  — end of each epoch

Checkpoint dict contract (shared with evaluate.py, app.py):
    {
        "state_dict":       model.state_dict(),
        "classes":          CLASSES,            # list[str], length 7
        "epoch":            int,
        "val_balanced_acc": float,
        "model_name":       "tf_efficientnet_b2",
    }

Early stopping
--------------
Patience of 7 epochs on val balanced accuracy (computed in both stages).
"""

from __future__ import annotations

import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score, f1_score

from dataset import HAMDataset
from model import CLASSES, HAMClassifier, build_model, freeze_backbone, save_checkpoint
from scripts.make_splits import validate_existing_splits
from transforms import get_eval_transform, get_train_transform

ROOT = Path(__file__).resolve().parent
SPLITS_DIR = ROOT / "splits"
DATA_DIR = ROOT / "data"


# ── Reproducibility ──────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── WeightedRandomSampler ────────────────────────────────────────────────────

def make_weighted_sampler(dataset: HAMDataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that oversamples rare classes.

    Per-sample weight = 1 / class_count[label].  Replacement=True so that
    every epoch sees exactly len(dataset) samples drawn from the weighted
    distribution.  ONLY used for the train loader — never val/test.
    """
    counts = dataset.class_counts          # {class_idx: count}
    weights = [1.0 / counts[label] for label in dataset.labels]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
    )


# ── DataLoaders ──────────────────────────────────────────────────────────────

def make_loaders(
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = HAMDataset(
        SPLITS_DIR / "train.csv",
        DATA_DIR,
        transform=get_train_transform(),
    )
    val_ds = HAMDataset(
        SPLITS_DIR / "val.csv",
        DATA_DIR,
        transform=get_eval_transform(),
    )

    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,          # replaces shuffle=True; mutually exclusive
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,           # avoid tiny last batch messing up BN
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


# ── Metric helpers ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: HAMClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    """Run one pass over *loader* and return loss + sklearn metrics."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "loss": total_loss / n,
        "balanced_acc": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# ── Training stages ──────────────────────────────────────────────────────────

def run_epoch(
    model: HAMClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    use_amp: bool,
    frozen_backbone: bool,
    writer: SummaryWriter,
    global_step: int,
) -> tuple[float, int]:
    """Train for one epoch. Returns (avg_loss, updated_global_step)."""
    model.train()
    if frozen_backbone:
        # BN running stats must not update while backbone is frozen
        model.backbone.eval()

    total_loss = 0.0
    n_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        global_step += 1

        if batch_idx % max(1, n_batches // 10) == 0:
            writer.add_scalar("train/step_loss", loss.item(), global_step)

    return total_loss / n_batches, global_step


# ── Main training function ───────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    # Validate that splits haven't drifted since generation
    validate_existing_splits()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  |  AMP: {use_amp}")

    # ── Directories ──
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=str(ROOT / "runs" / run_tag))
    print(f"TensorBoard run: runs/{run_tag}")

    # ── Data ──
    print("Building data loaders …")
    train_loader, val_loader = make_loaders(args.batch_size, args.num_workers)
    print(
        f"  Train batches: {len(train_loader)}  "
        f"Val batches: {len(val_loader)}"
    )

    # ── Model ──
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_val_acc = 0.0
    best_epoch = 0
    no_improve_count = 0
    global_step = 0

    # ── Stage 1: frozen backbone warm-up ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Stage 1: Warm-up ({args.warmup_epochs} epochs, backbone frozen, lr={args.warmup_lr})")
    print(f"{'='*60}")

    freeze_backbone(model, freeze=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.warmup_lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.warmup_epochs, eta_min=args.warmup_lr * 0.01
    )

    for epoch in range(1, args.warmup_epochs + 1):
        t0 = time.time()
        train_loss, global_step = run_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, use_amp, frozen_backbone=True, writer=writer,
            global_step=global_step,
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)

        elapsed = time.time() - t0
        print(
            f"  [S1 Ep {epoch:02d}/{args.warmup_epochs:02d}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_bal_acc={val_metrics['balanced_acc']:.4f}  "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}  "
            f"({elapsed:.0f}s)"
        )

        # TensorBoard
        writer.add_scalar("stage1/train_loss", train_loss, epoch)
        writer.add_scalar("stage1/val_loss", val_metrics["loss"], epoch)
        writer.add_scalar("stage1/val_balanced_acc", val_metrics["balanced_acc"], epoch)
        writer.add_scalar("stage1/val_macro_f1", val_metrics["macro_f1"], epoch)
        writer.add_scalar("stage1/val_weighted_f1", val_metrics["weighted_f1"], epoch)
        writer.add_scalar("stage1/lr", scheduler.get_last_lr()[0], epoch)

        # Checkpoint
        save_checkpoint(model, epoch, val_metrics["balanced_acc"], ckpt_dir / "last_model.pth")

        if val_metrics["balanced_acc"] > best_val_acc:
            best_val_acc = val_metrics["balanced_acc"]
            best_epoch = epoch
            no_improve_count = 0
            save_checkpoint(model, epoch, best_val_acc, ckpt_dir / "best_model.pth")
            print(f"    *** New best val_balanced_acc={best_val_acc:.4f} — checkpoint saved ***")
        else:
            no_improve_count += 1
            if no_improve_count >= args.patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # ── Stage 2: full fine-tune ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Stage 2: Fine-tune ({args.epochs - args.warmup_epochs} epochs, backbone unfrozen, lr={args.lr})")
    print(f"{'='*60}")

    freeze_backbone(model, freeze=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    stage2_epochs = args.epochs - args.warmup_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=stage2_epochs, eta_min=args.lr * 0.01
    )
    no_improve_count = 0  # reset patience counter for stage 2

    for epoch_offset in range(1, stage2_epochs + 1):
        epoch = args.warmup_epochs + epoch_offset
        t0 = time.time()

        train_loss, global_step = run_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, use_amp, frozen_backbone=False, writer=writer,
            global_step=global_step,
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)

        elapsed = time.time() - t0
        print(
            f"  [S2 Ep {epoch_offset:02d}/{stage2_epochs:02d}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_bal_acc={val_metrics['balanced_acc']:.4f}  "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}  "
            f"({elapsed:.0f}s)"
        )

        # TensorBoard
        writer.add_scalar("stage2/train_loss", train_loss, epoch_offset)
        writer.add_scalar("stage2/val_loss", val_metrics["loss"], epoch_offset)
        writer.add_scalar("stage2/val_balanced_acc", val_metrics["balanced_acc"], epoch_offset)
        writer.add_scalar("stage2/val_macro_f1", val_metrics["macro_f1"], epoch_offset)
        writer.add_scalar("stage2/val_weighted_f1", val_metrics["weighted_f1"], epoch_offset)
        writer.add_scalar("stage2/lr", scheduler.get_last_lr()[0], epoch_offset)

        # Checkpoint
        save_checkpoint(model, epoch, val_metrics["balanced_acc"], ckpt_dir / "last_model.pth")

        if val_metrics["balanced_acc"] > best_val_acc:
            best_val_acc = val_metrics["balanced_acc"]
            best_epoch = epoch
            no_improve_count = 0
            save_checkpoint(model, epoch, best_val_acc, ckpt_dir / "best_model.pth")
            print(f"    *** New best val_balanced_acc={best_val_acc:.4f} — checkpoint saved ***")
        else:
            no_improve_count += 1
            if no_improve_count >= args.patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    writer.close()
    print(f"\nTraining complete.")
    print(f"  Best val balanced accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"  Checkpoints: {ckpt_dir}/best_model.pth, {ckpt_dir}/last_model.pth")
    print(f"  TensorBoard: tensorboard --logdir runs/")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HAM10000 skin lesion classifier")
    p.add_argument("--epochs", type=int, default=30,
                   help="Total training epochs across both stages (default: 30)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for train and val loaders (default: 32)")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Stage-2 fine-tune learning rate (default: 1e-5)")
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="Number of stage-1 warm-up epochs with frozen backbone (default: 5)")
    p.add_argument("--warmup-lr", type=float, default=1e-3,
                   help="Stage-1 warm-up learning rate (default: 1e-3)")
    p.add_argument("--seed", type=int, default=42,
                   help="Global random seed (default: 42)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker processes (default: 4)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                   help="Directory to write checkpoints (default: checkpoints/)")
    p.add_argument("--patience", type=int, default=7,
                   help="Early-stopping patience in epochs (default: 7)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
