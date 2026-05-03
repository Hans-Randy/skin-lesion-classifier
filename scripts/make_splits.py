"""Generate train/val/test splits for HAM10000.

Strategy: StratifiedGroupKFold(n_splits=7) on (y=dx, groups=lesion_id).
  - Fold 0  → test  (~14.3%)
  - Fold 1  → val   (~14.3%)
  - Folds 2-6 → train (~71.4%)

See docs/splits.md for rationale and cache contract.

Usage:
    uv run python scripts/make_splits.py            # generate (skip if up-to-date)
    uv run python scripts/make_splits.py --overwrite  # force regenerate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = ROOT / "data" / "HAM10000_metadata.csv"
SPLITS_DIR = ROOT / "splits"
META_PATH = SPLITS_DIR / "_meta.json"

SEED = 42
N_SPLITS = 7
TEST_FOLD = 0
VAL_FOLD = 1
COLUMNS = ["image_id", "dx", "lesion_id"]


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_splits(overwrite: bool = False) -> None:
    SPLITS_DIR.mkdir(exist_ok=True)

    current_md5 = md5_of_file(METADATA_PATH)

    # Check if cached splits are still valid
    if not overwrite and META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        if meta.get("ham_metadata_md5") == current_md5:
            print("Splits are up-to-date. Use --overwrite to regenerate.")
            return
        else:
            print("HAM10000_metadata.csv has changed — regenerating splits.")

    print(f"Generating splits from {METADATA_PATH} ...")

    df = pd.read_csv(METADATA_PATH)

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(df, y=df["dx"], groups=df["lesion_id"]))

    test_idx = folds[TEST_FOLD][1]
    val_idx = folds[VAL_FOLD][1]

    all_idx = set(range(len(df)))
    test_set = set(test_idx)
    val_set = set(val_idx)
    train_set = all_idx - test_set - val_set

    train_df = df.iloc[sorted(train_set)][COLUMNS].reset_index(drop=True)
    val_df = df.iloc[sorted(val_set)][COLUMNS].reset_index(drop=True)
    test_df = df.iloc[sorted(test_set)][COLUMNS].reset_index(drop=True)

    # ── Zero-overlap assertion (lesion_id level) ──────────────────────────────
    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])

    assert len(train_lesions & val_lesions) == 0, "LEAK: train ∩ val"
    assert len(train_lesions & test_lesions) == 0, "LEAK: train ∩ test"
    assert len(val_lesions & test_lesions) == 0, "LEAK: val ∩ test"
    print("OK: Zero lesion_id overlap across all splits")

    # ── Every class present in val and test ───────────────────────────────────
    all_classes = set(df["dx"].unique())
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        missing = all_classes - set(split_df["dx"].unique())
        if missing:
            print(f"WARNING: classes missing from {split_name}: {missing}", file=sys.stderr)
        else:
            print(f"OK: All 7 classes present in {split_name}")

    # ── Write CSVs ────────────────────────────────────────────────────────────
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    # ── Write meta ────────────────────────────────────────────────────────────
    meta = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "test_fold": TEST_FOLD,
        "val_fold": VAL_FOLD,
        "ham_metadata_md5": current_md5,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    print(
        f"\nSplits written to {SPLITS_DIR}:\n"
        f"  train: {meta['train_rows']:,} rows\n"
        f"  val:   {meta['val_rows']:,} rows\n"
        f"  test:  {meta['test_rows']:,} rows\n"
    )

    # ── Per-class breakdown ───────────────────────────────────────────────────
    print("Class distribution per split:")
    summary = pd.DataFrame({
        "train": train_df["dx"].value_counts(),
        "val": val_df["dx"].value_counts(),
        "test": test_df["dx"].value_counts(),
    }).fillna(0).astype(int)
    print(summary.to_string())


def validate_existing_splits() -> None:
    """Assert cached splits are consistent with current metadata CSV.
    Called by train.py at startup."""
    if not META_PATH.exists():
        raise FileNotFoundError(
            "splits/_meta.json not found. Run: uv run python scripts/make_splits.py"
        )
    meta = json.loads(META_PATH.read_text())
    current_md5 = md5_of_file(METADATA_PATH)
    if meta["ham_metadata_md5"] != current_md5:
        raise ValueError(
            "HAM10000_metadata.csv has changed since splits were generated. "
            "Re-run: uv run python scripts/make_splits.py --overwrite"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HAM10000 train/val/test splits")
    parser.add_argument("--overwrite", action="store_true", help="Force regenerate even if cached")
    args = parser.parse_args()
    generate_splits(overwrite=args.overwrite)
