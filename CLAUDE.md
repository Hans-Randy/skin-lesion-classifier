# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

**Days 1–10 complete.** All modules are implemented and on `main`.

| File | Status | Notes |
|---|---|---|
| `pyproject.toml` / `uv.lock` | Done | torch 2.11.0, timm 1.0.26, albumentations 2.0.8, gradio 6.14.0, grad-cam 1.5.5 |
| `dataset.py` | Done | `HAMDataset`, two-folder image lookup, `class_counts` property |
| `transforms.py` | Done | `get_train_transform()`, `get_eval_transform()` |
| `model.py` | Done | `HAMClassifier`, `build_model()`, `freeze_backbone()`, `save_checkpoint()`, `load_checkpoint()` |
| `train.py` | Done | Two-stage loop, `WeightedRandomSampler`, AMP, TensorBoard, early stopping (patience=7) |
| `evaluate.py` | Done | Balanced acc, macro/weighted F1, confusion matrix PNG, per-class CSV |
| `gradcam.py` | Done | `generate_cam()` targeting `backbone.bn2`, triptych CLI |
| `app.py` | Done | Gradio Blocks, disclaimer in header+title+result, 10 MB upload cap |
| `scripts/make_splits.py` | Done | StratifiedGroupKFold(7), seed=42, zero-overlap assertion |
| `scripts/sanity_check.py` | Done | Load checkpoint, shape, prob-sum, collapse detection |
| `scripts/smoke_test.py` | Done | 1 epoch, 200-sample subset, CPU |
| `requirements.txt` | Done | HF Spaces compatible, exported via `uv export --no-hashes` |
| `splits/{train,val,test}.csv` | Done | 7153/1431/1431 rows, all 7 classes in val+test |
| `docs/splits.md` | Done | Split rationale and cache contract |

**Remaining:** run the full 30-epoch training (`uv run python train.py --epochs 30 ...`), then run `evaluate.py` and `gradcam.py` with the resulting checkpoint to populate `assets/`. Update README results table with real numbers.

## Project goal

Multi-class classifier for 7 skin lesion categories (HAM10000 dataset), with Grad-CAM interpretability and a Gradio demo. Educational/research only — never frame outputs as clinical advice.

## Planned commands

These are the commands the README commits to. Prefix with `uv run` to use the project venv:

```bash
uv run python train.py --epochs 30 --batch-size 32 --lr 1e-5 --warmup-epochs 5 --warmup-lr 1e-3
uv run python evaluate.py --checkpoint checkpoints/best_model.pth
uv run python app.py                      # Gradio demo
```

Dataset is downloaded out-of-band. Use the helper scripts (both wrap the Kaggle CLI):

```bash
# macOS / Linux
bash scripts/download_data.sh

# Windows (PowerShell)
.\scripts\download_data.ps1
```

Or manually:

```bash
kaggle datasets download kmader/skin-lesion-analysis-toward-melanoma-detection -p data/
unzip data/skin-lesion-analysis-toward-melanoma-detection.zip -d data/
```

No test runner, linter, or formatter is configured yet. If you add one, update this section.

## Python and dependencies

Use the `/uv-package-manager` skill for all Python package management, virtual environments, and dependency-related tasks. This project uses `uv` for fast, modern Python dependency resolution.

## Model assignment rules

- **Architecture decisions and reviews:** Use Opus
- **Implementation tasks (new features, refactors):** Use Sonnet  
- **Simple edits, formatting, renaming:** Use Haiku
- **Security-sensitive changes:** Always escalate to Opus for review

## Architecture decisions worth preserving

These are the choices baked into the README — deviating from them should be a deliberate decision, not an accident:

- **Backbone:** EfficientNet-B2 from `timm`, pretrained on ImageNet. Final layer replaced with `Linear(1408 → 7)` after `Dropout(0.3)`. Input is 224×224.
- **Two-stage training:** warm-up with frozen backbone at LR `1e-3`, then full fine-tune at LR `1e-5`. AdamW + CosineAnnealingLR, weight decay `1e-4`, 30 epochs total.
- **Class imbalance handling is load-bearing.** The dataset is ~67% `nv`. The README commits to two specific mitigations: `WeightedRandomSampler` for oversampling rare classes, and label smoothing `ε=0.1`. Don't drop these silently — a model that "looks fine" on raw accuracy can be useless here.
- **Reported metrics deliberately exclude raw accuracy.** Use balanced accuracy, macro F1, weighted F1, and per-class precision/recall/F1. A model that always predicts `nv` would hit ~67% raw accuracy, so it's misleading by construction.
- **Augmentation pipeline is `albumentations`-based** (not torchvision transforms): horizontal+vertical flip, ±30° rotation, HSV jitter, brightness/contrast, coarse dropout. HSV jitter is specifically there to handle skin-tone and lighting variation across the dataset — keep it.
- **Test split:** 70/15/15 (train/val/test), stratified by class **and grouped by `lesion_id`** using `StratifiedGroupKFold` — same lesion never appears in more than one split. Cached to `splits/{train,val,test}.csv` at `seed=42`. (Note: the README says "15% held out" — that describes the test slice only; the full split is 70/15/15.)

## Grad-CAM and known failure modes

The model occasionally attends to dermoscopy artifacts (ruler marks, hair) rather than lesion features. This is a documented failure mode and the reason Grad-CAM matters here — it surfaces when the model is right for the wrong reasons. The most clinically consequential confusion pair is `mel` ↔ `nv`; treat regressions in their per-class F1 as more serious than equivalent regressions on other classes.

## Disclaimer constraint

Any user-facing output (the Gradio app, README, demo text) must keep the "educational/research only, not for clinical use" disclaimer prominent. Don't soften or remove it.
