# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This repo is in early build-out. As of this writing the working tree contains only `README.md` — none of the modules referenced below (`train.py`, `evaluate.py`, `dataset.py`, `model.py`, `transforms.py`, `gradcam.py`, `app.py`) exist yet. The `feature/build-model` branch is where the initial implementation is being written. Treat the README as the design spec; do not assume any of the listed scripts or directories exist until you've verified with a tool.

## Project goal

Multi-class classifier for 7 skin lesion categories (HAM10000 dataset), with Grad-CAM interpretability and a Gradio demo. Educational/research only — never frame outputs as clinical advice.

## Planned commands

These are the commands the README commits to. The scripts must be implemented to match these CLIs:

```bash
python train.py --epochs 30 --batch-size 32 --lr 1e-5
python evaluate.py --checkpoint checkpoints/best_model.pth
python app.py                      # Gradio demo
```

Dataset is downloaded out-of-band via the Kaggle CLI into `data/`:

```bash
kaggle datasets download kmader/skin-lesion-analysis-toward-melanoma-detection
unzip skin-lesion-analysis-toward-melanoma-detection.zip -d data/
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
- **Test split:** 15% held out, stratified.

## Grad-CAM and known failure modes

The model occasionally attends to dermoscopy artifacts (ruler marks, hair) rather than lesion features. This is a documented failure mode and the reason Grad-CAM matters here — it surfaces when the model is right for the wrong reasons. The most clinically consequential confusion pair is `mel` ↔ `nv`; treat regressions in their per-class F1 as more serious than equivalent regressions on other classes.

## Disclaimer constraint

Any user-facing output (the Gradio app, README, demo text) must keep the "educational/research only, not for clinical use" disclaimer prominent. Don't soften or remove it.
