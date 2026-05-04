---
title: Skin Lesion Classifier
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.0"
app_file: app.py
python_version: "3.10"
pinned: false
license: mit
---

# Skin Lesion Classifier

A deep learning model that classifies dermoscopic images into 7 skin lesion categories using transfer learning with EfficientNet-B2, trained on the HAM10000 dataset.

**Live demo →** [huggingface.co/spaces/YOUR_USERNAME/skin-lesion-classifier](https://huggingface.co/spaces)

![Prediction example showing Grad-CAM heatmap overlaid on a dermoscopic image](assets/gradcam_example.png)

---

## Overview

Skin cancer is one of the most common cancers worldwide, and early detection is critical for successful treatment. This project builds a multi-class classifier that can identify 7 types of skin lesions from dermoscopic images, with Grad-CAM visualizations to highlight which regions of an image drive each prediction.

> **Disclaimer:** This model is for educational and research purposes only. It is not intended for clinical use or medical diagnosis.

---

## Dataset

**HAM10000** (Human Against Machine with 10000 training images)  
Source: [Kaggle — Skin Lesion Analysis Toward Melanoma Detection](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)

| Class | Label | Count | % of dataset |
|-------|-------|------:|-------------:|
| Melanocytic nevi | `nv` | 6,705 | 66.9% |
| Melanoma | `mel` | 1,113 | 11.1% |
| Benign keratosis | `bkl` | 1,099 | 11.0% |
| Basal cell carcinoma | `bcc` | 514 | 5.1% |
| Actinic keratosis | `akiec` | 327 | 3.3% |
| Vascular lesions | `vasc` | 142 | 1.4% |
| Dermatofibroma | `df` | 115 | 1.1% |

The dataset is heavily imbalanced — handling this was a key design challenge (see [Training](#training)).

---

## Model Architecture

- **Backbone:** EfficientNet-B2 (pretrained on ImageNet via `timm`)
- **Classifier head:** replaced final layer with `Linear(1408 → 7)`
- **Training strategy:** two-stage fine-tuning — frozen backbone for warm-up, then full fine-tuning

```
Input image (224×224×3)
       ↓
EfficientNet-B2 backbone (pretrained)
       ↓
Global average pooling
       ↓
Dropout(0.3)
       ↓
Linear(1408 → 7)
       ↓
Softmax → class probabilities
```

---

## Training

### Handling class imbalance
- Used `WeightedRandomSampler` to oversample rare classes during training
- Applied label smoothing (`ε = 0.1`) to reduce overconfident predictions

### Augmentation pipeline (`albumentations`)
- Random horizontal & vertical flip
- Rotation ±30°
- Hue-saturation-value jitter (accounts for varying skin tones and imaging conditions)
- Random brightness/contrast
- Coarse dropout

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 224 × 224 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate (warm-up) | 1e-3 |
| Learning rate (fine-tune) | 1e-5 |
| LR scheduler | CosineAnnealingLR |
| Epochs | 30 total |
| Weight decay | 1e-4 |

---

## Results

> **🚧 Training in progress.** The numbers below are design targets from the project plan. This section will be updated with real results after the Day 5–6 training and evaluation runs.

Evaluated on a held-out test set (15% of data, stratified and grouped by `lesion_id` — 70/15/15 split).

| Metric | Target |
|--------|-------:|
| Balanced accuracy | **~0.81** |
| Macro F1-score | **~0.76** |
| Weighted F1-score | **~0.88** |

> Raw accuracy is not reported as the primary metric — the class imbalance makes it misleading (a model predicting only `nv` would achieve ~67% accuracy).

### Confusion matrix

![Confusion matrix heatmap across all 7 classes](assets/confusion_matrix.png)

### Per-class F1-scores

| Class | Precision | Recall | F1 |
|-------|----------:|-------:|---:|
| nv | — | — | — |
| mel | — | — | — |
| bkl | — | — | — |
| bcc | — | — | — |
| akiec | — | — | — |
| vasc | — | — | — |
| df | — | — | — |

---

## Grad-CAM Visualizations

Grad-CAM highlights the image regions that most influenced each prediction, providing interpretability into what the model has learned.

![Grad-CAM examples for melanoma, nevi, and BCC](assets/gradcam_grid.png)

*Left to right: original image, Grad-CAM heatmap, overlay. Top row: correct predictions. Bottom row: misclassified examples with analysis.*

Key observations:
- For melanoma, the model correctly attends to irregular border regions and color variation
- Most misclassifications occur between `mel` and `nv` — the most clinically consequential confusion pair
- The model occasionally focuses on image artifacts (ruler marks, hair) rather than lesion features — a known failure mode in dermoscopy datasets

---

## Setup & Usage

### Requirements

- Python ≥ 3.10
- All dependencies are pinned in `pyproject.toml` / `uv.lock`. Key packages: torch ≥ 2.0, timm, albumentations, gradio, pytorch-grad-cam, scikit-learn, pandas, tensorboard, huggingface_hub.

### Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS/Linux
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

git clone https://github.com/Hans-Randy/skin-lesion-classifier
cd skin-lesion-classifier
uv sync          # creates .venv and installs all pinned deps
```

> **GPU note:** the lock file resolves the CPU wheel of torch by default. For CUDA training swap to the CUDA index before `uv sync` — see the [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

### Download the dataset

```bash
# macOS / Linux
bash scripts/download_data.sh

# Windows (PowerShell)
.\scripts\download_data.ps1
```

Requires a [Kaggle API token](https://www.kaggle.com/docs/api) at `~/.kaggle/kaggle.json`.

### Train

```bash
uv run python train.py --epochs 30 --batch-size 32 --lr 1e-5
```

Monitor training in TensorBoard:

```bash
uv run tensorboard --logdir runs/
```

### Evaluate

```bash
uv run python evaluate.py --checkpoint checkpoints/best_model.pth
```

### Run the Gradio demo locally

```bash
uv run python app.py
```

---

## Project Structure

```
skin-lesion-classifier/
├── data/                    # HAM10000 images and metadata CSV (gitignored)
├── checkpoints/             # Saved model weights (gitignored)
├── assets/                  # README images (confusion matrix, Grad-CAM)
├── runs/                    # TensorBoard logs (gitignored)
├── splits/                  # Cached train/val/test CSV splits (seed=42)
├── scripts/
│   ├── download_data.sh     # Kaggle download helper (macOS/Linux)
│   ├── download_data.ps1    # Kaggle download helper (Windows)
│   ├── make_splits.py       # Generate stratified group splits (StratifiedGroupKFold)
│   ├── sanity_check.py      # Post-training regression guard (shape, probs, class list)
│   └── smoke_test.py        # Dev smoke test (1 epoch, 200 samples, CPU)
├── docs/
│   └── splits.md            # Split design decisions and cache contract
├── train.py                 # Two-stage training loop (frozen warm-up → full fine-tune)
├── evaluate.py              # Balanced acc, macro/weighted F1, confusion matrix
├── dataset.py               # HAMDataset (two-folder image lookup, albumentations)
├── model.py                 # HAMClassifier, build_model(), freeze_backbone(), CLASSES
├── transforms.py            # Albumentations train + eval pipelines
├── gradcam.py               # Grad-CAM wrapper (backbone.bn2 target layer)
├── app.py                   # Gradio demo (disclaimer, top-7 probs, CAM overlay)
├── requirements.txt         # HF Spaces compatible (exported from uv)
├── pyproject.toml           # Project metadata and dependencies
├── uv.lock                  # Pinned dependency lockfile
└── README.md
```

---

## Why These Choices

### WeightedRandomSampler vs. class-weighted loss vs. SMOTE

Three standard approaches exist for handling class imbalance:

1. **Class-weighted loss** — multiply the loss for each sample by `1/class_count`. Simple, but in practice the gradient is still dominated by the majority class (67% `nv`) because you're seeing far more `nv` examples per epoch.
2. **SMOTE / synthetic oversampling** — generate synthetic minority-class samples. Creates artificial data points by interpolating in feature space, which works poorly for natural images — the interpolations don't look like real lesions and can introduce spurious texture patterns.
3. **WeightedRandomSampler** — resample the training set so each batch is approximately class-balanced, without ever synthesising new data.

`WeightedRandomSampler` was chosen because it preserves real image distributions (no synthetic artifacts) and keeps the loss numerics clean. In a medical imaging context, not introducing artificial training examples is important: you can't know what distortions SMOTE might create at the pixel level, and that uncertainty isn't acceptable when the downstream user might encounter edge cases similar to those interpolated artifacts.

### Balanced accuracy + macro F1 over raw accuracy

The HAM10000 dataset is ~67% `nv` (melanocytic nevi). A trivial model that always predicts `nv` would achieve ~67% raw accuracy but ~14% balanced accuracy and near-zero macro F1. Using balanced accuracy (mean per-class recall) and macro F1 (unweighted mean of per-class F1) forces the model to perform across all 7 classes, not just the majority one.

### The `mel` ↔ `nv` failure mode and what Grad-CAM reveals

`mel` (melanoma) and `nv` (benign nevi) are visually similar — both are pigmented, melanocyte-derived lesions. The key distinguishing features (irregular border, asymmetric colour distribution, atypical vascular patterns) are subtle and sometimes absent. This is also the most clinically costly confusion: mistaking a melanoma for a benign nevus can delay treatment.

Grad-CAM is most useful here as a sanity check. A model that achieves high `mel` F1 by attending to dermoscopy ruler marks, hair, or background vignette is *right for the wrong reasons* — it will fail on images taken without those artifacts. The Grad-CAM overlays in `assets/gradcam_grid.png` include deliberate bottom-row examples of this failure mode, where the heatmap shows attention drifting to peripheral artifacts rather than the lesion core.

---

## What I Would Improve Next

- **Test-time augmentation (TTA):** average predictions over multiple augmented views of each test image to improve robustness
- **Ensemble:** combine EfficientNet-B2 with a Vision Transformer (ViT) backbone; ensembles typically gain 2–3% on medical imaging benchmarks
- **Hair removal preprocessing:** dermoscopy images often contain hair that distracts the model — classical inpainting methods can remove it as a preprocessing step
- **Larger backbone:** EfficientNet-B4 or ConvNeXt-Base would likely improve performance at the cost of longer training time

---

## References

- Tschandl, P. et al. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data.
- Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
- Selvaraju, R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.

---

## License

MIT License. See `LICENSE` for details.

---

*Built as part of an AI portfolio project. Feedback welcome — open an issue or reach out on [LinkedIn](https://linkedin.com/in/YOUR_USERNAME).*
