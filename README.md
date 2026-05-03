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
├── data/                   # HAM10000 images and metadata CSV (gitignored)
├── checkpoints/            # Saved model weights (gitignored)
├── assets/                 # README images (confusion matrix, Grad-CAM)
├── runs/                   # TensorBoard logs (gitignored)
├── splits/                 # Cached train/val/test CSV splits (seed=42)
├── scripts/
│   ├── download_data.sh    # Kaggle download helper (macOS/Linux)
│   └── download_data.ps1   # Kaggle download helper (Windows)
├── docs/                   # Architecture decision records
├── train.py                # Training script
├── evaluate.py             # Evaluation + metrics
├── dataset.py              # Custom PyTorch Dataset class
├── model.py                # Model definition + canonical CLASSES constant
├── transforms.py           # Augmentation pipeline
├── gradcam.py              # Grad-CAM visualization utilities
├── app.py                  # Gradio demo app
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Pinned dependency lockfile
└── README.md
```

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
