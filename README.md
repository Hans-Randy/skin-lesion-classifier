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

Evaluated on a held-out test set (15% of data, stratified split).

| Metric | Score |
|--------|------:|
| Balanced accuracy | **0.81** |
| Macro F1-score | **0.76** |
| Weighted F1-score | **0.88** |

> Raw accuracy is not reported as the primary metric — the class imbalance makes it misleading (a model predicting only `nv` would achieve ~67% accuracy).

### Confusion matrix

![Confusion matrix heatmap across all 7 classes](assets/confusion_matrix.png)

### Per-class F1-scores

| Class | Precision | Recall | F1 |
|-------|----------:|-------:|---:|
| nv | 0.94 | 0.97 | 0.95 |
| mel | 0.72 | 0.68 | 0.70 |
| bkl | 0.71 | 0.74 | 0.72 |
| bcc | 0.82 | 0.80 | 0.81 |
| akiec | 0.70 | 0.67 | 0.68 |
| vasc | 0.85 | 0.88 | 0.86 |
| df | 0.65 | 0.62 | 0.63 |

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

```bash
python >= 3.10
torch >= 2.0
torchvision
timm
albumentations
gradio
scikit-learn
pandas
pytorch-grad-cam
huggingface_hub
```

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/skin-lesion-classifier
cd skin-lesion-classifier
pip install -r requirements.txt
```

### Download the dataset

```bash
kaggle datasets download kmader/skin-lesion-analysis-toward-melanoma-detection
unzip skin-lesion-analysis-toward-melanoma-detection.zip -d data/
```

### Train

```bash
python train.py --epochs 30 --batch-size 32 --lr 1e-5
```

### Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

### Run the Gradio demo locally

```bash
python app.py
```

---

## Project Structure

```
skin-lesion-classifier/
├── data/                   # HAM10000 images and metadata CSV
├── checkpoints/            # Saved model weights
├── assets/                 # README images (confusion matrix, Grad-CAM)
├── train.py                # Training script
├── evaluate.py             # Evaluation + metrics
├── dataset.py              # Custom PyTorch Dataset class
├── model.py                # Model definition
├── transforms.py           # Augmentation pipeline
├── gradcam.py              # Grad-CAM visualization utilities
├── app.py                  # Gradio demo app
├── requirements.txt
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
