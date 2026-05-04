"""Grad-CAM utilities for EfficientNet-B2.

Target layer
------------
``model.backbone.bn2`` — the BatchNormAct2d that follows conv_head (the final
1x1 expansion conv).  This is the last spatial feature map before global
average pooling, giving the highest-resolution attribution map available on
B2 at 224×224 input (~7×7 receptive grid).

Choosing bn2 over the last MBConv block (blocks.6.1.bn3) is deliberate:
  - conv_head expands channels 352→1408, so bn2 captures the full feature
    richness used by the classifier head.
  - pytorch-grad-cam requires that the target layer produces a 4-D tensor
    (B, C, H, W); bn2 satisfies this before AdaptiveAvgPool2d flattens it.

Usage (library)
---------------
    from gradcam import generate_cam
    orig_rgb, heatmap, overlay = generate_cam(model, image_tensor)

Usage (CLI)
-----------
    python gradcam.py --image data/HAM10000_images_part_1/ISIC_0024306.jpg \\
                      --checkpoint checkpoints/best_model.pth \\
                      --out assets/gradcam_example.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import CLASSES, HAMClassifier, load_checkpoint
from transforms import get_eval_transform, IMAGE_SIZE

ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"


# ── Core CAM function ────────────────────────────────────────────────────────

def generate_cam(
    model: HAMClassifier,
    image_tensor: torch.Tensor,
    target_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Grad-CAM for a single image tensor.

    Parameters
    ----------
    model : HAMClassifier
        Must be in eval mode.  Grad-CAM temporarily enables gradients
        internally — no need for torch.no_grad() context here.
    image_tensor : torch.Tensor
        Shape (1, 3, H, W), already normalized (ImageNet stats).
        Must be on the same device as the model.
    target_class : int | None
        Class index to explain.  None → use the predicted (argmax) class.

    Returns
    -------
    original_rgb : np.ndarray  shape (H, W, 3)  float32 in [0, 1]
    heatmap      : np.ndarray  shape (H, W, 3)  float32 in [0, 1]  (jet colormap)
    overlay      : np.ndarray  shape (H, W, 3)  float32 in [0, 1]  heatmap blended on image
    """
    assert image_tensor.dim() == 4 and image_tensor.shape[0] == 1, \
        "image_tensor must be shape (1, C, H, W)"

    # Target layer: bn2 is the BatchNormAct2d after conv_head
    # See module docstring for rationale.
    target_layer = model.backbone.bn2

    if target_class is None:
        with torch.no_grad():
            logits = model(image_tensor)
        target_class = int(logits.argmax(dim=1).item())

    targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)  # (1, H, W)

    grayscale_cam = grayscale_cam[0]  # (H, W)

    # Reconstruct original RGB image from normalized tensor
    # Reverse ImageNet normalization for display
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    original_rgb = np.clip(img_np * std + mean, 0, 1).astype(np.float32)

    # Build jet heatmap for standalone display
    heatmap_bgr = cv2.applyColorMap(
        (grayscale_cam * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Blended overlay
    overlay = show_cam_on_image(original_rgb, grayscale_cam, use_rgb=True)
    overlay = overlay.astype(np.float32) / 255.0

    return original_rgb, heatmap, overlay


# ── Triptych writer ──────────────────────────────────────────────────────────

def save_triptych(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    label: str,
    predicted_class: str,
    out_path: Path,
) -> None:
    """Save a (original | heatmap | overlay) triptych PNG."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        (original_rgb, f"Input: {label}"),
        (heatmap, "Grad-CAM heatmap"),
        (overlay, f"Overlay — pred: {predicted_class}"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(
        "EDUCATIONAL / RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS",
        fontsize=9, color="red", y=0.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Triptych saved -> {out_path}")


# ── Image loading helper ─────────────────────────────────────────────────────

def load_image_tensor(
    image_path: str | Path,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """Load an image file and return (normalized_tensor, original_rgb_array).

    Returns
    -------
    tensor       : torch.Tensor  shape (1, 3, H, W) on *device*
    original_rgb : np.ndarray    shape (H, W, 3) uint8, resized to IMAGE_SIZE
    """
    transform = get_eval_transform()
    img_pil = Image.open(image_path).convert("RGB")
    # Keep a resized copy for display before normalization
    img_resized = img_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    original_rgb = np.array(img_resized)

    img_np = np.array(img_pil)
    tensor = transform(image=img_np)["image"]  # (3, H, W)
    return tensor.unsqueeze(0).to(device), original_rgb


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Grad-CAM triptych for a single dermoscopy image"
    )
    p.add_argument("--image", required=True,
                   help="Path to input image (.jpg/.png)")
    p.add_argument("--checkpoint", required=True,
                   help="Path to trained model checkpoint (.pth)")
    p.add_argument("--out", type=str,
                   default=str(ASSETS_DIR / "gradcam_example.png"),
                   help="Output path for triptych PNG")
    p.add_argument("--target-class", type=int, default=None,
                   help="Class index to explain (default: predicted class)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, meta = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    print(f"Checkpoint loaded (epoch {meta.get('epoch', '?')}, "
          f"val_bal_acc={meta.get('val_balanced_acc', '?'):.4f})")

    tensor, _ = load_image_tensor(args.image, device)

    # Determine predicted class
    with torch.no_grad():
        logits = model(tensor)
    pred_idx = int(logits.argmax(dim=1).item())
    pred_class = CLASSES[pred_idx]
    probs = logits.softmax(dim=1).squeeze().cpu().tolist()

    print(f"Predicted: {pred_class} ({probs[pred_idx]:.3f})")
    print("Top-3:")
    for idx in sorted(range(len(CLASSES)), key=lambda i: probs[i], reverse=True)[:3]:
        print(f"  {CLASSES[idx]:8s}  {probs[idx]:.4f}")

    target_class = args.target_class if args.target_class is not None else pred_idx
    original_rgb, heatmap, overlay = generate_cam(model, tensor, target_class=target_class)

    save_triptych(
        original_rgb, heatmap, overlay,
        label=Path(args.image).stem,
        predicted_class=pred_class,
        out_path=Path(args.out),
    )
