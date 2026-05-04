"""Gradio demo for the HAM10000 skin lesion classifier.

EDUCATIONAL / RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS.
This tool has not been validated for clinical use. Do not use results
to make medical decisions.

Running locally:
    python app.py
    python app.py --checkpoint checkpoints/best_model.pth

HF Spaces: set the CHECKPOINT env var or place best_model.pth in the
repo root.  The app discovers the checkpoint automatically.

Architecture
------------
- Checkpoint loaded once at startup (lazy on first request if not found,
  to avoid crashing the import-time check on HF Spaces where the user
  must supply the .pth separately).
- Inference pipeline: numpy array (H, W, 3) → eval transform → model → softmax.
- Grad-CAM overlay: generate_cam() from gradcam.py, target = argmax class.
- Upload size is capped to 10 MB to prevent OOM on large files.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

from gradcam import generate_cam, load_image_tensor
from model import CLASSES, load_checkpoint
from transforms import get_eval_transform

ROOT = Path(__file__).resolve().parent
_DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "best_model.pth"
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# ── Model singleton (loaded once at startup) ─────────────────────────────────

_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_transform = get_eval_transform()


def _get_model():
    """Load model on first call; return cached instance thereafter."""
    global _model
    if _model is not None:
        return _model

    # Discover checkpoint: CLI arg > env var > default path
    ckpt_path = os.environ.get("CHECKPOINT", str(_DEFAULT_CHECKPOINT))
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run training first or set the CHECKPOINT environment variable."
        )
    _model, meta = load_checkpoint(ckpt_path, device=_device)
    _model.eval()
    print(
        f"[app] Model loaded from {ckpt_path} "
        f"(epoch {meta.get('epoch', '?')}, "
        f"val_bal_acc={meta.get('val_balanced_acc', '?'):.4f})"
    )
    return _model


# ── Inference helper ─────────────────────────────────────────────────────────

def _predict(image_np: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """Run inference + Grad-CAM on a uint8 RGB numpy array (H, W, 3).

    Returns
    -------
    label_probs : dict[str, float]   for gr.Label
    overlay     : np.ndarray (H, W, 3) uint8  for gr.Image
    """
    # ── Validate input ──
    if image_np is None:
        raise gr.Error("Please upload an image.")

    # Cap upload size (approximate: 3 bytes per pixel at native size)
    if image_np.nbytes > _MAX_UPLOAD_BYTES:
        raise gr.Error(
            f"Image is too large ({image_np.nbytes // 1024} KB). "
            f"Please upload an image smaller than {_MAX_UPLOAD_BYTES // 1024} KB."
        )

    # Validate that the array is actually an image
    if image_np.ndim not in (2, 3) or (image_np.ndim == 3 and image_np.shape[2] not in (1, 3, 4)):
        raise gr.Error("Unrecognised image format. Please upload a JPEG or PNG.")

    # Ensure RGB uint8
    pil_img = Image.fromarray(image_np).convert("RGB")
    image_np_rgb = np.array(pil_img, dtype=np.uint8)

    model = _get_model()

    # Apply eval transform: numpy → tensor (1, 3, 224, 224)
    result = _transform(image=image_np_rgb)
    tensor = result["image"].unsqueeze(0).to(_device)  # (1, 3, 224, 224)

    # Forward pass for probabilities
    with torch.no_grad():
        logits = model(tensor)
    probs = logits.softmax(dim=1).squeeze(0).cpu().tolist()

    label_probs = {cls: float(prob) for cls, prob in zip(CLASSES, probs)}

    # Grad-CAM overlay using predicted class
    pred_idx = int(logits.argmax(dim=1).item())
    _, _, overlay = generate_cam(model, tensor, target_class=pred_idx)
    overlay_uint8 = (overlay * 255).clip(0, 255).astype(np.uint8)

    return label_probs, overlay_uint8


# ── Gradio interface ─────────────────────────────────────────────────────────

_DISCLAIMER_MD = """
## Skin Lesion Classifier — HAM10000

> **EDUCATIONAL / RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS.**
> This tool has **not** been clinically validated. Results must not be used
> to make or inform medical decisions. Always consult a qualified
> dermatologist for any skin concern.

Upload a dermoscopy image to get class probabilities (7 HAM10000 categories)
and a Grad-CAM heatmap showing which regions influenced the prediction.
"""

_RESULT_DISCLAIMER = (
    "**Reminder:** This prediction is for research purposes only "
    "and is NOT a medical diagnosis."
)

_CLASS_DESCRIPTIONS = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc":   "Basal cell carcinoma",
    "bkl":   "Benign keratosis (solar lentigo / seborrhoeic keratosis / lichen planus-like)",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevi",
    "vasc":  "Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas)",
}

_EXAMPLES_DIR = ROOT / "assets" / "examples"


def _build_interface() -> gr.Blocks:
    with gr.Blocks(
        title="Skin Lesion Classifier — Educational/Research Only",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(_DISCLAIMER_MD)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="numpy",
                    label="Upload dermoscopy image",
                    image_mode="RGB",
                )
                submit_btn = gr.Button("Classify", variant="primary")

            with gr.Column(scale=1):
                label_output = gr.Label(
                    num_top_items=7,
                    label="Class probabilities",
                )
                cam_output = gr.Image(
                    type="numpy",
                    label="Grad-CAM overlay (predicted class)",
                )
                gr.Markdown(_RESULT_DISCLAIMER)

        # Wire up
        submit_btn.click(
            fn=_predict,
            inputs=[image_input],
            outputs=[label_output, cam_output],
        )
        # Also fire on image upload (convenience)
        image_input.change(
            fn=_predict,
            inputs=[image_input],
            outputs=[label_output, cam_output],
        )

        with gr.Accordion("Class legend", open=False):
            rows = "\n".join(
                f"| **{k}** | {v} |" for k, v in _CLASS_DESCRIPTIONS.items()
            )
            gr.Markdown(f"| Code | Description |\n|---|---|\n{rows}")

        with gr.Accordion("About this model", open=False):
            gr.Markdown(
                "**Architecture:** EfficientNet-B2 (timm, ImageNet pretrained), "
                "Dropout(0.3) + Linear(1408→7).\n\n"
                "**Training:** Two-stage fine-tune on HAM10000 (10,015 images, 7 classes). "
                "Stage 1: frozen backbone, LR 1e-3. Stage 2: full fine-tune, LR 1e-5. "
                "AdamW + CosineAnnealingLR, label smoothing ε=0.1, "
                "WeightedRandomSampler for class balance.\n\n"
                "**Metrics (target):** Balanced accuracy ≈ 0.81, Macro F1 ≈ 0.76.\n\n"
                "**Grad-CAM target layer:** `backbone.bn2` "
                "(BatchNormAct2d after final expansion conv).\n\n"
                "**Disclaimer:** This model was built for portfolio and educational "
                "purposes. It has **not** been validated for clinical use."
            )

    return demo


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch Gradio skin lesion demo")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (overrides CHECKPOINT env var)")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to serve on (default: 7860)")
    args = parser.parse_args()

    if args.checkpoint:
        os.environ["CHECKPOINT"] = args.checkpoint

    # Eagerly load model so startup errors are visible immediately
    try:
        _get_model()
    except FileNotFoundError as e:
        print(f"[app] WARNING: {e}")
        print("[app] App will launch but inference will fail until a checkpoint is provided.")

    demo = _build_interface()
    demo.launch(server_port=args.port, share=args.share)
