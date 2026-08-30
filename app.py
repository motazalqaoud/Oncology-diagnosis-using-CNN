"""
Gradio demo for oncology-diagnosis-cnn (Hugging Face Spaces entry point).

If checkpoints/best_model.pth is present (i.e. someone has run src/train.py on the
real HAM10000 dataset), predictions come from that trained model. If it is absent
-- which is the state of this repo out of the box, since the dataset is 2.7GB and
not bundled here -- the app runs in a clearly-labeled DEMO MODE: the architecture
and full inference pipeline run for real, but the classification head is untrained,
so the percentages shown are not meaningful. This mirrors the honest fallback
pattern used in the brain-tumor-segmentation Space in this same portfolio.
"""

import os
import sys

import gradio as gr
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from dataset import build_transforms, CLASS_NAMES, CLASS_DESCRIPTIONS, MALIGNANT_CLASSES  # noqa: E402
from model import load_model  # noqa: E402

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model, WEIGHTS_LOADED = load_model(num_classes=len(CLASS_NAMES),
                                    checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
transform = build_transforms(image_size=224, train=False)

MODE_BANNER = (
    "✅ **Trained model loaded** — predictions below reflect a model fine-tuned on HAM10000."
    if WEIGHTS_LOADED else
    "⚠️ **Demo mode — no trained checkpoint found.** The full pipeline (preprocessing, "
    "EfficientNet-B0 forward pass, softmax) is running for real, but the classification "
    "head has not been trained on skin lesion data, so the percentages below are not "
    "meaningful. Run `python src/train.py --data_dir <HAM10000 folder>` to produce "
    "`checkpoints/best_model.pth`, and this Space will automatically switch to real "
    "predictions."
)


def classify(image):
    if image is None:
        return {}, MODE_BANNER

    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    labels = {
        f"{cls} — {CLASS_DESCRIPTIONS[cls]}"
        + (" ⚠️" if cls in MALIGNANT_CLASSES else ""): float(p)
        for cls, p in zip(CLASS_NAMES, probs)
    }
    return labels, MODE_BANNER


with gr.Blocks(title="Oncology Diagnosis CNN — Skin Lesion Classifier") as demo:
    gr.Markdown("# Oncology Diagnosis CNN — Skin Lesion Classifier")
    gr.Markdown(
        "Upload a dermatoscopic image of a skin lesion. The model classifies it into one "
        "of the seven HAM10000 diagnostic categories and flags the three that are "
        "malignant or pre-malignant (akiec, bcc, mel)."
    )
    status = gr.Markdown(MODE_BANNER)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Skin lesion image")
            submit_btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            output_labels = gr.Label(num_top_classes=7, label="Predicted class probabilities")

    submit_btn.click(fn=classify, inputs=image_input, outputs=[output_labels, status])
    image_input.change(fn=classify, inputs=image_input, outputs=[output_labels, status])

    gr.Markdown(
        "---\n"
        "**This is a research and portfolio tool, not a medical device.** It has not been "
        "validated for clinical use and must never be used to make or defer an actual "
        "diagnosis. See a dermatologist for any concerning skin lesion.\n\n"
        "Dataset: [HAM10000](https://doi.org/10.7910/DVN/DBW86T) (Tschandl et al., 2018), "
        "10,015 dermatoscopic images across 7 diagnostic categories, verified by "
        "histopathology, follow-up, or expert consensus."
    )

if __name__ == "__main__":
    demo.launch()
