"""
Run inference on a single image from the command line.

Usage:
    python src/predict.py --image path/to/lesion.jpg --checkpoint checkpoints/best_model.pth
"""

import argparse

import torch
from PIL import Image

from preprocessing.dataset import build_transforms, CLASS_NAMES, CLASS_DESCRIPTIONS, MALIGNANT_CLASSES
from classification.model import load_model


def predict(image_path: str, checkpoint_path: str, image_size: int = 224, device: str = "cpu"):
    model, weights_loaded = load_model(num_classes=len(CLASS_NAMES),
                                        checkpoint_path=checkpoint_path, device=device)
    if not weights_loaded:
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Train a model first with src/train.py, "
            "or point --checkpoint at an existing best_model.pth."
        )

    transform = build_transforms(image_size, train=False)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    results = sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = predict(args.image, args.checkpoint, args.image_size, device)

    print(f"\nPredictions for {args.image}:\n")
    for cls, prob in results:
        flag = " [MALIGNANT]" if cls in MALIGNANT_CLASSES else ""
        print(f"  {cls:6s} {prob*100:5.1f}%  {CLASS_DESCRIPTIONS[cls]}{flag}")

    top_class = results[0][0]
    print(f"\nTop prediction: {top_class} ({CLASS_DESCRIPTIONS[top_class]})")
    print("\nThis is a research tool, not a diagnosis. See a dermatologist for any "
          "concerning skin lesion.")


if __name__ == "__main__":
    main()
