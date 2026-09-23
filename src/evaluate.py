"""
Evaluate a trained checkpoint on the held-out test split.

Reports both the 7-class metrics and the collapsed malignant-vs-benign metrics,
since sensitivity on the malignant classes (mel, bcc, akiec) is the number that
actually matters clinically -- a model that is 95% accurate overall but misses
malignant lesions is not a good screening tool.

Usage:
    python src/evaluate.py --data_dir /path/to/HAM10000 --checkpoint checkpoints/best_model.pth
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_fscore_support,
)

from preprocessing.dataset import HAM10000Dataset, CLASS_NAMES, MALIGNANT_CLASSES, stratified_split
from classification.model import SkinLesionClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_csv = os.path.join(args.data_dir, "HAM10000_metadata.csv")
    _train_idx, _val_idx, test_idx = stratified_split(metadata_csv)
    test_ds = HAM10000Dataset(metadata_csv, args.data_dir, indices=test_idx,
                               image_size=args.image_size, train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = SkinLesionClassifier(num_classes=len(CLASS_NAMES), pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    print("=" * 70)
    print("7-CLASS RESULTS")
    print("=" * 70)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=3, zero_division=0))

    print("Confusion matrix (rows=true, cols=predicted):")
    print("        " + "  ".join(f"{c:>6}" for c in CLASS_NAMES))
    cm = confusion_matrix(all_labels, all_preds)
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>6}  " + "  ".join(f"{v:>6}" for v in row))

    try:
        macro_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        print(f"\nMacro-average ROC-AUC (one-vs-rest): {macro_auc:.4f}")
    except ValueError as e:
        print(f"\nCould not compute macro AUC (likely a class missing from this test split): {e}")

    # Collapse to the clinically-actionable binary decision.
    malignant_idx = {CLASS_NAMES.index(c) for c in MALIGNANT_CLASSES}
    binary_labels = np.array([1 if l in malignant_idx else 0 for l in all_labels])
    binary_preds = np.array([1 if p in malignant_idx else 0 for p in all_preds])
    binary_malignant_prob = all_probs[:, list(malignant_idx)].sum(axis=1)

    print("\n" + "=" * 70)
    print("BINARY: MALIGNANT vs. BENIGN")
    print("=" * 70)
    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_labels, binary_preds, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(binary_labels, binary_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    try:
        binary_auc = roc_auc_score(binary_labels, binary_malignant_prob)
    except ValueError:
        binary_auc = float("nan")

    print(f"Sensitivity (recall on malignant): {sensitivity:.3f}")
    print(f"Specificity (recall on benign):    {specificity:.3f}")
    print(f"Precision:                          {precision:.3f}")
    print(f"F1:                                  {f1:.3f}")
    print(f"ROC-AUC:                             {binary_auc:.3f}")
    print(f"\nConfusion: TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"({fn} malignant lesion(s) in the test set were missed by the model -- "
          f"this is the number to scrutinize before considering any clinical use.)")


if __name__ == "__main__":
    main()
