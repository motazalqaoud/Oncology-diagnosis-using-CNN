"""
Train the skin lesion classifier on HAM10000.

Usage:
    python src/train.py --data_dir /path/to/HAM10000 --epochs 30 --batch_size 32

Expects --data_dir to contain HAM10000_metadata.csv and the image folder(s)
(see dataset.py for the exact expected layout).

Handles the well-documented class imbalance in HAM10000 (nv makes up ~67% of the
dataset, df and vasc each under 2%) via inverse-frequency class weighting in the
loss function, rather than naive oversampling, to avoid overfitting on duplicated
minority-class images.
"""

import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocessing.dataset import HAM10000Dataset, CLASS_NAMES, stratified_split
from classification.model import SkinLesionClassifier


def compute_class_weights(train_dataset, device):
    counts = train_dataset.class_counts().to_numpy().astype(float)
    counts = counts.clip(min=1)  # avoid div-by-zero if a class is absent from a small subset
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer, device, train: bool, scaler=None):
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    use_amp = scaler is not None and scaler.is_enabled()

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if train:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory with HAM10000_metadata.csv and images")
    parser.add_argument("--config", default=None,
                         help="Path to a JSON preset (see configs/) to set epochs/batch_size/lr/amp defaults. "
                              "Any explicit --flag still overrides the preset's value.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--freeze_backbone", action="store_true",
                         help="Only train the classifier head (faster, useful for a quick sanity run)")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=6, help="Early-stopping patience on val loss")
    parser.add_argument("--amp", action="store_true", default=None,
                         help="Use mixed precision (overrides config if set)")
    args = parser.parse_args()

    # Config file supplies defaults; explicit CLI flags (checked via the None
    # sentinel above) always win over the preset.
    config_defaults = {"epochs": 30, "batch_size": 32, "lr": 3e-4, "num_workers": 4, "amp": False}
    if args.config:
        with open(args.config) as f:
            preset = json.load(f)
        for key in ("epochs", "batch_size", "lr", "num_workers", "amp"):
            if key in preset:
                config_defaults[key] = preset[key]
        print(f"Loaded config preset: {args.config} ({preset.get('_comment', '')})")

    args.epochs = args.epochs if args.epochs is not None else config_defaults["epochs"]
    args.batch_size = args.batch_size if args.batch_size is not None else config_defaults["batch_size"]
    args.lr = args.lr if args.lr is not None else config_defaults["lr"]
    args.num_workers = args.num_workers if args.num_workers is not None else config_defaults["num_workers"]
    args.amp = args.amp if args.amp is not None else config_defaults["amp"]

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metadata_csv = os.path.join(args.data_dir, "HAM10000_metadata.csv")
    train_idx, val_idx, _test_idx = stratified_split(metadata_csv)
    print(f"Split sizes -> train: {len(train_idx)}  val: {len(val_idx)}  test: {len(_test_idx)}")

    train_ds = HAM10000Dataset(metadata_csv, args.data_dir, indices=train_idx,
                                image_size=args.image_size, train=True)
    val_ds = HAM10000Dataset(metadata_csv, args.data_dir, indices=val_idx,
                              image_size=args.image_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = SkinLesionClassifier(num_classes=len(CLASS_NAMES), pretrained=True,
                                  freeze_backbone=args.freeze_backbone).to(device)

    class_weights = compute_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    use_amp = args.amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.amp and not torch.cuda.is_available():
        print("Note: --amp was requested but no GPU is available; running in full precision.")

    log_path = os.path.join(args.output_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"])

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False, scaler=scaler)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"lr {current_lr:.2e} | {elapsed:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, val_loss, val_acc, current_lr, elapsed])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc},
                os.path.join(args.output_dir, "best_model.pth"),
            )
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: no val_loss improvement in {args.patience} epochs.")
                break

    print(f"Training complete. Best val_loss={best_val_loss:.4f}. "
          f"Checkpoint saved to {os.path.join(args.output_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
