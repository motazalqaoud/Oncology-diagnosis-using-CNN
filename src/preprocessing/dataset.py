"""
HAM10000 dataset loader for skin lesion classification.

Expects the standard HAM10000 distribution:
  - HAM10000_metadata.csv  (columns: lesion_id, image_id, dx, dx_type, age, sex, localization)
  - HAM10000_images_part_1/  and  HAM10000_images_part_2/  (or a single flattened images/ folder)

Download from the Harvard Dataverse (Tschandl et al., 2018):
  https://doi.org/10.7910/DVN/DBW86T

The seven diagnostic classes (dx column):
  akiec  - Actinic keratoses / intraepithelial carcinoma
  bcc    - Basal cell carcinoma
  bkl    - Benign keratosis-like lesions
  df     - Dermatofibroma
  mel    - Melanoma
  nv     - Melanocytic nevi
  vasc   - Vascular lesions

Three of these (akiec, bcc, mel) are malignant/pre-malignant; the rest are benign.
This module exposes both the full 7-class label space and a derived binary
malignant/benign label, since clinically the binary decision (refer vs. reassure)
is often the more actionable one.
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
MALIGNANT_CLASSES = {"akiec", "bcc", "mel"}

CLASS_DESCRIPTIONS = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma (pre-malignant)",
    "bcc": "Basal cell carcinoma (malignant)",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma (benign)",
    "mel": "Melanoma (malignant)",
    "nv": "Melanocytic nevus / common mole (benign)",
    "vasc": "Vascular lesion (benign)",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _find_image_path(image_root: str, image_id: str) -> str:
    """HAM10000 ships as two image folders; check both, plus a flattened layout."""
    candidates = [
        os.path.join(image_root, f"{image_id}.jpg"),
        os.path.join(image_root, "HAM10000_images_part_1", f"{image_id}.jpg"),
        os.path.join(image_root, "HAM10000_images_part_2", f"{image_id}.jpg"),
        os.path.join(image_root, "images", f"{image_id}.jpg"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Could not locate image '{image_id}.jpg' under {image_root}. "
        "Expected HAM10000_images_part_1/, HAM10000_images_part_2/, or images/."
    )


class HAM10000Dataset(Dataset):
    """
    Args:
        metadata_csv: path to HAM10000_metadata.csv
        image_root: directory containing the image subfolders (see _find_image_path)
        indices: optional list of row indices to include (used for train/val/test splits)
        image_size: resize/crop target
        train: if True, applies training-time augmentation; else deterministic eval transform
        binary: if True, __getitem__ returns the malignant/benign label instead of the 7-class label
    """

    def __init__(self, metadata_csv, image_root, indices=None, image_size=224,
                 train=True, binary=False):
        self.df = pd.read_csv(metadata_csv)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        self.image_root = image_root
        self.transform = build_transforms(image_size, train=train)
        self.binary = binary

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = _find_image_path(self.image_root, row["image_id"])
        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        dx = row["dx"]
        if self.binary:
            label = 1 if dx in MALIGNANT_CLASSES else 0
        else:
            label = CLASS_TO_IDX[dx]
        return image, label

    def class_counts(self):
        """Per-class sample counts, used to build a class-weighted loss for the imbalance."""
        return self.df["dx"].value_counts().reindex(CLASS_NAMES, fill_value=0)


def stratified_split(metadata_csv, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Returns (train_idx, val_idx, test_idx) stratified by lesion_id so that images of the
    same lesion (HAM10000 has repeat photos of some lesions) never leak across splits.
    """
    import numpy as np
    df = pd.read_csv(metadata_csv)
    rng = np.random.RandomState(seed)

    lesion_to_dx = df.groupby("lesion_id")["dx"].first()
    lesion_ids = lesion_to_dx.index.to_numpy()
    dx_labels = lesion_to_dx.to_numpy()

    train_lesions, val_lesions, test_lesions = [], [], []
    for cls in CLASS_NAMES:
        cls_lesions = lesion_ids[dx_labels == cls]
        rng.shuffle(cls_lesions)
        n = len(cls_lesions)
        n_val = max(1, int(n * val_frac))
        n_test = max(1, int(n * test_frac))
        val_lesions.extend(cls_lesions[:n_val])
        test_lesions.extend(cls_lesions[n_val:n_val + n_test])
        train_lesions.extend(cls_lesions[n_val + n_test:])

    train_set, val_set, test_set = set(train_lesions), set(val_lesions), set(test_lesions)
    train_idx = df.index[df["lesion_id"].isin(train_set)].tolist()
    val_idx = df.index[df["lesion_id"].isin(val_set)].tolist()
    test_idx = df.index[df["lesion_id"].isin(test_set)].tolist()
    return train_idx, val_idx, test_idx
