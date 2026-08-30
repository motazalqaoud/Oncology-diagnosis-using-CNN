"""
Model definition: EfficientNet-B0 backbone (ImageNet-pretrained) with a replaced
classification head, fine-tuned for HAM10000 skin lesion classification.

EfficientNet-B0 is chosen over deeper backbones (ResNet50, EfficientNet-B4+) as a
practical default: it trains fast on a single GPU, is small enough to serve cheaply
in a Hugging Face Space, and the literature (see README) shows it is competitive
with heavier architectures on this specific dataset once fine-tuned.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class SkinLesionClassifier(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )
        self.backbone = backbone

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("classifier"):
                    param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


def load_model(num_classes: int = 7, checkpoint_path: str = None, device: str = "cpu",
                pretrained_backbone: bool = True):
    """
    Instantiate the model and, if a checkpoint exists on disk, load trained weights
    on top. If a checkpoint is found, the backbone is built without downloading
    ImageNet weights first, since the checkpoint's full state dict (backbone + head)
    is about to overwrite them anyway -- this also means a trained checkpoint can be
    loaded on a machine with no internet access. If no checkpoint is found, the
    backbone falls back to ImageNet-pretrained features (pretrained_backbone=True)
    rather than random initialization, so demo mode still has sane low-level visual
    features even though the classification head is untrained.

    Returns (model, weights_loaded: bool) so callers (e.g. the Gradio app) can decide
    whether to show real predictions or fall back to a clearly-labeled demo mode.
    """
    import os
    checkpoint_exists = bool(checkpoint_path) and os.path.exists(checkpoint_path)

    model = SkinLesionClassifier(
        num_classes=num_classes,
        pretrained=(pretrained_backbone and not checkpoint_exists),
    )
    weights_loaded = False

    if checkpoint_exists:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        weights_loaded = True

    model.to(device)
    model.eval()
    return model, weights_loaded
