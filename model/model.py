"""
model/model.py — Fine-tuned EfficientNet-B0 for 7-class skin lesion classification.

Uses transfer learning with a custom classification head.
Supports freezing/unfreezing base layers for staged fine-tuning.
"""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


# 7 skin lesion classes from ISIC 2019 / HAM10000
CLASS_NAMES = [
    "Melanoma",
    "Melanocytic nevus",
    "Basal cell carcinoma",
    "Actinic keratosis",
    "Benign keratosis",
    "Dermatofibroma",
    "Vascular lesion",
]

NUM_CLASSES = len(CLASS_NAMES)


class SkinLesionModel(nn.Module):
    """
    EfficientNet-B0 backbone with a custom classification head.

    Architecture:
        EfficientNet-B0 (pretrained) → AdaptiveAvgPool → Dropout(0.3)
        → Linear(1280, 512) → BatchNorm → ReLU → Dropout(0.3)
        → Linear(512, 7)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super(SkinLesionModel, self).__init__()

        # Load pretrained EfficientNet-B0 backbone
        if pretrained:
            self.backbone = EfficientNet.from_pretrained("efficientnet-b0")
        else:
            self.backbone = EfficientNet.from_name("efficientnet-b0")

        # EfficientNet-B0 outputs 1280-dim features
        in_features = self.backbone._fc.in_features

        # Replace the default classifier with our custom head
        self.backbone._fc = nn.Identity()  # Remove original FC layer

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

        # Freeze all base layers initially for transfer learning
        self.freeze_base()

    def freeze_base(self):
        """Freeze all backbone parameters (for initial training with frozen features)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 2):
        """
        Unfreeze the last `n` MBConv blocks for fine-tuning.
        EfficientNet-B0 has 16 blocks (_blocks[0] to _blocks[15]).
        """
        # First, freeze everything
        self.freeze_base()

        # Unfreeze the last n blocks
        total_blocks = len(self.backbone._blocks)
        for i in range(total_blocks - n, total_blocks):
            for param in self.backbone._blocks[i].parameters():
                param.requires_grad = True

        # Always unfreeze batch norm in the head and the final conv + bn
        for param in self.backbone._bn1.parameters():
            param.requires_grad = True
        for param in self.backbone._conv_head.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Extract features using EfficientNet backbone
        features = self.backbone(x)  # (batch_size, 1280)
        # Pass through custom classifier head
        logits = self.classifier(features)  # (batch_size, num_classes)
        return logits


def get_model(pretrained: bool = True, fine_tune_blocks: int = 2) -> SkinLesionModel:
    """
    Factory function to create and configure the model.

    Args:
        pretrained: Whether to use ImageNet pretrained weights.
        fine_tune_blocks: Number of last blocks to unfreeze (0 = fully frozen).
    Returns:
        Configured SkinLesionModel instance.
    """
    model = SkinLesionModel(pretrained=pretrained)

    if fine_tune_blocks > 0:
        model.unfreeze_last_n_blocks(fine_tune_blocks)

    return model


if __name__ == "__main__":
    # Quick sanity check
    model = get_model(pretrained=False, fine_tune_blocks=2)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")  # Expected: (2, 7)
    print(f"Class names: {CLASS_NAMES}")

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
