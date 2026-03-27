"""
app/gradcam.py — Grad-CAM visualization for EfficientNet-B0.

Generates a heatmap overlay on the original image showing which regions
of the input most influenced the model's prediction.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
"""

import os
import sys
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import SkinLesionModel
from app.predict import preprocess_image


class GradCAM:
    """
    Grad-CAM implementation for EfficientNet-B0.

    Hooks into the last convolutional block to capture activations
    and gradients, then computes a class-discriminative heatmap.
    """

    def __init__(self, model: SkinLesionModel):
        self.model = model
        self.model.eval()

        # Storage for hooked values
        self.activations = None
        self.gradients = None

        # Register hooks on the last MBConv block of EfficientNet-B0
        # This is the last block before the final conv + pooling
        target_layer = self.model.backbone._blocks[-1]

        # Forward hook: capture activations
        target_layer.register_forward_hook(self._save_activations)
        # Backward hook: capture gradients
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        """Forward hook to store feature map activations."""
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        """Backward hook to store gradients."""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image: Image.Image,
        target_class: Optional[int] = None,
        device: torch.device = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given image.

        Args:
            image: PIL Image to analyze.
            target_class: Class index to generate heatmap for.
                          If None, uses the predicted class.
            device: Device for computation.
        Returns:
            Heatmap as a numpy array of shape (H, W) with values in [0, 1].
        """
        if device is None:
            device = next(self.model.parameters()).device

        # Preprocess image
        input_tensor = preprocess_image(image).to(device)
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)  # (1, num_classes)

        # Use predicted class if no target specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero existing gradients
        self.model.zero_grad()

        # Backward pass for the target class
        target_score = output[0, target_class]
        target_score.backward()

        # Compute Grad-CAM
        # Global average pool the gradients → channel importance weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam


def generate_gradcam_overlay(
    model: SkinLesionModel,
    image: Image.Image,
    target_class: Optional[int] = None,
    alpha: float = 0.5,
    device: torch.device = None,
) -> Image.Image:
    """
    Generate a Grad-CAM heatmap overlaid on the original image.

    Args:
        model: Trained SkinLesionModel.
        image: Original PIL Image.
        target_class: Class to generate heatmap for (None = predicted class).
        alpha: Blending factor for overlay (0=original, 1=heatmap).
        device: Computation device.
    Returns:
        PIL Image with Grad-CAM heatmap overlay.
    """
    # Generate heatmap
    gradcam = GradCAM(model)
    heatmap = gradcam.generate(image, target_class, device)

    # Resize heatmap to match original image size
    original_np = np.array(image.convert("RGB"))
    h, w = original_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert heatmap to color (JET colormap)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend with original image
    overlay = np.uint8(alpha * heatmap_colored + (1 - alpha) * original_np)

    return Image.fromarray(overlay)


if __name__ == "__main__":
    # Quick test with dummy image
    from model.model import SkinLesionModel

    model = SkinLesionModel(pretrained=False)
    model.eval()

    # Create a random test image
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    )

    overlay = generate_gradcam_overlay(model, dummy_image, device=torch.device("cpu"))
    print(f"✅ Grad-CAM overlay shape: {np.array(overlay).shape}")
    overlay.save("/tmp/gradcam_test.png")
    print("💾 Saved test overlay to /tmp/gradcam_test.png")
