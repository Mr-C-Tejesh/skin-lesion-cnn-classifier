"""
app/shap_explain.py — SHAP-based explainability for skin lesion predictions.

Uses SHAP's GradientExplainer to compute pixel-level contribution values,
showing which image regions most strongly drove the model's prediction.
"""

import os
import sys
import tempfile
from typing import Optional

import numpy as np
import torch
import shap
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import SkinLesionModel, CLASS_NAMES
from app.predict import preprocess_image, IMAGENET_MEAN, IMAGENET_STD


def create_background_data(n_samples: int = 50, image_size: int = 224) -> torch.Tensor:
    """
    Create a small background dataset for the SHAP explainer.

    Uses random noise normalized with ImageNet stats as a baseline.
    In production, you'd use a sample from the actual training set.

    Args:
        n_samples: Number of background samples.
        image_size: Size of each sample image.
    Returns:
        Tensor of shape (n_samples, 3, image_size, image_size).
    """
    # Generate random images and apply ImageNet normalization
    background = torch.randn(n_samples, 3, image_size, image_size)

    # Apply ImageNet normalization to match model expectations
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    background = background * std + mean

    return background


def generate_shap_plot(
    model: SkinLesionModel,
    image: Image.Image,
    target_class: Optional[int] = None,
    device: torch.device = None,
    n_background: int = 50,
    save_path: Optional[str] = None,
) -> str:
    """
    Generate a SHAP explanation plot for the model's prediction.

    Args:
        model: Trained SkinLesionModel.
        image: PIL Image to explain.
        target_class: Class index to explain (None = predicted class).
        device: Computation device.
        n_background: Number of background samples for the explainer.
        save_path: Path to save the plot. If None, saves to a temp file.
    Returns:
        Path to the saved SHAP plot image.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Preprocess the input image
    input_tensor = preprocess_image(image).to(device)  # (1, 3, 224, 224)

    # Determine target class if not specified
    if target_class is None:
        with torch.no_grad():
            logits = model(input_tensor)
            target_class = logits.argmax(dim=1).item()

    # Create background data
    background = create_background_data(n_background).to(device)

    # Force SHAP computation to CPU (GradientExplainer hooks often hang on MPS)
    cpu_device = torch.device("cpu")
    model_cpu = model.to(cpu_device)
    input_cpu = input_tensor.to(cpu_device)
    background_cpu = background.to(cpu_device)

    # Initialize SHAP GradientExplainer
    explainer = shap.GradientExplainer(model_cpu, background_cpu)

    # Compute SHAP values for the input image
    shap_values = explainer.shap_values(input_cpu)

    # Restore model to original device
    model.to(device)

    # shap_values is a list of arrays (one per class)
    # Get SHAP values for the target class
    if isinstance(shap_values, list):
        shap_for_class = shap_values[target_class]  # (1, 3, 224, 224)
    else:
        shap_for_class = shap_values

    # Convert to numpy and prepare for visualization
    shap_np = shap_for_class[0]  # (3, 224, 224)

    # Denormalize input image for display
    input_np = input_tensor[0].cpu().numpy()  # (3, 224, 224)
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    input_denorm = (input_np * std + mean)
    input_denorm = np.clip(input_denorm, 0, 1)

    # Transpose from (C, H, W) to (H, W, C) for matplotlib
    input_display = input_denorm.transpose(1, 2, 0)
    shap_display = shap_np.transpose(1, 2, 0)

    # Create the SHAP plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(input_display)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # SHAP values heatmap (mean across channels)
    shap_heatmap = np.mean(np.abs(shap_display), axis=2)
    im = axes[1].imshow(shap_heatmap, cmap="hot", interpolation="bilinear")
    axes[1].set_title("SHAP Value Magnitude", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # SHAP overlay on original image
    shap_overlay = input_display.copy()
    shap_normalized = shap_heatmap / (shap_heatmap.max() + 1e-8)
    # Create red overlay for important regions
    for c in range(3):
        shap_overlay[:, :, c] = (
            0.6 * input_display[:, :, c] + 0.4 * shap_normalized * (1 if c == 0 else 0.2)
        )
    shap_overlay = np.clip(shap_overlay, 0, 1)

    axes[2].imshow(shap_overlay)
    axes[2].set_title(
        f"SHAP Overlay — {CLASS_NAMES[target_class]}",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].axis("off")

    plt.suptitle(
        f"SHAP Explanation for: {CLASS_NAMES[target_class]}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    # Save plot
    if save_path is None:
        save_path = os.path.join(tempfile.gettempdir(), "shap_explanation.png")

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return save_path


if __name__ == "__main__":
    # Quick test with dummy model and image
    model = SkinLesionModel(pretrained=False)
    model.eval()

    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    )

    output_path = generate_shap_plot(
        model, dummy_image, device=torch.device("cpu"), n_background=10
    )
    print(f"✅ SHAP plot saved to: {output_path}")
