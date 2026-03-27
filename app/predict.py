"""
app/predict.py — Inference pipeline for skin lesion classification.

Loads the best model checkpoint, preprocesses uploaded images,
and returns top-3 predictions with confidence scores.
"""

import os
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import SkinLesionModel, CLASS_NAMES, NUM_CLASSES


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

# Default checkpoint path (relative to project root)
DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints",
    "best_model.pth",
)


def get_inference_transforms() -> transforms.Compose:
    """Standard inference transforms: resize, normalize (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model(checkpoint_path: str = DEFAULT_CHECKPOINT, device: torch.device = None) -> SkinLesionModel:
    """
    Load a trained SkinLesionModel from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: Device to load the model onto.
    Returns:
        Model in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model without pretrained weights (we're loading our own)
    model = SkinLesionModel(num_classes=NUM_CLASSES, pretrained=False)

    # Load trained weights
    if not os.path.exists(checkpoint_path):
        print(f"☁️ Checkpoint not found at: {checkpoint_path}")
        print("   Downloading weights from Hugging Face hub (tejesh-c/skin-lesion-efficientnet)...")
        try:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            hf_hub_download(
                repo_id="tejesh-c/skin-lesion-efficientnet",
                filename="best_model.pth",
                local_dir=os.path.dirname(checkpoint_path)
            )
            print(f"✅ Download complete!")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✅ Model loaded from: {checkpoint_path}")
    else:
        print(f"⚠️  Checkpoint still not found at: {checkpoint_path}")
        print("   Using randomly initialized model (for demo purposes).")

    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL image for model inference.

    Args:
        image: PIL Image (any size, any mode).
    Returns:
        Tensor of shape (1, 3, 224, 224) ready for the model.
    """
    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = get_inference_transforms()
    tensor = transform(image)           # (3, 224, 224)
    tensor = tensor.unsqueeze(0)        # (1, 3, 224, 224)
    return tensor


def predict(
    model: SkinLesionModel,
    image: Image.Image,
    device: torch.device = None,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Run inference on a single image and return top-k predictions.

    Args:
        model: Trained SkinLesionModel in eval mode.
        image: PIL Image to classify.
        device: Device for inference.
        top_k: Number of top predictions to return.
    Returns:
        List of (class_name, confidence_percentage) tuples, sorted by confidence.
    """
    if device is None:
        device = next(model.parameters()).device

    # Preprocess and move to device
    input_tensor = preprocess_image(image).to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(input_tensor)                    # (1, 7)
        probabilities = F.softmax(logits, dim=1)        # (1, 7)

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

    results = []
    for i in range(top_k):
        class_idx = top_indices[0, i].item()
        confidence = top_probs[0, i].item() * 100       # Convert to percentage
        class_name = CLASS_NAMES[class_idx]
        results.append((class_name, round(confidence, 2)))

    return results


def predict_from_path(
    image_path: str,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Convenience function: load model, load image from path, predict.

    Args:
        image_path: Path to the image file.
        checkpoint_path: Path to model checkpoint.
        top_k: Number of top predictions to return.
    Returns:
        List of (class_name, confidence_percentage) tuples.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    image = Image.open(image_path)
    return predict(model, image, device, top_k)


if __name__ == "__main__":
    # Quick test with a dummy image
    import numpy as np

    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    )
    device = torch.device("cpu")
    model = load_model(DEFAULT_CHECKPOINT, device)
    results = predict(model, dummy_image, device)

    print("\n🔍 Top-3 Predictions:")
    for rank, (name, conf) in enumerate(results, 1):
        print(f"   {rank}. {name}: {conf:.2f}%")
