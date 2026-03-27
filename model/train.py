"""
model/train.py — Training pipeline for the Skin Lesion Classifier.

Features:
    - Data augmentation (flip, rotation, color jitter)
    - Class imbalance handling via WeightedRandomSampler
    - Two-phase training: frozen backbone → fine-tuned last 2 blocks
    - Early stopping with best model checkpoint saving
    - Train/val split with comprehensive logging
"""

import os
import sys
import copy
import time
import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import get_model, CLASS_NAMES, NUM_CLASSES


# ─── Configuration ───────────────────────────────────────────────────────────

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training hyperparameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE_FROZEN = 1e-3      # Higher LR when backbone is frozen
LEARNING_RATE_FINETUNE = 1e-4    # Lower LR for fine-tuning
WEIGHT_DECAY = 1e-4
NUM_EPOCHS_FROZEN = 1           # Phase 1: train classifier head only (Reduced for quick verification)
NUM_EPOCHS_FINETUNE = 20         # Phase 2: fine-tune last 2 blocks
EARLY_STOP_PATIENCE = 7
VAL_SPLIT = 0.2                  # 80/20 train/val split


# ─── Data Transforms ─────────────────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """Training transforms with data augmentation."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """Validation transforms (no augmentation, only resize + normalize)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── Weighted Sampler for Class Imbalance ─────────────────────────────────────

def create_weighted_sampler(dataset) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to handle class imbalance.
    Oversamples minority classes so each class is seen equally during training.
    """
    # Count samples per class
    targets = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(targets)
    total_samples = len(targets)

    # Compute weight for each class (inverse frequency)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    # Assign weight to each sample based on its class
    sample_weights = [class_weights[t] for t in targets]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float64)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


# ─── Early Stopping ──────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = EARLY_STOP_PATIENCE, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            # Improvement found — reset counter
            self.best_loss = val_loss
            self.counter = 0
        else:
            # No improvement — increment counter
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ─── Training & Validation Loops ──────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple:
    """
    Train for one epoch.

    Returns:
        (avg_loss, accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Validate the model.

    Returns:
        (avg_loss, accuracy) for the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


# ─── Main Training Pipeline ──────────────────────────────────────────────────

def train_model(data_dir: str, output_dir: str = "checkpoints"):
    """
    Full two-phase training pipeline.

    Phase 1: Train only the classifier head (backbone frozen).
    Phase 2: Fine-tune last 2 blocks + classifier head.

    Args:
        data_dir: Path to dataset root (expects ImageFolder structure).
        output_dir: Directory to save model checkpoints.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Device selection: CUDA > MPS (Mac) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"🔧 Using device: {device}")

    # ── Load Dataset ──────────────────────────────────────────────────────
    print(f"📂 Loading dataset from: {data_dir}")

    # Load full dataset with training transforms first (we'll override for val)
    full_dataset = datasets.ImageFolder(root=data_dir, transform=get_train_transforms())

    # Print class distribution
    class_counts = Counter(full_dataset.targets)
    print("\n📊 Class distribution:")
    for idx, name in enumerate(CLASS_NAMES):
        count = class_counts.get(idx, 0)
        print(f"   {name}: {count} images")

    # Split into train/val
    total_size = len(full_dataset)
    val_size = int(total_size * VAL_SPLIT)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Override val transforms (no augmentation)
    val_dataset.dataset = copy.copy(full_dataset)
    val_dataset.dataset.transform = get_val_transforms()

    print(f"\n📦 Train: {train_size} | Val: {val_size}")

    # Create weighted sampler for training data (handles class imbalance)
    # We need to get the targets for the train subset
    train_targets = [full_dataset.targets[i] for i in train_dataset.indices]

    class_counts_train = Counter(train_targets)
    total_train = len(train_targets)
    class_weights = {cls: total_train / count for cls, count in class_counts_train.items()}
    sample_weights = [class_weights[t] for t in train_targets]
    sample_weights = torch.tensor(sample_weights, dtype=torch.float64)

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ── Initialize Model ──────────────────────────────────────────────────
    model = get_model(pretrained=True, fine_tune_blocks=0)  # Start fully frozen
    model = model.to(device)

    # Class-weighted loss function
    class_weight_tensor = torch.tensor(
        [class_weights.get(i, 1.0) for i in range(NUM_CLASSES)],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

    best_val_acc = 0.0
    best_model_state = None

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Train classifier head only (backbone frozen)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🧊 PHASE 1: Training classifier head (backbone frozen)")
    print("=" * 60)

    # Only optimize classifier parameters
    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE_FROZEN,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    for epoch in range(NUM_EPOCHS_FROZEN):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        elapsed = time.time() - start_time
        print(
            f"  Epoch [{epoch+1}/{NUM_EPOCHS_FROZEN}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            save_path = os.path.join(output_dir, "best_model.pth")
            torch.save(best_model_state, save_path)
            print(f"  ✅ New best model! Val Acc: {best_val_acc:.2f}% | Saved to {save_path}")

        # Early stopping check
        if early_stopping(val_loss):
            print(f"  ⏹ Early stopping triggered at epoch {epoch+1}")
            break

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: Fine-tune last 2 blocks + classifier
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🔥 PHASE 2: Fine-tuning last 2 blocks + classifier")
    print("=" * 60)

    # Load best model from Phase 1
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Unfreeze last 2 blocks
    model.unfreeze_last_n_blocks(2)

    # New optimizer with lower learning rate for fine-tuning
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE_FINETUNE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    for epoch in range(NUM_EPOCHS_FINETUNE):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        elapsed = time.time() - start_time
        print(
            f"  Epoch [{epoch+1}/{NUM_EPOCHS_FINETUNE}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  ✅ New best model! Val Acc: {best_val_acc:.2f}%")

        # Early stopping check
        if early_stopping(val_loss):
            print(f"  ⏹ Early stopping triggered at epoch {epoch+1}")
            break

    # ── Save Final Best Model ─────────────────────────────────────────────
    if best_model_state is not None:
        save_path = os.path.join(output_dir, "best_model.pth")
        torch.save(best_model_state, save_path)
        print(f"\n💾 Best model saved to: {save_path}")
        print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    else:
        print("\n⚠️  No model was saved (training may not have completed).")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Skin Lesion Classifier")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset root (ImageFolder structure with class subdirectories)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints (default: checkpoints/)",
    )
    args = parser.parse_args()

    train_model(data_dir=args.data_dir, output_dir=args.output_dir)
