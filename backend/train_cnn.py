# ================================================================
# backend\train_cnn.py
# Fine-tunes ResNet-50 on PlantVillage (38 crop disease classes)
# Run: python -m backend.train_cnn
# Estimated time: 2-4 hours on CPU (15 epochs)
# ================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet50, ResNet50_Weights

from backend.dataset import get_dataloaders

#  Config 
NUM_EPOCHS    = 15
LEARNING_RATE = 0.001
NUM_CLASSES   = 38
MODEL_DIR     = Path("models")
LOG_DIR       = Path("logs")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

print(f"Using device: {DEVICE}")


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Load ResNet-50 pretrained on ImageNet.
    Freeze early layers, unfreeze layer3 + layer4 + fc.

    Why freeze early layers?
    layer1/layer2 detect edges and textures — universal features
    already learned from ImageNet. Unfreezing wastes training time.
    layer3/layer4 detect high-level disease-specific features
    like lesion shapes and colour patterns — these need fine-tuning.
    """
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer3, layer4, fc
    for layer_name in ["layer3", "layer4", "fc"]:
        for param in getattr(model, layer_name).parameters():
            param.requires_grad = True

    # Replace final FC: 2048  38 classes
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(2048, num_classes),
    )

    return model.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer) -> tuple:
    """Run one training epoch. Returns (loss, accuracy)."""
    model.train()
    running_loss = running_correct = total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss    += loss.item() * images.size(0)
        _, preds         = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        total           += images.size(0)

    return running_loss / total, running_correct / total


def evaluate(model, loader, criterion) -> tuple:
    """Evaluate on val/test set. Returns (loss, accuracy)."""
    model.eval()
    running_loss = running_correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs        = model(images)
            loss           = criterion(outputs, labels)

            running_loss    += loss.item() * images.size(0)
            _, preds         = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            total           += images.size(0)

    return running_loss / total, running_correct / total


def main():
    loaders, class_names, sizes = get_dataloaders()
    print(f"Classes: {len(class_names)} | Train: {sizes['train']} | Val: {sizes['val']}")

    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0
    training_log = []

    print(f"\n{'='*55}")
    print(f"  Training ResNet-50 on {NUM_CLASSES} crop disease classes")
    print(f"  Device: {DEVICE} | Epochs: {NUM_EPOCHS}")
    print(f"{'='*55}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer
        )
        val_loss, val_acc = evaluate(model, loaders["val"], criterion)
        scheduler.step()

        elapsed = round(time.time() - t0)
        print(
            f"Ep {epoch:02d}/{NUM_EPOCHS} | "
            f"Train {train_acc:.3f} ({train_loss:.4f}) | "
            f"Val {val_acc:.3f} ({val_loss:.4f}) | "
            f"{elapsed}s"
        )

        training_log.append({
            "epoch"     : epoch,
            "train_acc" : round(train_acc, 4),
            "val_acc"   : round(val_acc, 4),
            "train_loss": round(train_loss, 4),
            "val_loss"  : round(val_loss, 4),
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_acc"    : val_acc,
                "class_names": class_names,
            }, MODEL_DIR / "best_model.pth")
            print(f"   New best saved: {val_acc:.3f}")

        # Save log after every epoch
        with open(LOG_DIR / "training_log.json", "w") as f:
            json.dump({
                "best_val_acc": best_val_acc,
                "num_epochs"  : NUM_EPOCHS,
                "num_classes" : NUM_CLASSES,
                "device"      : str(DEVICE),
                "history"     : training_log,
            }, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Best Val Acc : {best_val_acc:.4f}")
    print(f"  Model saved  : models/best_model.pth")
    print(f"  Log saved    : logs/training_log.json")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
