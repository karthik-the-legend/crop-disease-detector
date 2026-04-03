# ================================================================
# backend\dataset.py
# Creates PyTorch DataLoaders for train, val, test splits
# Uses torchvision.datasets.ImageFolder
# ================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets
from backend.transforms import TRAIN_TRANSFORMS, VAL_TRANSFORMS

DATA_DIR    = Path("data/processed")
BATCH_SIZE  = 32
NUM_WORKERS = 0   # Must be 0 on Windows — multiprocessing issues with >0


def get_dataloaders(batch_size: int = BATCH_SIZE):
    """
    Returns train, val, test DataLoaders and class_names list.

    ImageFolder reads the folder structure:
        data/processed/train/Tomato___Early_blight/img001.jpg
        data/processed/train/Tomato___healthy/img001.jpg

    Returns:
        loaders      : dict with keys 'train', 'val', 'test'
        class_names  : list of 38 class name strings (sorted alphabetically)
        dataset_sizes: dict with image counts per split
    """
    loaders       = {}
    dataset_sizes = {}
    class_names   = None

    for split in ["train", "val", "test"]:
        transform = TRAIN_TRANSFORMS if split == "train" else VAL_TRANSFORMS
        dataset   = datasets.ImageFolder(
            root      = str(DATA_DIR / split),
            transform = transform,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = NUM_WORKERS,
            pin_memory  = False,
        )
        dataset_sizes[split] = len(dataset)
        if split == "train":
            class_names = dataset.classes

    return loaders, class_names, dataset_sizes


if __name__ == "__main__":
    loaders, class_names, sizes = get_dataloaders()

    print(f"Classes     : {len(class_names)}")
    print(f"Train       : {sizes['train']} images")
    print(f"Val         : {sizes['val']} images")
    print(f"Test        : {sizes['test']} images")

    # Test one batch
    images, labels = next(iter(loaders["train"]))
    print(f"Batch shape : {images.shape}")   # [32, 3, 224, 224]
    print(f"Labels shape: {labels.shape}")   # [32]
    print(f"First class : {class_names[0]}")
