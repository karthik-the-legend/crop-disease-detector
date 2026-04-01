# ================================================================
# backend\dataset.py
# Creates PyTorch DataLoaders for train, val, test splits
# ================================================================
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets
from backend.transforms import TRAIN_TRANSFORMS, VAL_TRANSFORMS

DATA_DIR    = Path("data/processed")
BATCH_SIZE  = 32
NUM_WORKERS = 0  # Must be 0 on Windows

def get_dataloaders(batch_size: int = BATCH_SIZE):
    loaders       = {}
    dataset_sizes = {}
    class_names   = None

    for split in ["train", "val", "test"]:
        transform = TRAIN_TRANSFORMS if split == "train" else VAL_TRANSFORMS
        dataset   = datasets.ImageFolder(
            root=str(DATA_DIR / split),
            transform=transform
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=NUM_WORKERS,
            pin_memory=False
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

    images, labels = next(iter(loaders["train"]))
    print(f"Batch shape : {images.shape}")   # must be [32, 3, 224, 224]
    print(f"Labels shape: {labels.shape}")   # must be [32]
    print(f"First class : {class_names[0]}")