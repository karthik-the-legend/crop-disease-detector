# ================================================================
# backend\prepare_data.py
# Splits PlantVillage into train/val/test (80/10/10)
# Run: python backend\prepare_data.py
# ================================================================
import os, json, shutil, random
from pathlib import Path
from tqdm import tqdm

RAW_DIR  = Path("data/raw/PlantVillage2/plantvillage dataset/color")
PROC_DIR = Path("data/processed")
SPLITS   = {"train": 0.80, "val": 0.10, "test": 0.10}
SEED     = 42
random.seed(SEED)

def prepare():
    class_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} class folders in {RAW_DIR}")

    class_names  = []
    total_images = 0

    for cls_dir in tqdm(class_dirs, desc="Splitting classes"):
        images = (list(cls_dir.glob("*.jpg")) +
                  list(cls_dir.glob("*.JPG")) +
                  list(cls_dir.glob("*.jpeg")))
        random.shuffle(images)
        n      = len(images)
        n_tr   = int(n * SPLITS["train"])
        n_val  = int(n * SPLITS["val"])

        split_map = {
            "train": images[:n_tr],
            "val"  : images[n_tr:n_tr+n_val],
            "test" : images[n_tr+n_val:]
        }

        for split, imgs in split_map.items():
            dest = PROC_DIR / split / cls_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest / img.name)

        class_names.append(cls_dir.name)
        total_images += n

    with open("data/class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\n✓ Total images split: {total_images}")
    print(f"✓ Classes           : {len(class_names)}")
    print(f"✓ class_names.json saved")
    print(f"✓ Train: {len(list((PROC_DIR/'train').glob('**/*.jpg')))} images")
    print(f"✓ Val  : {len(list((PROC_DIR/'val').glob('**/*.jpg')))} images")
    print(f"✓ Test : {len(list((PROC_DIR/'test').glob('**/*.jpg')))} images")

if __name__ == "__main__":
    prepare()