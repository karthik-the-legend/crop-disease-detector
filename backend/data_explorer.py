# ================================================================
# backend\data_explorer.py
# Counts images per class, saves chart to docs\class_distribution.png
# Run: python backend\data_explorer.py
# ================================================================
import json
import matplotlib.pyplot as plt
from pathlib import Path

PROC_DIR = Path("data/processed/train")
OUT_PATH = Path("docs/class_distribution.png")

def explore():
    class_counts = {}
    for cls_dir in sorted(PROC_DIR.iterdir()):
        if cls_dir.is_dir():
            count = len(list(cls_dir.glob("*.jpg")) +
                        list(cls_dir.glob("*.JPG")) +
                        list(cls_dir.glob("*.jpeg")))
            class_counts[cls_dir.name] = count

    print(f"\n{'Class':<50} {'Images':>6}")
    print("-" * 58)
    for cls, count in class_counts.items():
        print(f"{cls:<50} {count:>6}")
    print("-" * 58)
    print(f"{'TOTAL':<50} {sum(class_counts.values()):>6}")

    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.barh(list(class_counts.keys()), list(class_counts.values()), color="steelblue")
    ax.set_xlabel("Number of Images")
    ax.set_title("PlantVillage — Training Images per Class")
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"\n✓ Chart saved to {OUT_PATH}")

if __name__ == "__main__":
    explore()