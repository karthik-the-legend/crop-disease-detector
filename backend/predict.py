# ================================================================
# backend\predict.py
# Loads best_model.pth and predicts crop disease from an image
# Usage: python -m backend.predict --image path/to/leaf.jpg
# ================================================================
import json, argparse
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from pathlib import Path
from backend.transforms import INFERENCE_TRANSFORM

MODEL_PATH      = Path("models/best_model.pth")
CLASS_NAMES_PATH = Path("data/class_names.json")
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path: Path = MODEL_PATH):
    """Load trained ResNet-50 from checkpoint."""
    checkpoint  = torch.load(model_path, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(2048, num_classes)
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()
    return model, class_names

def predict(image_path: str, top_k: int = 3):
    """
    Predict crop disease from a leaf image.
    Returns top_k predictions with confidence scores.
    """
    model, class_names = load_model()

    image  = Image.open(image_path).convert("RGB")
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs     = model(tensor)
        probs       = torch.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, top_k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "class"     : class_names[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        })

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to leaf image")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    results = predict(args.image, args.top_k)
    print(f"\nPredictions for: {args.image}")
    print("-" * 40)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['class']:<45} {r['confidence']:>6.2f}%")