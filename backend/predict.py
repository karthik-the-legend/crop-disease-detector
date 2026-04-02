# ================================================================
# backend\predict.py
# Loads trained ResNet-50, predicts disease from any crop image
# Singleton pattern: model loaded once at startup, reused
# ================================================================
import io, json, glob
from pathlib import Path
from typing import Union
import torch
import torch.nn.functional as F
from torchvision.models import resnet50
from PIL import Image
from backend.transforms import INFERENCE_TRANSFORM

MODEL_PATH     = Path("models/best_model.pth")
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESHOLD = 0.40

# ── Singleton ───────────────────────────────────────────────────
_MODEL       = None
_CLASS_NAMES = None

def load_model():
    global _MODEL, _CLASS_NAMES
    if _MODEL is None:
        print("[Predictor] Loading best_model.pth...")
        checkpoint   = torch.load(MODEL_PATH, map_location=DEVICE)
        _CLASS_NAMES = checkpoint["class_names"]

        model = resnet50()
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(2048, len(_CLASS_NAMES))
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(DEVICE)
        model.eval()
        _MODEL = model
        print(f"[Predictor] Loaded. Classes: {len(_CLASS_NAMES)}, Val Acc: {checkpoint['val_acc']:.3f}")
    return _MODEL, _CLASS_NAMES

def predict(image_input: Union[str, Path, bytes], top_k: int = 3) -> dict:
    """
    Predict crop disease from image.
    Args:
        image_input: file path (str/Path) OR raw bytes (from API upload)
        top_k: how many top predictions to return
    Returns dict with disease, confidence, top3, is_healthy, low_confidence
    """
    model, class_names = load_model()

    if isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input)).convert("RGB")
    else:
        img = Image.open(image_input).convert("RGB")

    tensor = INFERENCE_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits             = model(tensor)
        probs              = F.softmax(logits, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, k=top_k)

    top3 = [
        {"disease": class_names[i.item()], "confidence": round(p.item(), 4)}
        for p, i in zip(top_probs, top_idxs)
    ]
    best = top3[0]

    return {
        "disease"       : best["disease"],
        "confidence"    : best["confidence"],
        "top3"          : top3,
        "is_healthy"    : "healthy" in best["disease"].lower(),
        "low_confidence": best["confidence"] < CONF_THRESHOLD,
    }

if __name__ == "__main__":
    test_images = glob.glob("data/processed/test/**/*.jpg", recursive=True)[:10]
    for img_path in test_images:
        result     = predict(img_path)
        true_label = Path(img_path).parent.name
        correct    = true_label == result["disease"]
        print(f"{'✓' if correct else '✗'} True: {true_label[:40]:<40} Pred: {result['disease'][:35]} ({result['confidence']:.2f})")