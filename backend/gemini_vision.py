# ================================================================
# backend\gemini_vision.py
# Sends crop image to Groq Vision for visual explanation
# Called ONLY when CNN confidence < CNN_CONFIDENCE_HIGH (0.85)
# High confidence: trust CNN (fast). Low: get visual second opinion.
# ================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import io
from typing import Optional

from groq import Groq
from PIL import Image

from backend.config import GROQ_API_KEY, VISION_MODEL, CNN_CONFIDENCE_HIGH

_client = Groq(api_key=GROQ_API_KEY)

#  Vision prompt 
VISION_PROMPT_TEMPLATE = """You are an agricultural plant pathologist examining a crop leaf image.
The CNN model predicted: {disease_display_name}
CNN Confidence: {confidence_pct}%

Please:
1. Describe the specific visual symptoms you can see in this leaf image
   (lesion colour, shape, location, pattern, leaf discolouration, spots, mold, etc.)
2. State whether these symptoms are CONSISTENT or INCONSISTENT with the
   CNN prediction of {disease_display_name}
3. If inconsistent, suggest what the disease might be based on what you see.

Keep your response under 120 words. Be specific about what you see."""


def _image_to_base64(image_bytes: bytes) -> str:
    """Convert raw image bytes to base64 string for Groq API."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def explain_with_vision(
    image_bytes : bytes,
    disease_name: str,
    confidence  : float,
) -> Optional[str]:
    """
    Get Groq Vision's visual explanation of a crop disease image.
    Called only when CNN confidence < CNN_CONFIDENCE_HIGH (0.85).

    Args:
        image_bytes : Raw image bytes (JPEG/PNG)
        disease_name: PlantVillage class name e.g. "Tomato___Early_blight"
        confidence  : CNN top-1 confidence (0.0-1.0)

    Returns:
        str  : Visual explanation from Groq Vision
        None : If confidence is high (skip) or API call fails
    """
    # Skip if high confidence — CNN is trusted, save API call
    if confidence >= CNN_CONFIDENCE_HIGH:
        return None

    display  = (
        disease_name
        .replace("___", " — ")
        .replace("__",  " ")
        .replace("_",   " ")
    )
    conf_pct = round(confidence * 100, 1)
    prompt   = VISION_PROMPT_TEMPLATE.format(
        disease_display_name=display,
        confidence_pct=conf_pct,
    )

    try:
        b64_image = _image_to_base64(image_bytes)

        response = _client.chat.completions.create(
            model    = VISION_MODEL,
            messages = [
                {
                    "role"   : "user",
                    "content": [
                        {
                            "type"    : "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            max_tokens  = 200,
            temperature = 0.1,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        # Graceful degradation — vision failure must not break pipeline
        print(f"[GroqVision] Error: {e}")
        return None


if __name__ == "__main__":
    import glob

    # Test high-confidence skip
    print("Test: high confidence skip (conf=0.92) — must return None")
    result = explain_with_vision(b"fake", "Tomato___Early_blight", confidence=0.92)
    print(f"  Result: {result} {' PASSED' if result is None else ' FAILED'}\n")

    # Test with real images if available
    test_imgs = glob.glob("data/processed/test/**/*.jpg", recursive=True)[:3]
    if not test_imgs:
        test_imgs = glob.glob("data/**/*.jpg", recursive=True)[:3]

    if test_imgs:
        for img_path in test_imgs:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            disease = os.path.basename(os.path.dirname(img_path))
            print(f"Disease : {disease[:50]}")
            explanation = explain_with_vision(img_bytes, disease, confidence=0.62)
            print(f"Vision  : {explanation}\n")
    else:
        print("No test images found — skipping image tests")
        print("Place test images in data/processed/test/<class_name>/*.jpg")
