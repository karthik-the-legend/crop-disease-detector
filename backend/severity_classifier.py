# ================================================================
# backend\severity_classifier.py
# Rule-based severity — deterministic, never neural
# CRITICAL diseases must never be accidentally downgraded
# ================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from backend.config import (
    DISEASE_SEVERITY_MAP,
    SEVERITY_CRITICAL,
    SEVERITY_MODERATE,
    SEVERITY_MILD,
)


@dataclass
class SeverityResult:
    level  : str   # "critical" / "moderate" / "mild"
    message: str   # display message for the farmer
    colour : str   # hex colour for UI badge
    emoji  : str   # "" / "" / ""
    action : str   # short action instruction


SEVERITY_MESSAGES = {
    SEVERITY_CRITICAL: {
        "message": "CRITICAL — Immediate action required. This disease spreads rapidly and can devastate the entire crop.",
        "colour" : "#b81c2c",
        "emoji"  : "",
        "action" : "Apply treatment today. Isolate affected plants.",
    },
    SEVERITY_MODERATE: {
        "message": "MODERATE — Action recommended within 48 hours. Manageable with prompt treatment.",
        "colour" : "#c28f00",
        "emoji"  : "",
        "action" : "Apply treatment within 2 days.",
    },
    SEVERITY_MILD: {
        "message": "MILD — Low immediate risk. Monitor regularly and apply treatment if spreading.",
        "colour" : "#2a6b42",
        "emoji"  : "",
        "action" : "Monitor weekly. Apply treatment if worsening.",
    },
}


def classify_severity(disease_name: str) -> SeverityResult:
    """
    Classify severity for a PlantVillage disease class name.
    Uses DISEASE_SEVERITY_MAP from config.py — deterministic lookup.

    Why rule-based and not neural?
    A neural model might classify Tomato Late Blight as 67% CRITICAL.
    A rule-based system gives it 100%. For crop protection, CRITICAL
    diseases must never be accidentally downgraded to MODERATE.
    Farmers need certainty, not probability distributions.

    Args:
        disease_name: e.g. "Tomato___Late_blight" or "Apple___healthy"

    Returns:
        SeverityResult with level, message, colour, emoji, action
    """
    level = DISEASE_SEVERITY_MAP.get(disease_name, SEVERITY_MODERATE)
    info  = SEVERITY_MESSAGES[level]
    return SeverityResult(
        level   = level,
        message = info["message"],
        colour  = info["colour"],
        emoji   = info["emoji"],
        action  = info["action"],
    )


if __name__ == "__main__":
    test_cases = [
        ("Tomato___Late_blight",           SEVERITY_CRITICAL),
        ("Tomato___Bacterial_spot",         SEVERITY_CRITICAL),
        ("Tomato__Tomato_mosaic_virus",     SEVERITY_CRITICAL),
        ("Tomato___Early_blight",           SEVERITY_MODERATE),
        ("Apple___Apple_scab",              SEVERITY_MODERATE),
        ("Squash___Powdery_mildew",         SEVERITY_MODERATE),
        ("Tomato___Leaf_Mold",              SEVERITY_MILD),
        ("Apple___healthy",                 SEVERITY_MILD),
        ("Tomato___healthy",                SEVERITY_MILD),
        ("Potato___healthy",                SEVERITY_MILD),
    ]

    passed = 0
    print("=" * 60)
    print("Severity Classifier — 10 Test Cases")
    print("=" * 60)

    for disease, expected in test_cases:
        result = classify_severity(disease)
        ok     = result.level == expected
        if ok:
            passed += 1
        status = "" if ok else ""
        print(f"  {status} {disease[:45]:<45}  {result.emoji} {result.level}")

    print(f"\n{passed}/{len(test_cases)} tests passed")
    if passed == len(test_cases):
        print("ALL PASSED — proceed to Day 12")
    else:
        print("FAILURES — check DISEASE_SEVERITY_MAP in config.py")
