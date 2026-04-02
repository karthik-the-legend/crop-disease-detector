# ================================================================
# backend\multimodal_pipeline.py
# Orchestrates all 9 stages of the multimodal diagnosis pipeline
# Input : image bytes + language code
# Output: DiagnosisResponse (disease, severity, treatment, sources)
# ================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import Optional

from backend.gemini_vision import explain_with_vision
from backend.severity_classifier import classify_severity
from backend.treatment_pipeline import TreatmentPipeline
from backend.translate import from_english
from backend.models import DiagnosisResponse, TopPrediction, Source
from backend.config import SUPPORTED_LANGUAGES


class MultimodalPipeline:
    """
    Orchestrates the complete multimodal crop disease diagnosis.
    All heavy models are loaded once at startup — not per request.

    Usage:
        pipeline = MultimodalPipeline()
        result   = pipeline.diagnose(image_bytes, lang_code="te")
        print(result.disease_name, result.treatment)
    """

    def __init__(self):
        print("[MultimodalPipeline] Loading TreatmentPipeline (FAISS + Groq)...")
        self._treatment = TreatmentPipeline()
        print("[MultimodalPipeline] Ready.")

    def diagnose(
        self,
        image_bytes: bytes,
        lang_code  : str = "en",
        disease_name: Optional[str] = None,
        confidence  : Optional[float] = None,
        top3        : Optional[list]  = None,
        is_healthy  : Optional[bool]  = None,
        low_conf    : Optional[bool]  = None,
    ) -> DiagnosisResponse:
        """
        Full pipeline for a single crop image.

        Stage 1 : image bytes received
        Stage 2 : CNN result passed in (predict.py handles CNN)
        Stage 3 : Groq Vision confirms + explains (if confidence < 0.85)
        Stage 4 : classify_severity -> MILD / MODERATE / CRITICAL
        Stage 5 : TreatmentPipeline retrieves treatment from FAISS + Groq
        Stage 6 : Translate treatment to target language

        Args:
            image_bytes : Raw image bytes (JPEG/PNG)
            lang_code   : "en"/"hi"/"te"/"ta"/"kn"
            disease_name: CNN predicted class name
            confidence  : CNN top-1 confidence (0.0-1.0)
            top3        : List of top-3 predictions
            is_healthy  : Whether plant is healthy
            low_conf    : Whether confidence is below threshold

        Returns:
            DiagnosisResponse with all fields populated
        """
        t0       = time.time()
        language = SUPPORTED_LANGUAGES.get(lang_code, "English")

        # Defaults for when CNN result is not passed in
        disease_name = disease_name or "Tomato___healthy"
        confidence   = confidence   or 0.99
        top3_objs    = [TopPrediction(**p) for p in (top3 or [])]
        is_healthy   = is_healthy   if is_healthy is not None else True
        low_conf     = low_conf     if low_conf   is not None else False

        #  Stage 3: Groq Vision (only if confidence < 0.85) 
        gemini_explanation = explain_with_vision(
            image_bytes, disease_name, confidence
        )

        #  Stage 4: Severity Classification 
        sev_result = classify_severity(disease_name)

        #  Stage 5: RAG Treatment Retrieval (always in English) 
        rag_result   = self._treatment.ask(disease_name, language="English")
        treatment_en = rag_result["treatment"]
        sources_raw  = rag_result["sources"]

        #  Stage 6: Translate to farmer's language 
        treatment = from_english(treatment_en, lang_code) if lang_code != "en" else treatment_en

        if gemini_explanation and lang_code != "en":
            gemini_explanation = from_english(gemini_explanation, lang_code)

        latency_ms = round((time.time() - t0) * 1000)

        return DiagnosisResponse(
            disease_name       = disease_name,
            confidence         = confidence,
            severity           = sev_result.level,
            top3               = top3_objs,
            is_healthy         = is_healthy,
            gemini_explanation = gemini_explanation,
            treatment          = treatment,
            sources            = [Source(**s) for s in sources_raw],
            latency_ms         = latency_ms,
            lang_code          = lang_code,
            low_confidence     = low_conf,
        )


if __name__ == "__main__":
    import glob

    pipeline  = MultimodalPipeline()
    test_imgs = glob.glob("data/processed/test/**/*.jpg", recursive=True)[:3]

    if not test_imgs:
        test_imgs = glob.glob("data/**/*.jpg", recursive=True)[:3]

    if test_imgs:
        for img_path in test_imgs:
            with open(img_path, "rb") as f:
                img_bytes = f.read()

            disease = os.path.basename(os.path.dirname(img_path))
            result  = pipeline.diagnose(
                image_bytes  = img_bytes,
                lang_code    = "te",
                disease_name = disease,
                confidence   = 0.72,
                is_healthy   = "healthy" in disease.lower(),
            )
            print(f"\nDisease   : {result.disease_name}")
            print(f"Severity  : {result.severity}")
            print(f"Treatment : {result.treatment[:150]}")
            print(f"Sources   : {[s.source for s in result.sources]}")
            print(f"Latency   : {result.latency_ms}ms")
    else:
        # Test without real images
        print("\nNo images found — running with mock data\n")
        result = pipeline.diagnose(
            image_bytes  = b"mock",
            lang_code    = "te",
            disease_name = "Tomato___Early_blight",
            confidence   = 0.72,
            is_healthy   = False,
        )
        print(f"Disease   : {result.disease_name}")
        print(f"Severity  : {result.severity}")
        print(f"Treatment : {result.treatment[:200]}")
        print(f"Latency   : {result.latency_ms}ms")

        result2 = pipeline.diagnose(
            image_bytes  = b"mock",
            lang_code    = "en",
            disease_name = "Tomato___healthy",
            confidence   = 0.98,
            is_healthy   = True,
        )
        print(f"\nHealthy test — is_healthy: {result2.is_healthy}")
        print(f"Treatment : {result2.treatment}")
