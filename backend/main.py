# ================================================================
# backend\main.py
# FastAPI server — 5 endpoints exposing the multimodal pipeline
# Run: uvicorn backend.main:app --reload --port 8000
# ================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.multimodal_pipeline import MultimodalPipeline
from backend.voice_handler import transcribe_bytes, load_whisper_model
from backend.models import DiagnosisResponse, VoiceDiagnosisResponse, HealthResponse
from backend.config import CLASS_NAMES_PATH, AGRI_PDF_DIR


#  Startup / Shutdown lifecycle 
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Loading MultimodalPipeline...")
    app.state.pipeline = MultimodalPipeline()
    print("[Startup] Pre-loading Whisper model...")
    load_whisper_model()
    print("[Startup] All models loaded. Server ready.")
    yield
    print("[Shutdown] Goodbye.")


app = FastAPI(
    title       = "Crop Disease Detector — Multimodal AI",
    description = (
        "CNN + Groq Vision + RAG treatment retrieval "
        "for 38 crop diseases in 4 Indian languages."
    ),
    version  = "1.0.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


#  POST /diagnose 
@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose_image(
    image    : UploadFile = File(...),
    lang_code: str        = Query("en", description="hi/te/ta/kn/en"),
):
    """Upload a crop leaf photo. Returns disease, severity, treatment in your language."""
    allowed = ["image/jpeg", "image/png", "image/jpg"]
    if image.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {image.content_type}"
        )
    img_bytes = await image.read()
    return app.state.pipeline.diagnose(img_bytes, lang_code=lang_code)


#  POST /diagnose-voice 
@app.post("/diagnose-voice", response_model=VoiceDiagnosisResponse)
async def diagnose_with_voice(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
):
    """
    Upload crop image + voice note.
    Whisper transcribes voice and detects language automatically.
    """
    audio_allowed = [
        "audio/mpeg", "audio/wav", "audio/ogg",
        "audio/mp4", "audio/x-m4a", "audio/mp3",
    ]
    if audio.content_type not in audio_allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type: {audio.content_type}"
        )

    img_bytes   = await image.read()
    audio_bytes = await audio.read()
    ext         = "." + audio.filename.split(".")[-1]

    # Transcribe voice  detect language
    transcript_data = transcribe_bytes(audio_bytes, ext)
    lang_code       = transcript_data["lang_code"]

    # Run diagnosis in detected language
    base_result = app.state.pipeline.diagnose(img_bytes, lang_code=lang_code)

    return VoiceDiagnosisResponse(
        **base_result.model_dump(),
        transcript          = transcript_data["text"],
        transcript_language = transcript_data["language"],
    )


#  GET /diseases 
@app.get("/diseases")
async def list_diseases():
    """List all 38 crop disease classes the CNN can detect."""
    if not CLASS_NAMES_PATH.exists():
        # Return from config if file not found
        from backend.config import DISEASE_SEVERITY_MAP
        names = list(DISEASE_SEVERITY_MAP.keys())
        return {"diseases": names, "count": len(names)}

    with open(CLASS_NAMES_PATH) as f:
        names = json.load(f)
    return {"diseases": names, "count": len(names)}


#  GET /health 
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Frontend calls this on startup to confirm server is ready."""
    return HealthResponse(
        status     = "ok",
        cnn_loaded = True,
        rag_loaded = hasattr(app.state, "pipeline"),
    )


#  GET /sources 
@app.get("/sources")
async def list_sources():
    """List agricultural PDFs in the knowledge base."""
    pdfs = sorted(AGRI_PDF_DIR.glob("*.pdf"))
    return {
        "sources": [p.name for p in pdfs],
        "count"  : len(pdfs),
    }
