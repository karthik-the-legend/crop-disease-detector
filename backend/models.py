# ================================================================
# backend\models.py
# Pydantic schemas for all FastAPI request/response types
# ================================================================
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class SeverityLevel(str, Enum):
    MILD     = "mild"
    MODERATE = "moderate"
    CRITICAL = "critical"


class Source(BaseModel):
    source  : str   # "01_icar_tomato_diseases.pdf"
    page    : int   # 0-indexed page number
    content : str   # first 200 chars of chunk


class TopPrediction(BaseModel):
    disease    : str
    confidence : float


class DiagnosisResponse(BaseModel):
    disease_name       : str
    confidence         : float
    severity           : str                  # "mild"/"moderate"/"critical"
    top3               : List[TopPrediction]
    is_healthy         : bool
    gemini_explanation : Optional[str] = None # vision model explanation
    treatment          : str                  # RAG-grounded treatment text
    sources            : List[Source]         # which PDFs cited
    latency_ms         : int
    lang_code          : str  = "en"
    low_confidence     : bool = False


class VoiceDiagnosisResponse(DiagnosisResponse):
    transcript          : str  # Whisper transcription of farmer voice
    transcript_language : str  # detected language code


class HealthResponse(BaseModel):
    status     : str
    cnn_loaded : bool
    rag_loaded : bool
    version    : str = "1.0.0"


class ErrorResponse(BaseModel):
    error   : str
    message : str
    detail  : Optional[str] = None
