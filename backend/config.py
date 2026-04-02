# ================================================================
# backend\config.py
# ALL constants — never hardcode values in other files
# ================================================================
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

load_dotenv()

#  Paths 
BASE_DIR           = Path(__file__).parent.parent
AGRI_PDF_DIR       = BASE_DIR / "data" / "agri_pdfs"
VECTORSTORE_DIR    = BASE_DIR / "vectorstore"
LOGS_DIR           = BASE_DIR / "logs"
MODEL_PATH         = BASE_DIR / "models" / "best_model.pth"
CLASS_NAMES_PATH   = BASE_DIR / "data" / "class_names.json"

VECTORSTORE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

#  API Keys 
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

#  Embedding Model 
# Change this line in config.py
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

#  RAG Chunking 
# 250 words (not 300 like Aarogya AI)
# Agricultural PDFs are denser — pesticide dosages must stay
# together in one chunk, not split across two
CHUNK_SIZE      = 250
CHUNK_OVERLAP   = 40
TOP_K_RETRIEVAL = 3

#  LLM 
LLM_MODEL       = "llama-3.3-70b-versatile"   # Best Groq model for text
VISION_MODEL    = "meta-llama/llama-4-scout-17b-16e-instruct"  # Groq vision model
LLM_TEMPERATURE = 0.1   # near-zero: pesticide advice must be factual
LLM_MAX_TOKENS  = 600

#  CNN Confidence Thresholds 
CNN_CONFIDENCE_HIGH = 0.85  # above this: trust CNN, skip vision check
CNN_CONFIDENCE_LOW  = 0.40  # below this: warn user about low confidence

#  Whisper 
WHISPER_MODEL_SIZE = "base"

#  Languages 
SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "en": "English",
}

#  Severity constants 
SEVERITY_CRITICAL = "critical"
SEVERITY_MODERATE = "moderate"
SEVERITY_MILD     = "mild"

#  Disease Severity Map — all 38 PlantVillage classes 
# CRITICAL : high spread rate, can wipe out entire crop quickly
# MODERATE : manageable with treatment applied within 48 hours
# MILD     : slow spread, low immediate risk, monitor and act
DISEASE_SEVERITY_MAP = {
    "Tomato___Bacterial_spot"                           : SEVERITY_CRITICAL,
    "Tomato___Late_blight"                              : SEVERITY_CRITICAL,
    "Tomato__Tomato_YellowLeaf__Curl_Virus"             : SEVERITY_CRITICAL,
    "Tomato__Tomato_mosaic_virus"                       : SEVERITY_CRITICAL,
    "Potato___Late_blight"                              : SEVERITY_CRITICAL,
    "Orange___Haunglongbing_(Citrus_greening)"          : SEVERITY_CRITICAL,
    "Grape___Black_rot"                                 : SEVERITY_CRITICAL,
    "Corn_(maize)___Northern_Leaf_Blight"               : SEVERITY_CRITICAL,
    "Apple___Fire_blight"                               : SEVERITY_CRITICAL,
    "Tomato___Early_blight"                             : SEVERITY_MODERATE,
    "Tomato___Septoria_leaf_spot"                       : SEVERITY_MODERATE,
    "Tomato___Spider_mites Two-spotted_spider_mite"     : SEVERITY_MODERATE,
    "Tomato__Target_Spot"                               : SEVERITY_MODERATE,
    "Potato___Early_blight"                             : SEVERITY_MODERATE,
    "Apple___Apple_scab"                                : SEVERITY_MODERATE,
    "Apple___Cedar_apple_rust"                          : SEVERITY_MODERATE,
    "Cherry_(including_sour)___Powdery_mildew"          : SEVERITY_MODERATE,
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": SEVERITY_MODERATE,
    "Corn_(maize)___Common_rust_"                       : SEVERITY_MODERATE,
    "Grape___Esca_(Black_Measles)"                      : SEVERITY_MODERATE,
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"        : SEVERITY_MODERATE,
    "Peach___Bacterial_spot"                            : SEVERITY_MODERATE,
    "Pepper,_bell___Bacterial_spot"                     : SEVERITY_MODERATE,
    "Squash___Powdery_mildew"                           : SEVERITY_MODERATE,
    "Strawberry___Leaf_scorch"                          : SEVERITY_MODERATE,
    "Tomato___Leaf_Mold"                                : SEVERITY_MILD,
    "Apple___healthy"                                   : SEVERITY_MILD,
    "Blueberry___healthy"                               : SEVERITY_MILD,
    "Cherry_(including_sour)___healthy"                 : SEVERITY_MILD,
    "Corn_(maize)___healthy"                            : SEVERITY_MILD,
    "Grape___healthy"                                   : SEVERITY_MILD,
    "Peach___healthy"                                   : SEVERITY_MILD,
    "Pepper,_bell___healthy"                            : SEVERITY_MILD,
    "Potato___healthy"                                  : SEVERITY_MILD,
    "Raspberry___healthy"                               : SEVERITY_MILD,
    "Soybean___healthy"                                 : SEVERITY_MILD,
    "Strawberry___healthy"                              : SEVERITY_MILD,
    "Tomato___healthy"                                  : SEVERITY_MILD,
}
