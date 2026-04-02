# ================================================================
# backend\voice_handler.py
# Whisper singleton for farmer voice transcription
# Supports Hindi, Telugu, Tamil, Kannada, English
# ================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import time
from typing import Optional

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import whisper

from backend.config import WHISPER_MODEL_SIZE

#  Singleton 
_whisper_model = None

LANG_CODE_MAP = {
    "hindi"   : "hi",
    "telugu"  : "te",
    "tamil"   : "ta",
    "kannada" : "kn",
    "english" : "en",
    "hi"      : "hi",
    "te"      : "te",
    "ta"      : "ta",
    "kn"      : "kn",
    "en"      : "en",
}


def load_whisper_model() -> whisper.Whisper:
    """
    Load Whisper model as singleton — only loads once.
    Second call returns immediately with cached model.

    Returns:
        Loaded Whisper model
    """
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    print(f"[Whisper] Loading '{WHISPER_MODEL_SIZE}' model...")
    t0             = time.time()
    _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    print(f"[Whisper] Loaded in {round(time.time()-t0, 1)}s")
    return _whisper_model


def transcribe(audio_path: str) -> dict:
    """
    Transcribe an audio file from disk.

    Args:
        audio_path: Path to audio file (mp3/wav/ogg/m4a)

    Returns:
        {
            "text"    : str  — transcribed text,
            "language": str  — full language name e.g. "hindi",
            "lang_code": str — 2-letter code e.g. "hi"
        }
    """
    model  = load_whisper_model()
    result = model.transcribe(audio_path, fp16=False)

    lang_full = result.get("language", "english").lower()
    lang_code = LANG_CODE_MAP.get(lang_full, "en")

    return {
        "text"     : result["text"].strip(),
        "language" : lang_full,
        "lang_code": lang_code,
    }


def transcribe_bytes(audio_bytes: bytes, ext: str = ".mp3") -> dict:
    """
    Transcribe audio from raw bytes (used by FastAPI endpoint).

    Args:
        audio_bytes: Raw audio bytes
        ext        : File extension e.g. ".mp3", ".wav", ".ogg"

    Returns:
        Same dict as transcribe()
    """
    # Write bytes to temp file — Whisper requires a file path
    suffix = ext if ext.startswith(".") else f".{ext}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)   # clean up temp file

    return result


if __name__ == "__main__":
    import sys

    # Test singleton — second load must be instant
    print("Test 1: Singleton pattern")
    t1 = time.time()
    load_whisper_model()
    print(f"  First load: {round(time.time()-t1, 2)}s")

    t2 = time.time()
    load_whisper_model()
    second = round(time.time()-t2, 3)
    print(f"  Second load: {second}s {' PASSED (instant)' if second < 0.1 else ' FAILED'}\n")

    # Test with audio file if provided
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        print(f"Test 2: Transcribe {audio_path}")
        result = transcribe(audio_path)
        print(f"  Text     : {result['text']}")
        print(f"  Language : {result['language']}")
        print(f"  Lang code: {result['lang_code']}")

        print(f"\nTest 3: transcribe_bytes() gives same result")
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        ext    = "." + audio_path.split(".")[-1]
        result2 = transcribe_bytes(audio_bytes, ext)
        match   = result["text"] == result2["text"]
        print(f"  Text match: {' PASSED' if match else ' FAILED'}")
        print(f"  Text: {result2['text']}")
    else:
        print("Test 2: No audio file provided")
        print("  Usage: python -m backend.voice_handler <audio_file.mp3>")
        print("  Skipping transcription tests\n")

    print("All voice_handler tests complete.")
