# ================================================================
# backend\translate.py
# Multilingual translation using Groq LLM
# Pesticide names and dosages are NEVER translated
# ================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from functools import lru_cache
from groq import Groq
from backend.config import GROQ_API_KEY, LLM_MODEL

_client = Groq(api_key=GROQ_API_KEY)

# In-session cache to avoid re-translating identical strings
_translation_cache: dict[str, str] = {}

LANG_NAMES = {
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "en": "English",
}

# Pesticide names that must never be translated
PROTECTED_TERMS = [
    "Mancozeb", "Chlorothalonil", "Copper Oxychloride", "Metalaxyl",
    "Captan", "Carbendazim", "Propiconazole", "Tebuconazole",
    "Fenamidone", "Cymoxanil", "Iprodione", "Thiram",
    "Streptocycline", "Kasugamycin", "Bordeaux Mixture",
]


def _protect_terms(text: str) -> tuple[str, dict]:
    """
    Replace protected terms with placeholders before translation.
    Returns modified text and a map to restore them after.
    """
    placeholders = {}
    protected    = text

    for i, term in enumerate(PROTECTED_TERMS):
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        if pattern.search(protected):
            key              = f"__TERM{i}__"
            placeholders[key] = term
            protected         = pattern.sub(key, protected)

    # Also protect dosage patterns like "0.25%", "3g/litre", "250ml/ha"
    dosage_pattern = re.compile(
        r'\d+\.?\d*\s*(%|g/l|ml/l|kg/ha|ml/ha|g/litre|ml/litre|ppm|lbs|gms?)',
        re.IGNORECASE
    )
    for j, match in enumerate(dosage_pattern.finditer(protected)):
        key              = f"__DOSE{j}__"
        placeholders[key] = match.group()
        protected         = protected.replace(match.group(), key, 1)

    return protected, placeholders


def _restore_terms(text: str, placeholders: dict) -> str:
    """Restore protected terms after translation."""
    for key, original in placeholders.items():
        text = text.replace(key, original)
    return text


@lru_cache(maxsize=512)
def _cached_translate(text: str, lang_code: str) -> str:
    """LRU-cached translation call to Groq."""
    lang_name = LANG_NAMES.get(lang_code, "Hindi")

    protected, placeholders = _protect_terms(text)

    resp = _client.chat.completions.create(
        model      = LLM_MODEL,
        temperature= 0.1,
        max_tokens = 800,
        messages   = [
            {
                "role"   : "system",
                "content": (
                    f"Translate the following agricultural advice text to {lang_name}. "
                    f"Rules: (1) Keep all placeholder tokens like __TERM0__ or __DOSE0__ "
                    f"EXACTLY as they are — do not translate or modify them. "
                    f"(2) Keep numbers exactly as they are. "
                    f"(3) Return only the translated text, nothing else."
                ),
            },
            {"role": "user", "content": protected},
        ],
    )
    translated = resp.choices[0].message.content.strip()
    return _restore_terms(translated, placeholders)


def from_english(text: str, lang_code: str) -> str:
    """
    Translate text from English to target language.
    Pesticide names and dosages are preserved exactly.

    Args:
        text     : English text to translate
        lang_code: "hi"/"te"/"ta"/"kn" (returns original if "en")

    Returns:
        Translated text with pesticide names intact
    """
    if not text or lang_code == "en":
        return text

    cache_key = f"{lang_code}::{text}"
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]

    result                        = _cached_translate(text, lang_code)
    _translation_cache[cache_key] = result
    return result


def to_english(text: str, source_lang: str = "auto") -> str:
    """
    Translate farmer input to English for FAISS search.

    Args:
        text       : Text in any supported language
        source_lang: Language code or "auto" for auto-detect

    Returns:
        English translation
    """
    if not text:
        return text

    cache_key = f"en::{text}"
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]

    lang_hint = f"from {LANG_NAMES.get(source_lang, 'the given language')}" \
                if source_lang != "auto" else "detecting the language automatically"

    resp = _client.chat.completions.create(
        model      = LLM_MODEL,
        temperature= 0.1,
        max_tokens = 400,
        messages   = [
            {
                "role"   : "system",
                "content": (
                    f"Translate the following text to English, {lang_hint}. "
                    f"Keep pesticide names and dosages exactly as written. "
                    f"Return only the translated text."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    result                        = resp.choices[0].message.content.strip()
    _translation_cache[cache_key] = result
    return result


if __name__ == "__main__":
    test_text = "Apply Mancozeb 0.25% spray at 10-day intervals. Use Copper Oxychloride 3g/litre."
    print("Original (English):")
    print(f"  {test_text}\n")

    passed = 0
    for code, lang in [("hi", "Hindi"), ("te", "Telugu"), ("ta", "Tamil"), ("kn", "Kannada")]:
        translated = from_english(test_text, code)
        mancozeb_ok = "Mancozeb" in translated
        copper_ok   = "Copper Oxychloride" in translated or "Copper" in translated
        status      = "PASSED" if mancozeb_ok else "FAILED — Mancozeb was translated!"
        print(f"{lang}: {status}")
        print(f"  {translated[:150]}")
        print()
        if mancozeb_ok:
            passed += 1

    print(f"{passed}/4 languages preserved pesticide names correctly")
