# ================================================================
# backend\treatment_pipeline.py
# RAG pipeline — FAISS retrieval + direct Groq API (no langchain-groq)
# ================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from groq import Groq

from backend.config import (
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    TOP_K_RETRIEVAL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

SYSTEM_PROMPT = """You are an agricultural expert assistant helping Indian farmers.
You ONLY answer using information from the context provided.
You NEVER recommend treatments not mentioned in the context.

RULES (follow all 4 strictly):
1. Answer ONLY from the provided context. If the context does not contain
   the answer, say "I do not have specific treatment information for this
   disease. Please consult your local ICAR Krishi Vigyan Kendra."
2. Name the SPECIFIC pesticide exactly as written in the context
   (e.g. "Mancozeb 0.25%", not just "a fungicide").
3. Include the dosage (ml per litre, g per litre, or kg per hectare)
   if it is mentioned in the context.
4. End your response with:
   "Consult your local agriculture officer or KVK before applying any pesticide."
"""


class TreatmentPipeline:
    """
    RAG pipeline: FAISS retrieval + Groq LLM for treatment advice.

    Usage:
        pipe   = TreatmentPipeline()
        result = pipe.ask("Tomato___Early_blight", language="Telugu")
        print(result["treatment"])
        print(result["sources"])
    """

    def __init__(self):
        print("[TreatmentPipeline] Initialising...")

        self._embeddings = FastEmbedEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        self._vectorstore = FAISS.load_local(
            str(VECTORSTORE_DIR),
            self._embeddings,
            allow_dangerous_deserialization=True,
        )
        self._groq = Groq(api_key=GROQ_API_KEY)
        print("[TreatmentPipeline] Ready.")

    def _retrieve(self, query: str) -> tuple[str, list[dict]]:
        """Run FAISS similarity search, return context string + source list."""
        docs    = self._vectorstore.similarity_search(query, k=TOP_K_RETRIEVAL)
        context = "\n\n".join(d.page_content for d in docs)
        sources = [
            {
                "source" : d.metadata.get("source", "").split("/")[-1],
                "page"   : d.metadata.get("page", 0),
                "content": d.page_content[:200],
            }
            for d in docs
        ]
        return context, sources

    def _call_groq(self, system: str, user: str) -> str:
        """Call Groq LLM and return response text."""
        resp = self._groq.chat.completions.create(
            model       = LLM_MODEL,
            temperature = LLM_TEMPERATURE,
            max_tokens  = LLM_MAX_TOKENS,
            messages    = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content

    def ask(self, disease_name: str, language: str = "English") -> dict:
        """
        Get treatment advice for a diagnosed crop disease.

        Args:
            disease_name : PlantVillage class name e.g. "Tomato___Early_blight"
            language     : English / Hindi / Telugu / Tamil / Kannada

        Returns:
            {"treatment": str, "sources": list}
        """
        clean_name = (
            disease_name
            .replace("___", " ")
            .replace("__",  " ")
            .replace("_",   " ")
        )

        if "healthy" in disease_name.lower():
            treatment_en = (
                f"The {clean_name} appears healthy. No disease detected. "
                "Maintain good farming practices: proper irrigation, "
                "balanced fertilisation, and regular monitoring."
            )
            sources = []

        else:
            query      = (
                f"{clean_name} fungicide pesticide treatment spray dosage "
                f"mancozeb copper chlorothalonil ml litre hectare"
            )
            context, sources = self._retrieve(query)

            user_msg     = f"Context:\n{context}\n\nDisease / Question:\n{query}"
            treatment_en = self._call_groq(SYSTEM_PROMPT, user_msg)

        # Translate if needed
        lang_map  = {
            "Hindi"  : "hi", "Telugu": "te",
            "Tamil"  : "ta", "Kannada": "kn", "English": "en",
        }
        lang_code = lang_map.get(language, "en")
        treatment = self._translate(treatment_en, lang_code) if lang_code != "en" else treatment_en

        return {"treatment": treatment, "sources": sources}

    def _translate(self, text: str, lang_code: str) -> str:
        """Translate using Groq — no external API needed."""
        lang_names = {"hi": "Hindi", "te": "Telugu", "ta": "Tamil", "kn": "Kannada"}
        lang_name  = lang_names.get(lang_code, "Hindi")
        return self._call_groq(
            system = (
                f"Translate the following agricultural advice to {lang_name}. "
                f"Keep pesticide names, dosages and numbers exactly as they are. "
                f"Return only the translated text."
            ),
            user = text,
        )


if __name__ == "__main__":
    pipe = TreatmentPipeline()

    print("\n" + "="*55)
    print("Test 1 — Tomato Early Blight (English):")
    r1 = pipe.ask("Tomato___Early_blight", "English")
    print(r1["treatment"])
    print(f"Sources: {[s['source'] for s in r1['sources']]}")

    print("\n" + "="*55)
    print("Test 2 — Potato Late Blight (Telugu):")
    r2 = pipe.ask("Potato___Late_blight", "Telugu")
    print(r2["treatment"][:300])

    print("\n" + "="*55)
    print("Test 3 — Healthy plant:")
    r3 = pipe.ask("Tomato___healthy")
    print(r3["treatment"])
