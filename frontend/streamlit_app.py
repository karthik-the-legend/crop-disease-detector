# ================================================================
# frontend\streamlit_app.py
# Crop Disease Detector — Day 21 Version
# Added: gTTS audio + config.toml theme + custom CSS
# ================================================================
import io, json, requests
import streamlit as st
import plotly.express as px
import pandas as pd
from gtts import gTTS
from PIL import Image

import gdown
import os

MODEL_PATH = "models/best_model.pth"
os.makedirs("models", exist_ok=True)

# Download model from Google Drive if not exists or too small (old model)
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000000:
    print("Downloading trained model from Google Drive...")
    url = "https://drive.google.com/uc?id=1uTMU4NsYLAO2vhqqLR6u7rQ2KZ66pgXk"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Model downloaded successfully!")

#  Page config 
st.set_page_config(
    page_title            = " Crop Disease Detector",
    page_icon             = "",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

#  Custom CSS 
st.markdown("""
<style>
    .stProgress > div > div { background: #1a6b52 !important; }
    .streamlit-expanderHeader {
        font-size: 0.8rem !important;
        color: #55ddaa !important;
        background: #1a2820 !important;
        border-radius: 4px !important;
    }
    [data-testid="metric-container"] {
        background: #1a1d24;
        border: 1px solid #333;
        border-radius: 6px;
    }
    footer { visibility: hidden; }
    .main .block-container { padding-top: 1.5rem; }
    .severity-critical {
        background: #dc354522;
        border: 1px solid #dc354555;
        border-radius: 8px;
        padding: 10px 16px;
    }
    .severity-moderate {
        background: #ffc10722;
        border: 1px solid #ffc10755;
        border-radius: 8px;
        padding: 10px 16px;
    }
</style>
""", unsafe_allow_html=True)

API_BASE = "http://localhost:8001"

LANG_MAP = {
    "Telugu" : "te",
    "Hindi"  : "hi",
    "Tamil"  : "ta",
    "Kannada": "kn",
    "English": "en",
}

GTTS_LANG = {"hi": "hi", "te": "te", "ta": "ta", "kn": "kn", "en": "en"}

SEVERITY_STYLES = {
    "critical": ("", "#dc3545", "CRITICAL — Apply treatment TODAY"),
    "moderate": ("", "#ffc107", "MODERATE — Apply within 48 hours"),
    "mild"    : ("", "#28a745", "MILD — Monitor regularly"),
}


#  Helpers 
def diagnose_image(img_bytes: bytes, lang_code: str) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/diagnose?lang_code={lang_code}",
            files   = {"image": ("upload.jpg", img_bytes, "image/jpeg")},
            timeout = 90,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Run: uvicorn backend.main:app --port 8001")
        return {}
    except Exception as e:
        st.error(f"Error: {e}")
        return {}


def check_backend() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def text_to_speech(text: str, lang_code: str) -> bytes:
    """
    Convert treatment text to speech using gTTS.
    Cached so same text+language never regenerates in same session.
    """
    gtts_lang = GTTS_LANG.get(lang_code, "en")
    try:
        tts = gTTS(text=text[:500], lang=gtts_lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return b""   # TTS failure must never break the app


def render_severity_badge(level: str):
    emoji, colour, label = SEVERITY_STYLES.get(level, SEVERITY_STYLES["mild"])
    st.markdown(
        f'<span style="background:{colour}22;color:{colour};'
        f'border:1px solid {colour}55;padding:6px 16px;'
        f'border-radius:100px;font-size:0.85rem;font-weight:700;">'
        f'{emoji} {label}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")


def render_top3_chart(top3: list):
    if not top3:
        return
    names = [
        p.get("disease", "")
         .replace("___", " — ")
         .replace("__", " ")
         .replace("_", " ")[:35]
        for p in top3
    ]
    confs = [round(p.get("confidence", 0) * 100, 1) for p in top3]
    df    = pd.DataFrame({"Disease": names, "Confidence %": confs})
    fig   = px.bar(
        df,
        x                      = "Confidence %",
        y                      = "Disease",
        orientation            = "h",
        color                  = "Confidence %",
        color_continuous_scale = ["#28a745", "#ffc107", "#dc3545"],
        range_x                = [0, 100],
        title                  = "Top-3 CNN Predictions",
        text                   = "Confidence %",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        height              = 200,
        margin              = dict(l=0, r=40, t=40, b=0),
        showlegend          = False,
        coloraxis_showscale = False,
        plot_bgcolor        = "rgba(0,0,0,0)",
        paper_bgcolor       = "rgba(0,0,0,0)",
        font_color          = "#ffffff",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_result(result: dict, lang_name: str, lang_code: str, auto_tts: bool):
    if not result:
        return

    disease    = result.get("disease_name", "Unknown")
    conf       = result.get("confidence", 0)
    severity   = result.get("severity", "mild")
    treatment  = result.get("treatment", "")
    sources    = result.get("sources", [])
    latency    = result.get("latency_ms", 0)
    is_healthy = result.get("is_healthy", False)
    low_conf   = result.get("low_confidence", False)
    top3       = result.get("top3", [])
    groq_exp   = result.get("gemini_explanation")

    display_name = (
        disease
        .replace("___", " — ")
        .replace("__",  " ")
        .replace("_",   " ")
    )

    # Disease header
    st.markdown(f"### {' Plant Appears Healthy' if is_healthy else ' ' + display_name}")

    # Severity badge
    render_severity_badge(severity)

    if low_conf:
        st.warning(" Low confidence — retake photo in better lighting.")

    # Confidence
    st.markdown(f"**CNN Confidence: {conf*100:.1f}%**")
    st.progress(float(conf))
    st.caption(f" {latency}ms | Language: {lang_name}")

    # Top-3 chart
    if low_conf and top3:
        render_top3_chart(top3)
    elif top3:
        with st.expander(" Top 3 Predictions"):
            render_top3_chart(top3)

    st.divider()

    # Groq Vision XAI explanation
    if groq_exp:
        st.markdown("####  Visual AI Explanation")
        st.info(f"**Groq Vision observed:** {groq_exp}")
        st.divider()

    # Treatment text
    st.markdown("###  Treatment Advice")
    st.markdown(treatment)

    # gTTS audio playback
    if auto_tts and treatment:
        with st.spinner(" Generating audio..."):
            audio_bytes = text_to_speech(treatment, lang_code)
        if audio_bytes:
            st.markdown("** Listen to treatment advice:**")
            st.audio(audio_bytes, format="audio/mp3", start_time=0)
        else:
            st.caption(" Audio unavailable — check internet connection")

    # Sources
    if sources:
        with st.expander(f" Sources ({len(sources)} verified passages)"):
            st.caption("Every recommendation is grounded in these verified documents.")
            for s in sources:
                st.markdown(f"** {s['source']} — Page {s['page']+1}**")
                st.caption(str(s.get("content", ""))[:200] + "...")

    st.divider()


#  Sidebar 
with st.sidebar:
    st.markdown("##  Crop Disease Detector")
    st.markdown("---")

    selected_lang = st.selectbox(" Select Your Language", list(LANG_MAP.keys()), index=0)
    lang_code     = LANG_MAP[selected_lang]

    st.markdown("---")
    st.markdown("###  Knowledge Base")
    st.markdown(" ICAR Crop Disease Guides")
    st.markdown(" FAO Pesticide Bulletins")
    st.markdown(" 38 Disease Classes")
    st.markdown(" 54,306 Training Images")

    st.markdown("---")
    auto_tts = st.checkbox(" Auto-play treatment audio", value=True)

    st.markdown("---")
    if check_backend():
        st.success(" Backend: Online")
    else:
        st.error(" Backend: Offline\nRun: uvicorn backend.main:app --port 8001")

    st.markdown("---")
    st.warning(
        " Agricultural advice only. "
        "Always consult your local KVK or agriculture officer "
        "before applying any pesticide."
    )


#  Main 
st.title(" Crop Disease Detector")
st.caption(
    f"CNN ResNet-50 + Groq Vision + RAG Treatment  "
    f"ICAR / FAO Sources  Language: {selected_lang}"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(" Diseases",        "38 classes")
c2.metric(" Training Images", "54,306")
c3.metric(" Languages",       "4 Indian")
c4.metric(" PDF Sources",     "10 ICAR/FAO")

st.divider()

#  Section 1: Image upload 
st.markdown("###  Upload Crop Leaf Photo")
uploaded = st.file_uploader(
    "Upload a crop leaf photo",
    type = ["jpg", "jpeg", "png"],
    help = "Upload a clear photo of a single crop leaf",
)

if uploaded:
    col_img, col_result = st.columns([1, 2])
    with col_img:
        st.image(uploaded, caption="Uploaded Leaf", use_container_width=True)
        st.caption(f"File: {uploaded.name} | Size: {len(uploaded.getvalue())//1024}KB")
    with col_result:
        with st.spinner(" Analysing — this may take 5-10 seconds..."):
            result = diagnose_image(uploaded.getvalue(), lang_code)
        render_result(result, selected_lang, lang_code, auto_tts)
else:
    st.info(
        " Upload a crop leaf photo above to get started.\n\n"
        "The AI will detect the disease, assess severity, and provide "
        "treatment advice from ICAR/FAO sources in your language."
    )

#  Section 2: Voice + Image 
st.markdown("---")
st.markdown("###  Or Upload Image + Voice Together")
st.caption("Whisper auto-detects your language — no need to select manually")

col_v1, col_v2 = st.columns(2)
with col_v1:
    st.markdown("** Crop leaf photo**")
    voice_image = st.file_uploader(
        "Crop leaf photo",
        type             = ["jpg", "jpeg", "png"],
        key              = "voice_image",
        label_visibility = "collapsed",
    )
with col_v2:
    st.markdown("** Voice note (any Indian language)**")
    voice_audio = st.file_uploader(
        "Voice note",
        type             = ["mp3", "wav", "ogg", "m4a"],
        key              = "voice_audio",
        label_visibility = "collapsed",
    )

if voice_image and voice_audio:
    st.audio(voice_audio, format=voice_audio.type)
    if st.button(" Diagnose with Voice", type="primary"):
        with st.spinner(" Whisper transcribing + analysing..."):
            try:
                r = requests.post(
                    f"{API_BASE}/diagnose-voice",
                    files   = {
                        "image": (voice_image.name, voice_image.getvalue(), "image/jpeg"),
                        "audio": (voice_audio.name, voice_audio.getvalue(), voice_audio.type),
                    },
                    timeout = 120,
                )
                r.raise_for_status()
                vr         = r.json()
                transcript = vr.get("transcript", "")
                v_lang     = vr.get("transcript_language", "unknown")
                v_code     = vr.get("lang_code", "en")

                st.success(f" Whisper heard ({v_lang}): **{transcript}**")

                col_vi, col_vr = st.columns([1, 2])
                with col_vi:
                    st.image(voice_image, caption="Uploaded Leaf", use_container_width=True)
                with col_vr:
                    render_result(vr, v_lang, v_code, auto_tts)

            except Exception as e:
                st.error(f"Voice diagnosis failed: {e}")

elif voice_image and not voice_audio:
    st.caption(" Add a voice note to use voice diagnosis")
elif voice_audio and not voice_image:
    st.caption(" Add a leaf photo to use voice diagnosis")
