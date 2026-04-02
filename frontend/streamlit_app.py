# ================================================================
# frontend\streamlit_app.py
# Crop Disease Detector — Streamlit Frontend
# Run: streamlit run frontend\streamlit_app.py
# Requires: FastAPI backend at http://localhost:8000
# ================================================================
import io, json, requests
import streamlit as st
from PIL import Image

#  Page config — must be FIRST st command 
st.set_page_config(
    page_title         = " Crop Disease Detector",
    page_icon          = "",
    layout             = "wide",
    initial_sidebar_state = "expanded",
)

API_BASE = "http://localhost:8000"

LANG_MAP = {
    "Telugu" : "te",
    "Hindi"  : "hi",
    "Tamil"  : "ta",
    "Kannada": "kn",
    "English": "en",
}

SEVERITY_STYLES = {
    "critical": ("", "#dc3545", "CRITICAL — Apply treatment TODAY"),
    "moderate": ("", "#ffc107", "MODERATE — Apply within 48 hours"),
    "mild"    : ("", "#28a745", "MILD — Monitor regularly"),
}


#  Helpers 
def diagnose_image(img_bytes: bytes, lang_code: str) -> dict:
    """Call POST /diagnose and return parsed JSON."""
    try:
        r = requests.post(
            f"{API_BASE}/diagnose?lang_code={lang_code}",
            files   = {"image": ("upload.jpg", img_bytes, "image/jpeg")},
            timeout = 90,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI backend. Make sure it is running at localhost:8000")
        return {}
    except Exception as e:
        st.error(f"Error: {e}")
        return {}


def check_backend() -> bool:
    """Check if FastAPI backend is reachable."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def render_severity_badge(level: str):
    """Render a coloured severity badge using HTML."""
    emoji, colour, label = SEVERITY_STYLES.get(level, SEVERITY_STYLES["mild"])
    st.markdown(
        f'<span style="background:{colour}22;color:{colour};'
        f'border:1px solid {colour}55;padding:6px 16px;'
        f'border-radius:100px;font-size:0.85rem;font-weight:700;">'
        f'{emoji} {label}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")   # spacing


#  Sidebar 
with st.sidebar:
    st.markdown("##  Crop Disease Detector")
    st.markdown("---")

    selected_lang = st.selectbox(
        " Select Your Language",
        list(LANG_MAP.keys()),
        index=0,
    )
    lang_code = LANG_MAP[selected_lang]

    st.markdown("---")
    st.markdown("###  Knowledge Base")
    st.markdown(" ICAR Crop Disease Guides")
    st.markdown(" FAO Pesticide Bulletins")
    st.markdown(" 38 Disease Classes")
    st.markdown(" 54,306 Training Images")

    st.markdown("---")

    # Backend status indicator
    backend_ok = check_backend()
    if backend_ok:
        st.success(" Backend: Online")
    else:
        st.error(" Backend: Offline\nRun: uvicorn backend.main:app --port 8000")

    st.markdown("---")
    st.warning(
        " Agricultural advice only. "
        "Always consult your local KVK or agriculture officer "
        "before applying any pesticide."
    )


#  Main area 
st.title(" Crop Disease Detector")
st.caption(
    f"CNN ResNet-50 + Groq Vision + RAG Treatment  "
    f"ICAR / FAO Sources  Language: {selected_lang}"
)

# Metric row
c1, c2, c3, c4 = st.columns(4)
c1.metric(" Diseases", "38 classes")
c2.metric(" Training Images", "54,306")
c3.metric(" Languages", "4 Indian")
c4.metric(" PDF Sources", "10 ICAR/FAO")

st.divider()

#  Image upload 
uploaded = st.file_uploader(
    " Upload a crop leaf photo",
    type    = ["jpg", "jpeg", "png"],
    help    = "Upload a clear photo of a single crop leaf for disease detection",
)

if uploaded:
    col_img, col_result = st.columns([1, 2])

    with col_img:
        st.image(uploaded, caption="Uploaded Leaf", use_container_width=True)
        st.caption(f"File: {uploaded.name} | Size: {len(uploaded.getvalue())//1024}KB")

    with col_result:
        with st.spinner(" Analysing disease — this may take 5-10 seconds..."):
            result = diagnose_image(uploaded.getvalue(), lang_code)

        if result:
            disease    = result.get("disease_name", "Unknown")
            conf       = result.get("confidence", 0)
            severity   = result.get("severity", "mild")
            treatment  = result.get("treatment", "")
            sources    = result.get("sources", [])
            latency    = result.get("latency_ms", 0)
            is_healthy = result.get("is_healthy", False)
            low_conf   = result.get("low_confidence", False)
            top3       = result.get("top3", [])
            gemini_exp = result.get("gemini_explanation")

            # Clean disease name
            display_name = (
                disease
                .replace("___", " — ")
                .replace("__",  " ")
                .replace("_",   " ")
            )

            # Disease name header
            if is_healthy:
                st.markdown("###  Plant Appears Healthy")
            else:
                st.markdown(f"###  {display_name}")

            # Severity badge
            render_severity_badge(severity)

            # Low confidence warning
            if low_conf:
                st.warning(" Low confidence — consider retaking photo in better lighting.")

            # Confidence bar
            st.markdown(f"**CNN Confidence: {conf*100:.1f}%**")
            st.progress(float(conf))
            st.caption(f" Response time: {latency}ms | Language: {selected_lang}")

            # Top-3 predictions
            if top3:
                with st.expander(" Top 3 Predictions"):
                    for p in top3:
                        pname = (
                            p.get("disease", "")
                            .replace("___", " — ")
                            .replace("_", " ")
                        )
                        pconf = p.get("confidence", 0)
                        st.markdown(f"**{pname}** — {pconf*100:.1f}%")
                        st.progress(float(pconf))

            st.divider()

            # Gemini/Groq Vision explanation
            if gemini_exp:
                with st.expander(" Visual AI Explanation (Groq Vision)"):
                    st.info(gemini_exp)

            # Treatment
            st.markdown("###  Treatment Advice")
            st.markdown(treatment)

            # Sources
            if sources:
                with st.expander(f" Sources Used ({len(sources)} verified passages)"):
                    st.caption(
                        "Every recommendation comes from these "
                        "verified agricultural documents."
                    )
                    for s in sources:
                        st.markdown(f"** {s['source']} — Page {s['page']+1}**")
                        st.caption(str(s.get('content', ''))[:200] + "...")

            st.divider()

elif not uploaded:
    # Placeholder when no image uploaded
    st.info(
        " Upload a crop leaf photo above to get started.\n\n"
        "The AI will detect the disease, assess severity, and provide "
        "treatment advice from ICAR/FAO sources in your language."
    )
