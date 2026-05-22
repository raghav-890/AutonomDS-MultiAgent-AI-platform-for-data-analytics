"""
AutonomDS — Streamlit Main App
================================
Dark mode, multi-page, startup-grade data science platform UI.
Cloud-ready: reads API URL from st.secrets or STREAMLIT_API_URL env var.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parents[2]))

import streamlit as st

# ── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AutonomDS — Autonomous Data Science Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/your-org/autonomous-ds-agent",
        "Report a bug": "https://github.com/your-org/autonomous-ds-agent/issues",
        "About": "**AutonomDS** — Your autonomous AI data scientist team.",
    },
)

from app.frontend.styles.theme import inject_css
inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size: 2.5rem;">🤖</div>
        <div style="font-size: 1.4rem; font-weight: 800; 
             background: linear-gradient(135deg, #6366f1, #8b5cf6);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            AutonomDS
        </div>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">
            Autonomous AI Data Scientist
        </div>
    </div>
    <hr style="border-color: #1e293b; margin: 0.5rem 0;">
    """, unsafe_allow_html=True)

    st.markdown("### 🧭 Navigation")
    pages = {
        "🏠 Home": "home",
        "📂 Upload Dataset": "upload",
        "🔍 EDA Explorer": "eda",
        "⚙️ Pipeline Control": "pipeline",
        "🏆 Model Leaderboard": "models",
        "📖 Experiment History": "experiments",
        "📄 Reports": "reports",
    }
    selected = st.selectbox("Go to", list(pages.keys()), label_visibility="collapsed")
    page_key = pages[selected]

    st.markdown("<hr style='border-color: #1e293b;'>", unsafe_allow_html=True)

    # ── System status indicators ──────────────────────────────────────────────
    st.markdown("### 🔌 System Status")

    # Resolve API base URL: st.secrets > env var > localhost default
    try:
        _api_url = st.secrets.get("API_BASE_URL", None)
    except Exception:
        _api_url = None
    if not _api_url:
        _api_url = os.environ.get("STREAMLIT_API_URL", "http://localhost:8000")

    # Store globally for all pages
    if "api_base_url" not in st.session_state:
        st.session_state["api_base_url"] = _api_url

    # Ollama status (local only)
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            st.success(f"✅ Ollama ({len(models)} models)")
        else:
            st.info("🤖 Ollama offline → HF fallback")
    except Exception:
        st.info("🤖 Using HuggingFace LLM")

    # API health check against configured URL
    try:
        import httpx
        r = httpx.get(f"{_api_url}/health", timeout=5.0)
        if r.status_code == 200:
            st.success("✅ API Server")
        else:
            st.error("❌ API Server error")
    except Exception:
        st.error(f"❌ API unreachable\n`{_api_url}`")

    st.markdown("<hr style='border-color: #1e293b;'>", unsafe_allow_html=True)
    st.caption("v1.0.0 · MIT License · [GitHub](https://github.com/your-org/autonomous-ds-agent)")

# ── Route to page ─────────────────────────────────────────────────────────────
if page_key == "home":
    from app.frontend.pages.home import render_home
    render_home()
elif page_key == "upload":
    from app.frontend.pages.upload_page import render_upload
    render_upload()
elif page_key == "eda":
    from app.frontend.pages.eda_page import render_eda
    render_eda()
elif page_key == "pipeline":
    from app.frontend.pages.pipeline_page import render_pipeline
    render_pipeline()
elif page_key == "models":
    from app.frontend.pages.models_page import render_models
    render_models()
elif page_key == "experiments":
    from app.frontend.pages.experiments_page import render_experiments
    render_experiments()
elif page_key == "reports":
    from app.frontend.pages.reports_page import render_reports
    render_reports()
