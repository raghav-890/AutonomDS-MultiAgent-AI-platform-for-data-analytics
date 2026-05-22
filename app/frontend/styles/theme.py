"""
AutonomDS Theme — Custom CSS Injection
========================================
Dark mode, glassmorphism, gradient text, animations.
"""

import streamlit as st


def inject_css() -> None:
    """Inject all custom CSS for the dark-mode startup-grade UI."""
    st.markdown("""
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg: #0a0f1e;
        --surface: #0f172a;
        --card: #1e293b;
        --card-hover: #263352;
        --border: #1e2d45;
        --primary: #6366f1;
        --primary-glow: rgba(99, 102, 241, 0.3);
        --secondary: #8b5cf6;
        --accent: #ec4899;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --text: #e2e8f0;
        --text-muted: #64748b;
        --gradient: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
    }

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        color: var(--text) !important;
    }

    .stApp {
        background: var(--bg) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }

    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: var(--card) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
    }

    /* ── Cards / Containers ── */
    .autonomds-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }

    .autonomds-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: var(--gradient);
    }

    .autonomds-card:hover {
        border-color: var(--primary);
        box-shadow: 0 0 24px var(--primary-glow);
        transform: translateY(-2px);
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: all 0.2s ease;
    }

    .metric-card:hover {
        box-shadow: 0 0 20px var(--primary-glow);
        border-color: var(--primary);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: var(--gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 0.78rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }

    /* ── Agent Status Badges ── */
    .agent-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.8rem;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    .badge-running {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid var(--primary);
        color: #a5b4fc;
    }

    .badge-completed {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid var(--success);
        color: #6ee7b7;
    }

    .badge-failed {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid var(--danger);
        color: #fca5a5;
    }

    .badge-pending {
        background: rgba(100, 116, 139, 0.15);
        border: 1px solid #334155;
        color: var(--text-muted);
    }

    /* ── Progress Bar ── */
    .stProgress > div > div {
        background: var(--gradient) !important;
        border-radius: 100px !important;
    }

    .stProgress > div {
        background: var(--card) !important;
        border-radius: 100px !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.25s ease !important;
        letter-spacing: 0.02em !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px var(--primary-glow) !important;
    }

    /* ── File Uploader ── */
    [data-testid="stFileUploader"] {
        background: var(--card) !important;
        border: 2px dashed var(--border) !important;
        border-radius: 12px !important;
        transition: border-color 0.2s !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary) !important;
    }

    /* ── DataFrames ── */
    [data-testid="stDataFrame"] {
        background: var(--card) !important;
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
    }

    /* ── Alerts / Info boxes ── */
    .stAlert {
        border-radius: 10px !important;
        border-left-width: 4px !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface) !important;
        border-radius: 10px !important;
        gap: 0.25rem !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-muted) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--card) !important;
        color: var(--text) !important;
        font-weight: 600 !important;
    }

    /* ── Headers ── */
    h1 { font-weight: 800 !important; letter-spacing: -0.02em !important; }
    h2 { font-weight: 700 !important; color: var(--text) !important; }
    h3 { font-weight: 600 !important; color: var(--text) !important; }

    /* ── Gradient title ── */
    .gradient-text {
        background: var(--gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
    }

    /* ── Chat bubbles ── */
    .chat-bubble {
        padding: 0.8rem 1.2rem;
        border-radius: 16px;
        margin: 0.5rem 0;
        max-width: 85%;
        line-height: 1.6;
        font-size: 0.9rem;
    }

    .chat-bubble-agent {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px 16px 16px 4px;
    }

    .chat-bubble-user {
        background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2));
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 16px 16px 4px 16px;
        margin-left: auto;
    }

    /* ── Leaderboard ── */
    .leaderboard-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.9rem 1.2rem;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        margin: 0.5rem 0;
        transition: all 0.2s;
    }

    .leaderboard-row:hover { border-color: var(--primary); }
    .leaderboard-row.rank-1 { border-color: #f59e0b; box-shadow: 0 0 12px rgba(245,158,11,0.2); }

    /* ── Pulse animation for running agents ── */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }

    /* ── Slide-in animation ── */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .slide-in { animation: slideIn 0.4s ease forwards; }

    /* ── Input fields ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--card) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
        border-radius: 8px !important;
    }

    /* ── Select box ── */
    .stSelectbox > div > div {
        background: var(--card) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
    }

    /* ── Dividers ── */
    hr { border-color: var(--border) !important; }

    /* ── Success / Error status ── */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }

    .status-dot-green { background: var(--success); box-shadow: 0 0 6px var(--success); }
    .status-dot-red { background: var(--danger); box-shadow: 0 0 6px var(--danger); }
    .status-dot-yellow { background: var(--warning); box-shadow: 0 0 6px var(--warning); }
    .status-dot-blue { background: var(--primary); box-shadow: 0 0 6px var(--primary); animation: pulse 2s infinite; }
    </style>
    """, unsafe_allow_html=True)
