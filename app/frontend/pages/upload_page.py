"""Upload page — drag-and-drop dataset upload with live preview."""
import io
import httpx
import pandas as pd
import streamlit as st


def render_upload() -> None:
    st.markdown('<h2 class="gradient-text">📂 Upload Dataset</h2>', unsafe_allow_html=True)
    st.caption("Supported formats: CSV, Excel (.xlsx/.xls), Parquet, SQLite (.db/.sqlite) · Max 200MB")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader(
            "Drag and drop your dataset here",
            type=["csv", "xlsx", "xls", "parquet", "db", "sqlite"],
            help="Upload a tabular dataset to begin autonomous analysis",
        )

        target_col = st.text_input(
            "Target column (optional)",
            placeholder="e.g. survived, price, churn",
            help="Leave blank to auto-detect",
        )

    with col2:
        st.markdown("""
        <div class="autonomds-card">
            <div style="font-weight:700; margin-bottom:0.8rem;">💡 Tips</div>
            <ul style="color:#94a3b8; font-size:0.85rem; padding-left:1rem; line-height:2;">
                <li>CSV auto-detects delimiter</li>
                <li>Excel uses first sheet</li>
                <li>SQLite uses first table</li>
                <li>Target auto-detected if not specified</li>
                <li>Max 200MB file size</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if uploaded is not None:
        content = uploaded.read()
        
        # Show file info
        mb = len(content) / 1e6
        st.markdown(f"""
        <div class="autonomds-card" style="margin: 1rem 0;">
            <div style="display:flex; gap:2rem; align-items:center;">
                <div><span style="color:#64748b; font-size:0.8rem;">FILE</span><br>
                    <strong>{uploaded.name}</strong></div>
                <div><span style="color:#64748b; font-size:0.8rem;">SIZE</span><br>
                    <strong>{mb:.1f} MB</strong></div>
                <div><span style="color:#64748b; font-size:0.8rem;">TYPE</span><br>
                    <strong>{uploaded.type or 'unknown'}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Preview
        try:
            if uploaded.name.endswith(".csv"):
                df_preview = pd.read_csv(io.BytesIO(content), nrows=100)
            elif uploaded.name.endswith((".xlsx", ".xls")):
                df_preview = pd.read_excel(io.BytesIO(content), nrows=100)
            elif uploaded.name.endswith(".parquet"):
                df_preview = pd.read_parquet(io.BytesIO(content))
                df_preview = df_preview.head(100)
            else:
                df_preview = None

            if df_preview is not None:
                st.markdown("#### 👀 Data Preview")
                st.dataframe(df_preview, use_container_width=True, height=300)

                # Stats
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df_preview.columns)}</div><div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{int(df_preview.isnull().sum().sum())}</div><div class="metric-label">Missing Values</div></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{df_preview.select_dtypes("number").shape[1]}</div><div class="metric-label">Numeric Cols</div></div>', unsafe_allow_html=True)
                with m4:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{df_preview.select_dtypes("object").shape[1]}</div><div class="metric-label">Categorical Cols</div></div>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Preview not available: {e}")

        # Upload button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Upload & Start Analysis", use_container_width=True):
            with st.spinner("Uploading and validating dataset..."):
                try:
                    files = {"file": (uploaded.name, content, uploaded.type or "application/octet-stream")}
                    data = {}
                    if target_col:
                        data["target_column"] = target_col
                    
                    resp = httpx.post(
                        "http://localhost:8000/api/v1/upload",
                        files=files,
                        data=data,
                        timeout=30.0,
                    )
                    if resp.status_code == 201:
                        result = resp.json()
                        st.session_state["experiment_id"] = result["experiment_id"]
                        st.session_state["upload_result"] = result
                        st.success(f"✅ Dataset uploaded! Experiment ID: `{result['experiment_id']}`")
                        st.json(result)
                    else:
                        st.error(f"Upload failed: {resp.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
                    st.info("Make sure the API server is running: `uvicorn app.api.main:app --reload`")
