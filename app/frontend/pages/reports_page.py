"""Reports download page."""
import httpx
import streamlit as st


def render_reports() -> None:
    st.markdown('<h2 class="gradient-text">📄 Reports</h2>', unsafe_allow_html=True)
    exp_id = st.session_state.get("experiment_id")
    if not exp_id:
        st.warning("Run the pipeline first to generate reports.")
        return

    st.markdown(f"**Experiment:** `{exp_id}`")

    try:
        summary_resp = httpx.get(
            f"http://localhost:8000/api/v1/reports/{exp_id}/summary",
            timeout=10.0,
        )
        if summary_resp.status_code == 200:
            summary = summary_resp.json()
            st.markdown("### 📊 Experiment Summary")
            
            cols = st.columns(3)
            with cols[0]:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{summary.get("best_model","N/A")}</div><div class="metric-label">Best Model</div></div>', unsafe_allow_html=True)
            with cols[1]:
                ds = summary.get("dataset", {})
                st.markdown(f'<div class="metric-card"><div class="metric-value">{ds.get("n_rows",0):,}</div><div class="metric-label">Dataset Rows</div></div>', unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{ds.get("task_type","N/A")}</div><div class="metric-label">Task Type</div></div>', unsafe_allow_html=True)

            # Leaderboard preview
            lb = summary.get("leaderboard", [])
            if lb:
                st.markdown("### 🏆 Top Models")
                import pandas as pd
                df_lb = pd.DataFrame(lb)
                st.dataframe(df_lb, use_container_width=True)
    except Exception:
        st.info("Summary not available yet.")

    # Downloads
    st.markdown("---")
    st.markdown("### ⬇️ Download Reports")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="autonomds-card" style="text-align:center;">
            <div style="font-size:2rem;">📕</div>
            <div style="font-weight:700; margin:0.5rem 0;">PDF Report</div>
            <div style="color:#64748b; font-size:0.85rem;">Full experiment report with charts, leaderboard, and interpretability</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⬇️ Download PDF", use_container_width=True, key="dl_pdf"):
            try:
                resp = httpx.get(f"http://localhost:8000/api/v1/reports/{exp_id}/pdf", timeout=30.0)
                if resp.status_code == 200:
                    st.download_button("💾 Save PDF", resp.content, f"report_{exp_id}.pdf", "application/pdf")
                else:
                    st.error("PDF not ready yet.")
            except Exception as e:
                st.error(str(e))

    with col2:
        st.markdown("""
        <div class="autonomds-card" style="text-align:center;">
            <div style="font-size:2rem;">📝</div>
            <div style="font-weight:700; margin:0.5rem 0;">Markdown Report</div>
            <div style="color:#64748b; font-size:0.85rem;">Full report in Markdown format for GitHub or Notion</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⬇️ Download Markdown", use_container_width=True, key="dl_md"):
            try:
                resp = httpx.get(f"http://localhost:8000/api/v1/reports/{exp_id}/markdown", timeout=30.0)
                if resp.status_code == 200:
                    st.download_button("💾 Save Markdown", resp.content, f"report_{exp_id}.md", "text/markdown")
                else:
                    st.error("Markdown report not ready yet.")
            except Exception as e:
                st.error(str(e))
