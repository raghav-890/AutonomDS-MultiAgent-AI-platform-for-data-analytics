"""Model Leaderboard page."""
import httpx
import streamlit as st


def render_models() -> None:
    st.markdown('<h2 class="gradient-text">🏆 Model Leaderboard</h2>', unsafe_allow_html=True)
    exp_id = st.session_state.get("experiment_id")
    if not exp_id:
        st.warning("Upload and run pipeline first.")
        return

    try:
        resp = httpx.get(f"http://localhost:8000/api/v1/pipeline/result/{exp_id}", timeout=10.0)
        if resp.status_code != 200:
            st.info("Run the pipeline first to see model results.")
            return
        result = resp.json()
    except Exception:
        st.info("API not connected.")
        return

    leaderboard = result.get("leaderboard", [])
    if not leaderboard:
        st.info("No model results yet.")
        return

    # Best model highlight
    best = leaderboard[0]
    best_metrics = best.get("metrics", {})
    primary = list(best_metrics.keys())[0] if best_metrics else "score"
    best_score = best_metrics.get(primary, 0)

    st.markdown(f"""
    <div class="autonomds-card" style="margin-bottom:1.5rem; border-color:#f59e0b;">
        <div style="color:#f59e0b; font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em;">🥇 Best Model</div>
        <div style="font-size:1.8rem; font-weight:800; margin: 0.3rem 0;">{best["model_name"]}</div>
        <div style="color:#64748b;">{primary.upper()}: <strong style="color:#a5b4fc">{best_score:.4f}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    # Full leaderboard
    st.markdown("### All Models")
    for entry in leaderboard:
        metrics = entry.get("metrics", {})
        primary_score = list(metrics.values())[0] if metrics else 0
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(entry["rank"], f"#{entry['rank']}")
        rank_class = "rank-1" if entry["rank"] == 1 else ""

        metrics_str = " · ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        st.markdown(f"""
        <div class="leaderboard-row {rank_class}">
            <div style="font-size:1.2rem; width:2rem;">{medal}</div>
            <div style="flex:1;">
                <div style="font-weight:700;">{entry["model_name"]}</div>
                <div style="color:#64748b; font-size:0.8rem;">{metrics_str}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.1rem; font-weight:700; color:#a5b4fc">{primary_score:.4f}</div>
                <div style="color:#64748b; font-size:0.75rem;">CV: {entry.get('cv_mean', 0):.4f}</div>
            </div>
            <div style="text-align:right; color:#64748b; font-size:0.8rem;">
                {entry.get("training_time_seconds", 0):.1f}s
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Feature importance
    fi = result.get("feature_importance", {})
    if fi:
        st.markdown("### 🔬 Top Feature Importance (SHAP)")
        top_features = list(fi.items())[:10]
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=[v for _, v in reversed(top_features)],
            y=[k for k, _ in reversed(top_features)],
            orientation="h",
            marker_color="#6366f1",
        ))
        fig.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font_color="#e2e8f0", height=350,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
