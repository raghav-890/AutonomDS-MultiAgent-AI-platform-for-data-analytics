"""Home page — landing dashboard."""
import streamlit as st


def render_home() -> None:
    st.markdown("""
    <div class="slide-in" style="text-align:center; padding: 3rem 0 2rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🤖</div>
        <h1 class="gradient-text" style="font-size: 3rem; margin-bottom: 0.5rem;">AutonomDS</h1>
        <p style="color: #64748b; font-size: 1.15rem; max-width: 600px; margin: 0 auto;">
            Your <strong style="color:#a5b4fc">autonomous AI data scientist team</strong> — 
            upload any dataset and watch 11 specialized agents collaborate to deliver 
            production-grade ML insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    cols = st.columns(3)
    features = [
        ("🔍", "Automated EDA", "Missing values, correlations, outliers, class imbalance — fully automated."),
        ("🧹", "Smart Data Cleaning", "KNN imputation, categorical encoding, leakage detection, RobustScaler."),
        ("⚙️", "Feature Engineering", "RFECV selection, polynomial interactions, PCA reduction."),
        ("🏋️", "Optuna HPO Training", "Cross-validated training with Bayesian hyperparameter search."),
        ("🔬", "SHAP Explainability", "TreeExplainer + LIME for global and local model interpretations."),
        ("📄", "PDF Reports", "Professional ReportLab reports with leaderboard and insights."),
    ]
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="autonomds-card" style="margin-bottom:1rem;">
                <div style="font-size:1.8rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-weight:700; font-size:1rem; margin-bottom:0.4rem;">{title}</div>
                <div style="color:#64748b; font-size:0.85rem; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Architecture diagram
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔄 Pipeline Architecture")
    pipeline_stages = [
        ("📂", "Ingest"), ("🔍", "EDA"), ("🧹", "Clean"), ("⚙️", "Features"),
        ("🤔", "Select"), ("🏋️", "Train"), ("📊", "Evaluate"),
        ("🔬", "Explain"), ("📄", "Report"), ("🧠", "Memory"),
    ]
    cols2 = st.columns(len(pipeline_stages))
    for i, (icon, label) in enumerate(pipeline_stages):
        with cols2[i]:
            st.markdown(f"""
            <div style="text-align:center; padding:0.8rem 0.3rem;">
                <div style="font-size:1.5rem;">{icon}</div>
                <div style="font-size:0.7rem; color:#64748b; margin-top:0.3rem; font-weight:600;">{label}</div>
                {"<div style='text-align:center; color:#6366f1; font-size:1rem;'>→</div>" if i < len(pipeline_stages)-1 else ""}
            </div>
            """, unsafe_allow_html=True)

    st.info("👈 **Get started:** Upload a dataset using the sidebar navigation.", icon="🚀")
