"""Experiments history page."""
import httpx
import streamlit as st


def render_experiments() -> None:
    st.markdown('<h2 class="gradient-text">📖 Experiment History</h2>', unsafe_allow_html=True)
    
    try:
        resp = httpx.get("http://localhost:8000/api/v1/experiments", timeout=10.0)
        experiments = resp.json() if resp.status_code == 200 else []
    except Exception:
        experiments = []

    if not experiments:
        st.info("No experiments stored yet. Run the pipeline to build your experiment history.")
        return

    st.markdown(f"**{len(experiments)} experiment(s) in memory**")

    for exp in experiments:
        with st.expander(f"🔬 {exp.get('experiment_id', '?')} — {exp.get('filename', 'unknown')}"):
            cols = st.columns(4)
            with cols[0]: st.metric("Task", exp.get("task_type", "N/A"))
            with cols[1]: st.metric("Best Model", exp.get("best_model", "N/A"))
            with cols[2]: st.metric("Timestamp", exp.get("timestamp", "N/A")[:10] if exp.get("timestamp") else "N/A")

    # Semantic search
    st.markdown("---")
    st.markdown("### 🔎 Semantic Experiment Search")
    query = st.text_input("Search experiments by description", placeholder="classification with imbalanced data, tree-based models...")
    if query and st.button("Search"):
        try:
            resp = httpx.post(
                "http://localhost:8000/api/v1/experiments/similar",
                json={"query": query, "n_results": 5},
                timeout=15.0,
            )
            if resp.status_code == 200:
                results = resp.json()
                for r in results:
                    st.markdown(f"""
                    <div class="autonomds-card" style="margin:0.5rem 0;">
                        <strong>{r.get('experiment_id')}</strong>
                        <p style="color:#94a3b8; font-size:0.85rem; margin:0.3rem 0">{r.get('document','')[:200]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Search failed.")
        except Exception as e:
            st.error(str(e))
