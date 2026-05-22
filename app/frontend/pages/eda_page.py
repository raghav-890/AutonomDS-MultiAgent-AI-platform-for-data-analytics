"""
EDA Explorer Page
==================
Rich interactive EDA dashboard for the Streamlit frontend.

Displays:
- Dataset overview metrics
- Missing values chart
- Correlation heatmap
- Class balance chart
- Feature distributions
- LLM-generated insights
- Data quality warnings
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path


def render_eda() -> None:
    st.markdown('<h2 class="gradient-text">🔍 EDA Explorer</h2>', unsafe_allow_html=True)

    exp_id = st.session_state.get("experiment_id")
    if not exp_id:
        _render_empty_state()
        return

    # ── Try to load EDA results from API ────────────────────────────────────
    eda_results: dict = {}
    eda_charts:  list[str] = []
    eda_insights: list[str] = []
    eda_warnings: list[str] = []
    dataset_info: dict = {}

    try:
        import httpx
        resp = httpx.get(
            f"http://localhost:8000/api/v1/pipeline/result/{exp_id}",
            timeout=5.0,
        )
        if resp.status_code == 200:
            result = resp.json()
            eda_results  = result.get("eda_results", {})
            eda_charts   = result.get("eda_charts", [])
            eda_insights = result.get("eda_insights", [])
            eda_warnings = result.get("eda_warnings", [])
            dataset_info = result.get("dataset_info", {})
    except Exception:
        pass

    if not eda_results and not eda_charts:
        _render_no_results(exp_id)
        return

    # ── Dataset Overview ─────────────────────────────────────────────────────
    _render_dataset_overview(dataset_info, eda_results)

    # ── Data Quality Warnings ────────────────────────────────────────────────
    if eda_warnings:
        _render_warnings(eda_warnings)

    # ── LLM Insights ────────────────────────────────────────────────────────
    if eda_insights:
        _render_insights(eda_insights)

    st.markdown("---")

    # ── Interactive Charts ───────────────────────────────────────────────────
    if eda_charts:
        _render_charts(eda_charts)
    else:
        st.info("Run the pipeline to generate interactive EDA charts.", icon="📊")

    # ── Raw Statistics ───────────────────────────────────────────────────────
    _render_statistics(eda_results)


# ── Sub-renderers ─────────────────────────────────────────────────────────────

def _render_empty_state() -> None:
    """Show when no experiment is loaded."""
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
        <h3 style="color: #64748b;">No Dataset Loaded</h3>
        <p style="color: #475569;">Upload a dataset first, then run the pipeline to see EDA results here.</p>
    </div>
    """, unsafe_allow_html=True)


def _render_no_results(exp_id: str) -> None:
    """Show when experiment exists but pipeline hasn't run."""
    st.info(
        f"Experiment `{exp_id}` found. Run the **Pipeline Control** to generate EDA analysis.",
        icon="⚙️",
    )
    st.markdown("### 🎬 What EDA Will Show")
    preview_items = [
        ("📉", "Missing Values Heatmap", "Column-level analysis of null patterns and percentages"),
        ("🔗", "Correlation Matrix", "Pairwise Pearson correlations across all numeric features"),
        ("⚖️", "Class Balance Chart", "Target distribution and imbalance ratio"),
        ("📈", "Feature Distributions", "Histograms for all numeric columns"),
        ("🔍", "Outlier Analysis", "IQR-based outlier detection per feature"),
        ("🤖", "LLM Insights", "Natural language EDA commentary from the AI agent"),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(preview_items):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="autonomds-card" style="margin-bottom:1rem; padding: 1.2rem;">
                <div style="font-size:1.6rem; margin-bottom:0.4rem;">{icon}</div>
                <div style="font-weight:700; font-size:0.9rem; margin-bottom:0.3rem;">{title}</div>
                <div style="color:#64748b; font-size:0.8rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


def _render_dataset_overview(dataset_info: dict, eda_results: dict) -> None:
    """Render top-level dataset metric cards."""
    st.markdown("### 📋 Dataset Overview")

    n_rows   = dataset_info.get("n_rows") or eda_results.get("n_rows", 0)
    n_cols   = dataset_info.get("n_cols") or eda_results.get("n_cols", 0)
    target   = dataset_info.get("target_column", "N/A")
    task     = dataset_info.get("task_type", "N/A")
    mem_mb   = dataset_info.get("memory_mb", 0)
    filename = dataset_info.get("filename", "Unknown")

    missing  = eda_results.get("missing_summary", {})
    missing_pct = missing.get("total_missing_pct", 0)

    num_cols = eda_results.get("numeric_cols", [])
    cat_cols = eda_results.get("categorical_cols", [])

    # Metric cards row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    metrics = [
        (m1, f"{n_rows:,}", "Rows"),
        (m2, str(n_cols), "Columns"),
        (m3, str(len(num_cols)), "Numeric"),
        (m4, str(len(cat_cols)), "Categorical"),
        (m5, f"{missing_pct:.1f}%", "Missing"),
        (m6, f"{mem_mb:.1f} MB", "Memory"),
    ]
    for col, val, label in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Info row
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.markdown(f"""
        <div class="autonomds-card" style="padding:1rem;">
            <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em;">File</div>
            <div style="font-weight:600; font-size:0.9rem; margin-top:0.3rem; color:#a5b4fc;">{filename}</div>
        </div>
        """, unsafe_allow_html=True)
    with info_col2:
        st.markdown(f"""
        <div class="autonomds-card" style="padding:1rem;">
            <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em;">Target Column</div>
            <div style="font-weight:600; font-size:0.9rem; margin-top:0.3rem; color:#a5b4fc;">{target}</div>
        </div>
        """, unsafe_allow_html=True)
    with info_col3:
        st.markdown(f"""
        <div class="autonomds-card" style="padding:1rem;">
            <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em;">Task Type</div>
            <div style="font-weight:600; font-size:0.9rem; margin-top:0.3rem; color:#a5b4fc;">{task.replace("_", " ").title()}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


def _render_warnings(warnings: list[str]) -> None:
    """Render data quality warning banners."""
    st.markdown("### ⚠️ Data Quality Warnings")
    for w in warnings:
        if w.startswith("⚠️"):
            st.warning(w)
        elif w.startswith("ℹ️"):
            st.info(w)
        else:
            st.warning(w)


def _render_insights(insights: list[str]) -> None:
    """Render LLM-generated insights as styled cards."""
    st.markdown("### 🤖 AI-Generated Insights")
    insight_icons = ["💡", "🔍", "📊", "⚡", "🎯", "🧠", "📈"]

    cols = st.columns(2)
    for i, insight in enumerate(insights):
        icon = insight_icons[i % len(insight_icons)]
        with cols[i % 2]:
            st.markdown(f"""
            <div class="autonomds-card" style="margin-bottom:0.8rem; padding: 1rem 1.2rem;">
                <div style="display:flex; align-items:flex-start; gap:0.8rem;">
                    <span style="font-size:1.2rem; flex-shrink:0; margin-top:0.1rem;">{icon}</span>
                    <span style="font-size:0.87rem; line-height:1.6; color:#cbd5e1;">{insight}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


def _render_charts(chart_paths: list[str]) -> None:
    """Render interactive Plotly charts from saved HTML files."""
    st.markdown("### 📈 Interactive Analysis Charts")

    chart_tabs_map = {
        "missing_values.html":     "📉 Missing Values",
        "correlation_matrix.html": "🔗 Correlations",
        "class_balance.html":      "⚖️ Class Balance",
        "distributions.html":      "📊 Distributions",
        "leaderboard.html":        "🏆 Model Leaderboard",
        "confusion_matrix.html":   "🧩 Confusion Matrix",
    }

    # Organise chart paths by known names
    named_charts: list[tuple[str, str]] = []
    for path in chart_paths:
        name = Path(path).name
        label = chart_tabs_map.get(name, f"📄 {name.replace('.html', '').replace('_', ' ').title()}")
        named_charts.append((label, path))

    if not named_charts:
        st.info("No charts found for this experiment.")
        return

    # Display in tabs
    tab_labels = [label for label, _ in named_charts]
    tabs = st.tabs(tab_labels)
    for tab, (label, path) in zip(tabs, named_charts):
        with tab:
            try:
                html_content = Path(path).read_text(encoding="utf-8")
                st.components.v1.html(html_content, height=500, scrolling=True)
            except FileNotFoundError:
                st.warning(f"Chart file not found: `{path}`")
            except Exception as e:
                st.error(f"Failed to load chart: {e}")


def _render_statistics(eda_results: dict) -> None:
    """Render raw EDA statistics in expandable sections."""
    if not eda_results:
        return

    st.markdown("---")
    st.markdown("### 🗂️ Detailed Statistics")

    # Missing values table
    missing = eda_results.get("missing_summary", {}).get("by_column", {})
    if missing:
        with st.expander(f"📉 Missing Values Detail ({len(missing)} columns)", expanded=False):
            import pandas as pd
            rows = [
                {"Column": col, "Missing Count": v["count"], "Missing %": f"{v['pct']:.2f}%"}
                for col, v in missing.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Outlier table
    outliers = eda_results.get("outlier_summary", {}).get("by_column", {})
    if outliers:
        with st.expander(f"🔍 Outlier Detail ({len(outliers)} columns)", expanded=False):
            import pandas as pd
            rows = [
                {
                    "Column": col,
                    "Outlier Count": v["count"],
                    "Outlier %": f"{v['pct']:.2f}%",
                    "Lower Fence": v.get("lower_fence", "N/A"),
                    "Upper Fence": v.get("upper_fence", "N/A"),
                }
                for col, v in outliers.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Skewness table
    skewness = eda_results.get("skewness", {})
    if skewness:
        with st.expander(f"📐 Skewness Analysis ({len(skewness)} skewed features)", expanded=False):
            import pandas as pd
            rows = [
                {
                    "Column": col,
                    "Skewness": round(val, 4),
                    "Level": "Extreme" if abs(val) > 5 else "High" if abs(val) > 2 else "Moderate",
                    "Action": "Log transform" if abs(val) > 2 else "Consider transform",
                }
                for col, val in sorted(skewness.items(), key=lambda x: abs(x[1]), reverse=True)
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Class balance
    cb = eda_results.get("class_balance", {})
    dist = cb.get("distribution", {})
    if dist:
        with st.expander(f"⚖️ Class Distribution ({cb.get('n_classes', 0)} classes)", expanded=False):
            import pandas as pd
            rows = [
                {"Class": cls, "Count": v["count"], "Percentage": f"{v['pct']:.2f}%"}
                for cls, v in dist.items()
            ]
            df_cb = pd.DataFrame(rows)
            st.dataframe(df_cb, use_container_width=True, hide_index=True)
            ratio = cb.get("imbalance_ratio", 1)
            if cb.get("is_imbalanced"):
                st.warning(f"⚠️ Imbalance ratio: **{ratio:.1f}x** — class weighting recommended.")
            else:
                st.success(f"✅ Classes are reasonably balanced (ratio: {ratio:.1f}x).")
