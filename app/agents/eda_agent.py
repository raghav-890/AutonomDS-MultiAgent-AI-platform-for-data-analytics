"""
EDA Agent
==========
Performs comprehensive Exploratory Data Analysis on the ingested dataset:

1. Missing values analysis — count, percentage, heatmap
2. Correlation analysis — Pearson matrix, top correlated pairs
3. Outlier detection — IQR-based per numeric column
4. Class balance analysis — target distribution
5. Skewness analysis — highly skewed features flagged
6. Distribution plots — histograms + box plots
7. LLM-generated natural language insights (optional, graceful fallback)

All charts are saved as Plotly HTML files for display in Streamlit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.agents.base_agent import BaseAgent, AgentError
from app.orchestration.state import AgentState, PipelineStage
from app.utils.config import get_settings
from app.utils.helpers import now_iso


# ── Colour palette (matches dark theme) ──────────────────────────────────────
COLORS = {
    "primary":   "#6366f1",
    "secondary": "#8b5cf6",
    "accent":    "#ec4899",
    "success":   "#10b981",
    "warning":   "#f59e0b",
    "danger":    "#ef4444",
    "bg":        "#0a0f1e",
    "surface":   "#0f172a",
    "card":      "#1e293b",
    "text":      "#e2e8f0",
    "muted":     "#64748b",
}

_PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["card"],
    font_color=COLORS["text"],
    margin=dict(l=40, r=40, t=60, b=40),
)

_MAX_DIST_COLS = 12   # Max columns to plot distributions for


class EDAAgent(BaseAgent):
    """
    Exploratory Data Analysis Agent.

    Reads the raw uploaded file, computes statistics, generates
    interactive Plotly charts, and optionally queries the LLM for
    natural-language insights about the dataset.
    """

    name = "eda_agent"
    description = "Performs automated EDA: stats, correlations, outliers, imbalance, LLM insights"
    stage = PipelineStage.EDA
    max_retries = 1

    # ── Main entry point ──────────────────────────────────────────────────────

    def execute(self, state: AgentState) -> AgentState:
        raw_path = state.get("raw_data_path", "")
        if not raw_path or not Path(raw_path).exists():
            raise AgentError(f"Raw data file not found: {raw_path!r}")

        settings = get_settings()
        exp_id   = state.get("experiment_id", "default")
        target   = state.get("target_column", "")

        # ── Load dataset ─────────────────────────────────────────────────────
        df = self._load_dataframe(raw_path)
        self.log_action("data_loaded", rows=len(df), cols=len(df.columns))

        # Separate column types
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Remove target from feature lists (if identified)
        if target and target in num_cols:
            num_cols = [c for c in num_cols if c != target]
        if target and target in cat_cols:
            cat_cols = [c for c in cat_cols if c != target]

        # ── Chart output directory ────────────────────────────────────────────
        chart_dir = Path(settings.reports_dir) / exp_id / "eda_charts"
        chart_dir.mkdir(parents=True, exist_ok=True)
        chart_paths: list[str] = []

        # ── 1. Missing values ────────────────────────────────────────────────
        missing_summary = self._analyze_missing(df)
        if missing_summary["total_missing"] > 0:
            p = self._plot_missing(df, missing_summary, chart_dir)
            if p:
                chart_paths.append(p)

        # ── 2. Correlation matrix ─────────────────────────────────────────────
        corr_matrix: list[list[float]] = []
        if len(num_cols) >= 2:
            corr_matrix, corr_path = self._plot_correlation(df, num_cols, chart_dir)
            if corr_path:
                chart_paths.append(corr_path)

        # ── 3. Outlier detection ─────────────────────────────────────────────
        outlier_summary = self._analyze_outliers(df, num_cols)

        # ── 4. Class balance ─────────────────────────────────────────────────
        class_balance: dict[str, Any] = {}
        if target and target in df.columns:
            class_balance = self._analyze_class_balance(df, target)
            p = self._plot_class_balance(df, target, class_balance, chart_dir)
            if p:
                chart_paths.append(p)

        # ── 5. Skewness analysis ─────────────────────────────────────────────
        skewness = self._analyze_skewness(df, num_cols)

        # ── 6. Distribution plots ────────────────────────────────────────────
        if num_cols:
            p = self._plot_distributions(df, num_cols[:_MAX_DIST_COLS], chart_dir)
            if p:
                chart_paths.append(p)

        # ── 7. LLM insights ──────────────────────────────────────────────────
        warnings: list[str] = self._build_warnings(missing_summary, outlier_summary,
                                                    skewness, class_balance)
        nl_insights = self._generate_insights(df, missing_summary, outlier_summary,
                                              skewness, class_balance, target, warnings)

        # ── Build EDA results dict ────────────────────────────────────────────
        eda_results: dict[str, Any] = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "missing_summary": missing_summary,
            "outlier_summary": outlier_summary,
            "class_balance": class_balance,
            "skewness": skewness,
            "warnings": warnings,
            "completed_at": now_iso(),
        }

        # ── Update state ──────────────────────────────────────────────────────
        state = dict(state)                                     # type: ignore[assignment]
        state["eda_results"]  = eda_results
        state["eda_charts"]   = chart_paths
        state["eda_insights"] = nl_insights
        state["eda_warnings"] = warnings

        # Populate column type lists for downstream agents
        all_num = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
        state["numeric_columns"]     = [c for c in all_num if c != target]
        state["categorical_columns"] = [c for c in all_cat if c != target]
        state["feature_columns"]     = [c for c in df.columns if c != target]

        self.logger.info(
            "eda_complete",
            charts=len(chart_paths),
            warnings=len(warnings),
            insights=len(nl_insights),
        )
        return state                                             # type: ignore[return-value]

    # ── Confidence scoring ────────────────────────────────────────────────────

    def compute_confidence(self, state: AgentState) -> float:
        errors = state.get("errors", [])
        agent_errors = [e for e in errors if self.name in e]
        if agent_errors:
            return 0.3
        # Penalise if many warnings (data quality issues)
        warnings = state.get("eda_warnings", [])
        if len(warnings) >= 5:
            return 0.6
        return 1.0

    def _success_message(self, state: AgentState) -> str:
        n_charts = len(state.get("eda_charts", []))
        n_insights = len(state.get("eda_insights", []))
        return (
            f"✅ EDA complete — {n_charts} charts, {n_insights} insights generated. "
            f"{len(state.get('eda_warnings', []))} data quality warnings."
        )

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_dataframe(self, path: str) -> pd.DataFrame:
        """Load any supported format into a DataFrame."""
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            return pd.read_csv(path, low_memory=False)
        elif ext in (".xls", ".xlsx"):
            return pd.read_excel(path)
        elif ext == ".parquet":
            return pd.read_parquet(path)
        elif ext in (".db", ".sqlite", ".sqlite3"):
            import sqlite3
            conn = sqlite3.connect(path)
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
            table = tables["name"].iloc[0] if not tables.empty else None
            if not table:
                raise AgentError("No tables found in SQLite database.")
            df = pd.read_sql(f"SELECT * FROM {table}", conn)
            conn.close()
            return df
        else:
            raise AgentError(f"Unsupported file format: {ext}")

    # ── Analysis methods ──────────────────────────────────────────────────────

    def _analyze_missing(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute missing-value counts and percentages per column."""
        missing_counts = df.isnull().sum()
        missing_pct    = (missing_counts / len(df) * 100).round(2)
        missing_cols   = {
            col: {"count": int(missing_counts[col]), "pct": float(missing_pct[col])}
            for col in df.columns
            if missing_counts[col] > 0
        }
        return {
            "by_column": missing_cols,
            "total_missing": int(missing_counts.sum()),
            "total_missing_pct": round(float(missing_counts.sum()) / (len(df) * len(df.columns)) * 100, 2),
            "cols_with_missing": len(missing_cols),
            "rows_with_any_missing": int(df.isnull().any(axis=1).sum()),
        }

    def _analyze_outliers(self, df: pd.DataFrame, num_cols: list[str]) -> dict[str, Any]:
        """IQR-based outlier detection for each numeric column."""
        outliers: dict[str, Any] = {}
        for col in num_cols:
            try:
                series = df[col].dropna()
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                mask = (series < lower) | (series > upper)
                n_out = int(mask.sum())
                if n_out > 0:
                    outliers[col] = {
                        "count": n_out,
                        "pct": round(n_out / len(series) * 100, 2),
                        "lower_fence": round(float(lower), 4),
                        "upper_fence": round(float(upper), 4),
                    }
            except Exception:
                continue
        return {
            "by_column": outliers,
            "cols_with_outliers": len(outliers),
            "total_outlier_cells": sum(v["count"] for v in outliers.values()),
        }

    def _analyze_class_balance(self, df: pd.DataFrame, target: str) -> dict[str, Any]:
        """Compute class distribution for classification targets."""
        vc = df[target].value_counts(normalize=False)
        vc_pct = df[target].value_counts(normalize=True) * 100
        dist = {str(k): {"count": int(v), "pct": round(float(vc_pct[k]), 2)}
                for k, v in vc.items()}
        counts = list(vc.values)
        imbalance_ratio = round(max(counts) / min(counts), 2) if min(counts) > 0 else 999
        return {
            "distribution": dist,
            "n_classes": len(dist),
            "imbalance_ratio": imbalance_ratio,
            "is_imbalanced": imbalance_ratio > 3.0,
        }

    def _analyze_skewness(self, df: pd.DataFrame, num_cols: list[str]) -> dict[str, float]:
        """Compute skewness for each numeric feature."""
        result: dict[str, float] = {}
        for col in num_cols:
            try:
                sk = float(df[col].skew())
                if abs(sk) > 0.5:           # Only report meaningful skew
                    result[col] = round(sk, 4)
            except Exception:
                continue
        return result

    def _build_warnings(
        self,
        missing: dict[str, Any],
        outliers: dict[str, Any],
        skewness: dict[str, float],
        class_balance: dict[str, Any],
    ) -> list[str]:
        """Build a list of data-quality warnings."""
        warnings: list[str] = []

        mp = missing.get("total_missing_pct", 0)
        if mp > 30:
            warnings.append(f"⚠️ High missingness: {mp:.1f}% of all values are missing.")
        elif mp > 5:
            warnings.append(f"ℹ️ Missing values detected: {mp:.1f}% of dataset.")

        n_out_cols = outliers.get("cols_with_outliers", 0)
        if n_out_cols > 0:
            warnings.append(f"⚠️ Outliers detected in {n_out_cols} column(s) — consider capping or removal.")

        high_skew = {c: s for c, s in skewness.items() if abs(s) > 2}
        if high_skew:
            cols = ", ".join(list(high_skew.keys())[:3])
            warnings.append(f"⚠️ Highly skewed features: {cols} — consider log/sqrt transform.")

        if class_balance.get("is_imbalanced"):
            ratio = class_balance.get("imbalance_ratio", 1)
            warnings.append(f"⚠️ Class imbalance detected (ratio {ratio:.1f}x) — consider SMOTE or class weights.")

        return warnings

    # ── Chart methods ─────────────────────────────────────────────────────────

    def _plot_missing(self, df: pd.DataFrame, summary: dict, chart_dir: Path) -> str | None:
        """Generate an interactive bar chart of missing value percentages."""
        try:
            by_col = summary["by_column"]
            if not by_col:
                return None
            cols   = list(by_col.keys())
            pcts   = [by_col[c]["pct"] for c in cols]
            counts = [by_col[c]["count"] for c in cols]

            fig = go.Figure(go.Bar(
                x=cols, y=pcts,
                text=[f"{p:.1f}% ({c})" for p, c in zip(pcts, counts)],
                textposition="outside",
                marker_color=[COLORS["danger"] if p > 30 else COLORS["warning"] if p > 10
                              else COLORS["primary"] for p in pcts],
            ))
            fig.update_layout(
                **_PLOTLY_LAYOUT,
                title="Missing Values by Column (%)",
                xaxis_title="Column",
                yaxis_title="Missing (%)",
                xaxis=dict(tickangle=-45),
                height=400,
            )
            path = str(chart_dir / "missing_values.html")
            fig.write_html(path)
            return path
        except Exception as e:
            self.logger.warning("missing_plot_failed", error=str(e))
            return None

    def _plot_correlation(
        self, df: pd.DataFrame, num_cols: list[str], chart_dir: Path
    ) -> tuple[list[list[float]], str | None]:
        """Generate a correlation heatmap. Returns (matrix, path)."""
        try:
            corr = df[num_cols].corr().round(3)
            matrix = corr.values.tolist()

            fig = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="RdBu",
                zmid=0,
                text=corr.values.round(2),
                texttemplate="%{text}",
                colorbar=dict(title="r"),
            ))
            fig.update_layout(
                **_PLOTLY_LAYOUT,
                title="Feature Correlation Matrix",
                height=max(400, len(num_cols) * 35 + 100),
            )
            path = str(chart_dir / "correlation_matrix.html")
            fig.write_html(path)
            return matrix, path
        except Exception as e:
            self.logger.warning("correlation_plot_failed", error=str(e))
            return [], None

    def _plot_class_balance(
        self,
        df: pd.DataFrame,
        target: str,
        summary: dict[str, Any],
        chart_dir: Path,
    ) -> str | None:
        """Generate a class balance bar/pie chart."""
        try:
            dist   = summary.get("distribution", {})
            labels = list(dist.keys())
            counts = [dist[k]["count"] for k in labels]

            color_seq = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                         COLORS["success"], COLORS["warning"], COLORS["danger"]]
            fig = make_subplots(rows=1, cols=2,
                                specs=[[{"type": "bar"}, {"type": "pie"}]])

            fig.add_trace(go.Bar(
                x=labels, y=counts,
                marker_color=color_seq[:len(labels)],
                text=counts, textposition="outside",
                name="Count",
            ), row=1, col=1)

            fig.add_trace(go.Pie(
                labels=labels, values=counts,
                hole=0.45,
                marker_colors=color_seq[:len(labels)],
            ), row=1, col=2)

            fig.update_layout(
                **_PLOTLY_LAYOUT,
                title=f"Class Balance — Target: `{target}`",
                showlegend=True,
                height=380,
            )
            path = str(chart_dir / "class_balance.html")
            fig.write_html(path)
            return path
        except Exception as e:
            self.logger.warning("class_balance_plot_failed", error=str(e))
            return None

    def _plot_distributions(
        self, df: pd.DataFrame, num_cols: list[str], chart_dir: Path
    ) -> str | None:
        """Generate a grid of distribution histograms + box plots."""
        try:
            n = len(num_cols)
            cols_per_row = min(4, n)
            rows = (n + cols_per_row - 1) // cols_per_row

            fig = make_subplots(
                rows=rows, cols=cols_per_row,
                subplot_titles=num_cols,
                shared_yaxes=False,
            )
            for idx, col in enumerate(num_cols):
                r, c = divmod(idx, cols_per_row)
                series = df[col].dropna()
                color = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]][idx % 3]
                fig.add_trace(
                    go.Histogram(
                        x=series,
                        name=col,
                        marker_color=color,
                        opacity=0.75,
                        showlegend=False,
                        nbinsx=30,
                    ),
                    row=r + 1, col=c + 1,
                )

            fig.update_layout(
                **_PLOTLY_LAYOUT,
                title="Feature Distributions",
                height=max(300, rows * 220),
            )
            path = str(chart_dir / "distributions.html")
            fig.write_html(path)
            return path
        except Exception as e:
            self.logger.warning("distributions_plot_failed", error=str(e))
            return None

    # ── LLM Insights ─────────────────────────────────────────────────────────

    def _generate_insights(
        self,
        df: pd.DataFrame,
        missing: dict[str, Any],
        outliers: dict[str, Any],
        skewness: dict[str, float],
        class_balance: dict[str, Any],
        target: str,
        warnings: list[str],
    ) -> list[str]:
        """
        Generate natural-language EDA insights via LLM.
        Falls back to rule-based insights if LLM is unavailable.
        """
        # Build statistical summary for the prompt
        stat_summary = (
            f"Dataset: {len(df)} rows, {len(df.columns)} columns.\n"
            f"Target column: {target or 'unknown'}.\n"
            f"Missing values: {missing.get('total_missing_pct', 0):.1f}% total.\n"
            f"Columns with outliers: {outliers.get('cols_with_outliers', 0)}.\n"
            f"Highly skewed columns: {list(skewness.keys())[:5]}.\n"
            f"Class balance: {class_balance.get('distribution', {})}.\n"
            f"Data quality warnings: {warnings}.\n"
        )

        system_prompt = (
            "You are a senior data scientist performing exploratory data analysis. "
            "Given the following dataset statistics, generate 5 concise, actionable insights "
            "about this dataset. Each insight should be one sentence. Focus on:\n"
            "- Data quality issues\n"
            "- Feature characteristics\n"
            "- Recommendations for preprocessing\n"
            "- Potential modelling challenges\n"
            "- Class imbalance or target distribution\n"
            "Output ONLY a JSON array of 5 insight strings. No other text."
        )

        try:
            response = self.ask_llm(system_prompt, stat_summary)
            # Extract JSON array from response
            import re
            match = re.search(r"\[.*?\]", response, re.DOTALL)
            if match:
                insights = json.loads(match.group())
                if isinstance(insights, list) and all(isinstance(i, str) for i in insights):
                    return insights[:7]
        except Exception as e:
            self.logger.warning("llm_insights_failed", error=str(e))

        # ── Rule-based fallback ────────────────────────────────────────────
        return self._rule_based_insights(df, missing, outliers, skewness, class_balance, target)

    def _rule_based_insights(
        self,
        df: pd.DataFrame,
        missing: dict[str, Any],
        outliers: dict[str, Any],
        skewness: dict[str, float],
        class_balance: dict[str, Any],
        target: str,
    ) -> list[str]:
        """Generate rule-based EDA insights without LLM."""
        insights: list[str] = []
        n_rows, n_cols = len(df), len(df.columns)

        insights.append(
            f"Dataset contains {n_rows:,} rows and {n_cols} columns "
            f"({'large' if n_rows > 50_000 else 'medium' if n_rows > 5_000 else 'small'} dataset)."
        )

        mp = missing.get("total_missing_pct", 0)
        if mp > 0:
            insights.append(
                f"{mp:.1f}% of values are missing across {missing.get('cols_with_missing', 0)} column(s) "
                "— imputation strategy recommended before training."
            )

        n_out = outliers.get("cols_with_outliers", 0)
        if n_out > 0:
            insights.append(
                f"Outliers detected in {n_out} numeric column(s) via IQR analysis "
                "— consider RobustScaler or winsorization."
            )

        high_skew = {c: s for c, s in skewness.items() if abs(s) > 2}
        if high_skew:
            insights.append(
                f"{len(high_skew)} feature(s) show high skewness (|skew| > 2) "
                "— log or Box-Cox transformation recommended."
            )

        if class_balance.get("is_imbalanced"):
            ratio = class_balance.get("imbalance_ratio", 1)
            insights.append(
                f"Target class imbalance ratio is {ratio:.1f}x "
                "— SMOTE oversampling or class_weight='balanced' advised."
            )
        elif class_balance.get("n_classes"):
            n_cls = class_balance["n_classes"]
            insights.append(
                f"Target has {n_cls} balanced class(es) — no imbalance correction needed."
            )

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        insights.append(
            f"Feature mix: {len(num_cols)} numeric, {len(cat_cols)} categorical columns "
            "— one-hot or target encoding recommended for categoricals."
        )

        return insights[:6]
