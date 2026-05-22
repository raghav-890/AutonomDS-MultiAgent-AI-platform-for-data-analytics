"""
Report Generation Agent
========================
Produces downloadable artifacts:
- PDF report (ReportLab)
- Markdown report
- Experiment summary JSON
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agents.base_agent import BaseAgent, AgentError
from app.orchestration.state import AgentState, PipelineStage
from app.utils.config import get_settings
from app.utils.helpers import now_iso, safe_json_serialize


class ReportAgent(BaseAgent):
    name = "report_agent"
    description = "Generates PDF, Markdown, and JSON experiment reports"
    stage = PipelineStage.REPORT
    max_retries = 1

    def execute(self, state: AgentState) -> AgentState:
        settings = get_settings()
        exp_id = state.get("experiment_id", "default")
        report_dir = Path(settings.reports_dir) / exp_id
        report_dir.mkdir(parents=True, exist_ok=True)

        dataset_info = state.get("dataset_info", {})
        leaderboard = state.get("leaderboard", [])
        eda_insights = state.get("eda_insights", [])
        cleaning_actions = state.get("cleaning_actions", [])
        feature_report = state.get("feature_report", {})
        explainability = state.get("explainability_report", "")
        best_model = state.get("best_model_name", "N/A")

        # ── Markdown report ──────────────────────────────────────────────
        md_path = report_dir / "report.md"
        md_content = self._build_markdown(
            exp_id, dataset_info, leaderboard, eda_insights,
            cleaning_actions, feature_report, explainability, best_model, state
        )
        md_path.write_text(md_content, encoding="utf-8")

        # ── PDF report ───────────────────────────────────────────────────
        pdf_path = report_dir / "report.pdf"
        try:
            self._build_pdf(pdf_path, exp_id, dataset_info, leaderboard, best_model, eda_insights)
        except Exception as e:
            self.logger.warning("pdf_generation_failed", error=str(e))
            pdf_path = None  # type: ignore[assignment]

        # ── JSON summary ─────────────────────────────────────────────────
        summary = {
            "experiment_id": exp_id,
            "generated_at": now_iso(),
            "dataset": safe_json_serialize(dataset_info),
            "best_model": best_model,
            "leaderboard": safe_json_serialize(leaderboard[:5]),
            "eda_insights": eda_insights,
            "cleaning_actions": cleaning_actions,
            "feature_importance": safe_json_serialize(
                dict(list(state.get("feature_importance", {}).items())[:10])
            ),
        }
        json_path = report_dir / "summary.json"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # ── LLM executive summary ─────────────────────────────────────────
        exec_summary = self._generate_exec_summary(state)

        state = dict(state)  # type: ignore[assignment]
        state["markdown_report_path"] = str(md_path)
        state["pdf_report_path"] = str(pdf_path) if pdf_path else ""
        state["report_summary"] = exec_summary

        self.logger.info("report_generated", dir=str(report_dir))
        return state  # type: ignore[return-value]

    def _build_markdown(
        self, exp_id: str, dataset_info: dict, leaderboard: list,
        insights: list, actions: list, feature_report: dict,
        explainability: str, best_model: str, state: AgentState
    ) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"# AutonomDS Experiment Report",
            f"**Experiment ID:** `{exp_id}`  ",
            f"**Generated:** {now}  ",
            f"**Best Model:** {best_model}",
            "",
            "---",
            "",
            "## 📊 Dataset Overview",
            f"- **File:** {dataset_info.get('filename', 'N/A')}",
            f"- **Shape:** {dataset_info.get('n_rows', 0):,} rows × {dataset_info.get('n_cols', 0)} columns",
            f"- **Task Type:** {dataset_info.get('task_type', 'N/A')}",
            f"- **Target Column:** `{dataset_info.get('target_column', 'N/A')}`",
            f"- **Missing Data:** {dataset_info.get('missing_pct', 0):.1f}%",
            f"- **Memory:** {dataset_info.get('memory_mb', 0):.1f} MB",
            "",
            "### Dataset Summary",
            dataset_info.get("llm_summary", "_No summary available._"),
            "",
            "---",
            "",
            "## 🔍 EDA Insights",
        ]
        for insight in insights:
            lines.append(f"- {insight}")

        lines += ["", "---", "", "## 🧹 Data Cleaning", "Actions taken:"]
        for action in actions:
            lines.append(f"- {action}")

        lines += ["", "---", "", "## ⚙️ Feature Engineering"]
        lines.append(f"- Original features: {feature_report.get('n_features_original', 'N/A')}")
        lines.append(f"- Final features: {feature_report.get('n_features_final', 'N/A')}")
        for action in feature_report.get("actions", []):
            lines.append(f"- {action}")

        lines += ["", "---", "", "## 🏆 Model Leaderboard", ""]
        if leaderboard:
            headers = ["Rank", "Model", "Metric", "Score", "CV Mean", "Time (s)"]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for entry in leaderboard:
                metrics = entry.get("metrics", {})
                primary_metric = list(metrics.keys())[0] if metrics else "N/A"
                primary_score = metrics.get(primary_metric, 0)
                lines.append(
                    f"| {entry['rank']} | {entry['model_name']} | "
                    f"{primary_metric} | {primary_score:.4f} | "
                    f"{entry.get('cv_mean', 0):.4f} | "
                    f"{entry.get('training_time_seconds', 0):.1f} |"
                )

        lines += ["", "---", "", "## 🔬 Model Explainability", ""]
        if explainability:
            lines.append(explainability)
        else:
            lines.append("_Explainability report not available._")

        lines += ["", "---", "", "## 📝 Agent Execution Log", ""]
        for record in state.get("agent_executions", []):
            status = record.get("status", "unknown")
            icon = "✅" if status == "completed" else "❌"
            lines.append(
                f"- {icon} **{record.get('agent_name', '?')}** — "
                f"{status} ({record.get('duration_seconds', 0):.1f}s)"
            )

        lines += ["", "---", "", "*Generated by AutonomDS — Autonomous Data Science Platform*"]
        return "\n".join(lines)

    def _build_pdf(
        self, pdf_path: Path, exp_id: str, dataset_info: dict,
        leaderboard: list, best_model: str, insights: list
    ) -> None:
        """Generate PDF using ReportLab."""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=20, spaceAfter=12, alignment=TA_CENTER)
        h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceBefore=12, spaceAfter=6)
        body_style = styles["BodyText"]
        body_style.fontSize = 10

        story = [
            Paragraph("AutonomDS Experiment Report", title_style),
            Paragraph(f"Experiment: {exp_id}", body_style),
            Spacer(1, 0.2*inch),
            HRFlowable(width="100%"),
            Spacer(1, 0.1*inch),

            Paragraph("Dataset Overview", h2_style),
            Paragraph(f"File: {dataset_info.get('filename', 'N/A')}", body_style),
            Paragraph(f"Shape: {dataset_info.get('n_rows', 0):,} rows × {dataset_info.get('n_cols', 0)} columns", body_style),
            Paragraph(f"Task: {dataset_info.get('task_type', 'N/A')} | Target: {dataset_info.get('target_column', 'N/A')}", body_style),
            Spacer(1, 0.15*inch),

            Paragraph("Key EDA Insights", h2_style),
        ]

        for ins in insights[:5]:
            story.append(Paragraph(f"• {ins[:200]}", body_style))

        story += [Spacer(1, 0.15*inch), Paragraph("Model Leaderboard", h2_style)]

        if leaderboard:
            table_data = [["Rank", "Model", "Score", "Time (s)"]]
            for entry in leaderboard[:5]:
                metrics = entry.get("metrics", {})
                score = list(metrics.values())[0] if metrics else 0
                table_data.append([
                    str(entry["rank"]),
                    entry["model_name"][:30],
                    f"{score:.4f}",
                    f"{entry.get('training_time_seconds', 0):.1f}",
                ])
            t = Table(table_data, colWidths=[0.6*inch, 2.5*inch, 1*inch, 1*inch])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6366f1")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, 0), 11),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#818cf8")),
                ("TEXTCOLOR", (0, 1), (-1, 1), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 2), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
            ]))
            story.append(t)

        story += [
            Spacer(1, 0.2*inch),
            HRFlowable(width="100%"),
            Paragraph("Generated by AutonomDS — Autonomous Data Science Platform", body_style),
        ]

        doc.build(story)

    def _generate_exec_summary(self, state: AgentState) -> str:
        system = (
            "You are a senior data scientist. Write a 3-paragraph executive summary "
            "of this ML experiment for a business audience. Be specific about results."
        )
        lb = state.get("leaderboard", [])
        best_metrics = lb[0]["metrics"] if lb else {}
        user = (
            f"Experiment: {state.get('experiment_id')}\n"
            f"Dataset: {state.get('dataset_info', {}).get('filename')}, "
            f"{state.get('dataset_info', {}).get('n_rows', 0):,} rows\n"
            f"Task: {state.get('task_type')}\n"
            f"Best model: {state.get('best_model_name')} — Metrics: {best_metrics}\n"
            f"Top features: {list(state.get('feature_importance', {}).keys())[:5]}\n"
            "Write executive summary:"
        )
        return self.ask_llm(system, user)

    def _success_message(self, state: AgentState) -> str:
        return f"✅ Reports generated: PDF + Markdown + JSON summary for experiment {state.get('experiment_id')}."
