"""Pipeline Control page — trigger, monitor, and chat with the pipeline."""
import time
import httpx
import streamlit as st


def render_pipeline() -> None:
    st.markdown('<h2 class="gradient-text">⚙️ Pipeline Control</h2>', unsafe_allow_html=True)

    exp_id = st.session_state.get("experiment_id")
    if not exp_id:
        st.warning("⚠️ No dataset uploaded yet. Go to **Upload Dataset** first.")
        return

    st.markdown(f"**Experiment:** `{exp_id}`")

    # Pipeline settings
    with st.expander("⚙️ Pipeline Settings", expanded=False):
        target_col = st.text_input("Override target column", value="")
        selected_models = st.multiselect(
            "Select models to train",
            ["LogisticRegression", "RandomForestClassifier", "XGBClassifier",
             "LGBMClassifier", "RandomForestRegressor", "XGBRegressor", "LGBMRegressor", "Ridge"],
            default=[],
            help="Leave empty for auto-selection",
        )
        async_mode = st.toggle("Run asynchronously (Celery)", value=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        run_btn = st.button("🚀 Run Full Pipeline", use_container_width=True)

    if run_btn:
        payload = {
            "experiment_id": exp_id,
            "async_mode": async_mode,
        }
        if target_col:
            payload["target_column"] = target_col
        if selected_models:
            payload["selected_models"] = selected_models

        with st.spinner("Triggering pipeline..."):
            try:
                resp = httpx.post(
                    "http://localhost:8000/api/v1/pipeline/run",
                    json=payload,
                    timeout=30.0,
                )
                if resp.status_code in (200, 202):
                    result = resp.json()
                    st.session_state["task_id"] = result.get("task_id")
                    st.session_state["pipeline_running"] = True
                    st.success(f"✅ Pipeline started! Mode: {result.get('mode', 'sync')}")
                else:
                    st.error(f"Failed: {resp.text}")
            except Exception as e:
                st.error(f"API Error: {e}")

    # Live monitoring
    if st.session_state.get("pipeline_running") or True:
        st.markdown("---")
        st.markdown("### 🔄 Live Agent Monitor")

        try:
            resp = httpx.get(
                f"http://localhost:8000/api/v1/pipeline/status/{exp_id}",
                timeout=10.0,
            )
            if resp.status_code == 200:
                status = resp.json()
                _render_agent_monitor(status)
            else:
                st.info("No pipeline status available yet.")
        except Exception:
            _render_agent_monitor_demo()

    # Chat interface
    st.markdown("---")
    st.markdown("### 💬 Experiment Chat Assistant")
    _render_chat(exp_id)


def _render_agent_monitor(status: dict) -> None:
    """Render the agent execution monitor."""
    progress = status.get("progress_pct", 0)
    st.progress(progress / 100, text=f"Progress: {progress:.0f}%")

    current = status.get("current_stage", "idle")
    current_agent = status.get("current_agent", "")
    pipeline_status = status.get("status", "unknown")

    col1, col2, col3 = st.columns(3)
    with col1:
        color = {"completed": "success", "running": "info", "failed": "error"}.get(pipeline_status, "info")
        getattr(st, color)(f"Status: **{pipeline_status.upper()}**")
    with col2:
        st.info(f"Stage: **{current}**")
    with col3:
        if current_agent:
            st.info(f"Agent: **{current_agent}**")

    # Agent execution list
    executions = status.get("agent_executions", [])
    if executions:
        st.markdown("**Agent Execution Log:**")
        for record in executions:
            status_val = record.get("status", "pending")
            icon = {"completed": "✅", "running": "⏳", "failed": "❌", "pending": "⏸️"}.get(status_val, "❓")
            duration = record.get("duration_seconds", 0)
            confidence = record.get("confidence", 1.0)
            st.markdown(
                f'<div class="leaderboard-row">'
                f'<span>{icon}</span>'
                f'<span style="flex:1;font-weight:600">{record.get("agent_name","?")}</span>'
                f'<span class="agent-badge badge-{status_val}">{status_val}</span>'
                f'<span style="color:#64748b;font-size:0.8rem">{duration:.1f}s</span>'
                f'<span style="color:#64748b;font-size:0.8rem">conf: {confidence:.0%}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    # Messages
    messages = status.get("messages", [])
    if messages:
        st.markdown("**Pipeline Messages:**")
        for msg in messages[-5:]:
            content = msg.get("content", "")
            st.markdown(
                f'<div class="chat-bubble chat-bubble-agent">{content}</div>',
                unsafe_allow_html=True
            )

    # Errors
    errors = status.get("errors", [])
    if errors:
        with st.expander(f"⚠️ Errors ({len(errors)})", expanded=False):
            for err in errors:
                st.error(err)


def _render_agent_monitor_demo() -> None:
    """Demo view when API is not available."""
    st.info("Connect the API server to see live agent monitoring.")
    stages = [
        ("ingestion", "completed", 1.2, 0.95),
        ("eda", "completed", 4.5, 0.88),
        ("cleaning", "running", 2.1, 0.92),
        ("feature_engineering", "pending", 0, 0),
    ]
    for stage, status, duration, confidence in stages:
        icon = {"completed": "✅", "running": "⏳", "pending": "⏸️"}.get(status, "❓")
        st.markdown(
            f'<div class="leaderboard-row">'
            f'<span>{icon}</span>'
            f'<span style="flex:1;font-weight:600">{stage}_agent</span>'
            f'<span class="agent-badge badge-{status}">{status}</span>'
            f'<span style="color:#64748b;font-size:0.8rem">{duration:.1f}s</span>'
            f'<span style="color:#64748b;font-size:0.8rem">conf: {confidence:.0%}</span>'
            f'</div>',
            unsafe_allow_html=True
        )


def _render_chat(exp_id: str) -> None:
    """Render experiment chat interface."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display history
    for msg in st.session_state.chat_history:
        role_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-agent"
        prefix = "You" if msg["role"] == "user" else "🤖 AutonomDS"
        st.markdown(
            f'<div class="chat-bubble {role_class}">'
            f'<strong>{prefix}:</strong> {msg["content"]}'
            f'</div>',
            unsafe_allow_html=True
        )

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask about this experiment...", placeholder="e.g. Why was XGBoost selected? What are the top features?")
        submitted = st.form_submit_button("Send →")

    if submitted and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        try:
            resp = httpx.post(
                "http://localhost:8000/api/v1/pipeline/chat",
                json={"experiment_id": exp_id, "message": user_input},
                timeout=30.0,
            )
            if resp.status_code == 200:
                answer = resp.json().get("response", "No response.")
                st.session_state.chat_history.append({"role": "agent", "content": answer})
        except Exception as e:
            st.session_state.chat_history.append({"role": "agent", "content": f"Error: {e}"})
        st.rerun()
