"""
MicroLLM Studio - Streamlit Dashboard (STABLE VERSION)
Production-safe for Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np

# -------------------------------------------------
# Page config (MUST be first Streamlit call)
# -------------------------------------------------
st.set_page_config(
    page_title="MicroLLM Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Session state init
# -------------------------------------------------
st.session_state.setdefault("models", [])
st.session_state.setdefault("training_history", [])
st.session_state.setdefault("active_training", False)
st.session_state.setdefault("training_progress", 0)

# -------------------------------------------------
# Custom CSS (safe)
# -------------------------------------------------
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.markdown("## 🤖 MicroLLM Studio")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🎓 Training", "🔍 Models", "📊 Analytics", "⚙️ Settings"],
        key="nav_page"
    )

    st.markdown("---")
    st.markdown("### System Status")
    st.write("🟢 Active" if st.session_state.active_training else "⚪ Idle")
    st.write(f"Models: {len(st.session_state.models)}")

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
if page == "🏠 Dashboard":
    st.markdown('<div class="main-header">MicroLLM Studio Dashboard</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models", len(st.session_state.models))
    col2.metric("Training Jobs", len(st.session_state.training_history))
    col3.metric("GPU Usage", "45%")
    col4.metric("Avg Accuracy", "87%")

    st.markdown("---")

    epochs = list(range(1, 11))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=np.linspace(2.5, 1.0, 10), name="Train Loss"))
    fig.add_trace(go.Scatter(x=epochs, y=np.linspace(2.6, 1.1, 10), name="Val Loss"))
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TRAINING
# -------------------------------------------------
elif page == "🎓 Training":
    st.markdown('<div class="main-header">Training</div>', unsafe_allow_html=True)

    tabs = st.tabs(["New Training", "Queue", "History"])

    # ---- NEW TRAINING
    with tabs[0]:
        model_name = st.text_input("Model Name", key="model_name")
        model_type = st.selectbox(
            "Architecture",
            ["ARSLM-Micro", "ARSLM-Small", "ARSLM-Medium"],
            key="model_type"
        )
        epochs = st.slider("Epochs", 1, 50, 10, key="epochs")

        if st.button("🚀 Start Training", key="start_training"):
            if model_name:
                st.session_state.models.append({
                    "name": model_name,
                    "type": model_type,
                    "accuracy": "Training",
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                st.session_state.training_history.append({
                    "model": model_name,
                    "status": "running"
                })
                st.session_state.active_training = True
                st.session_state.training_progress = 0
                st.success("Training started")
            else:
                st.error("Model name required")

    # ---- QUEUE (SAFE PROGRESS)
    with tabs[1]:
        if st.session_state.active_training:
            progress = st.progress(st.session_state.training_progress)

            if st.session_state.training_progress < 100:
                st.session_state.training_progress += 5
                progress.progress(st.session_state.training_progress)
                st.info("Training in progress...")
            else:
                st.success("Training completed")
                st.session_state.active_training = False
        else:
            st.info("No active training")

    # ---- HISTORY
    with tabs[2]:
        if st.session_state.training_history:
            st.dataframe(pd.DataFrame(st.session_state.training_history))
        else:
            st.info("No history")

# -------------------------------------------------
# MODELS
# -------------------------------------------------
elif page == "🔍 Models":
    st.markdown('<div class="main-header">Models</div>', unsafe_allow_html=True)

    if not st.session_state.models:
        st.info("No models available")
    else:
        for i, model in enumerate(st.session_state.models):
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                col1.markdown(f"### {model['name']}")
                col2.write(model["type"])
                col3.button("Deploy", key=f"deploy_{i}")
                st.markdown("---")

# -------------------------------------------------
# ANALYTICS
# -------------------------------------------------
elif page == "📊 Analytics":
    st.markdown('<div class="main-header">Analytics</div>', unsafe_allow_html=True)

    dates = pd.date_range("2024-01-01", periods=30)
    acc = 70 + np.random.randn(30).cumsum()

    fig = px.line(x=dates, y=acc, labels={"x": "Date", "y": "Accuracy"})
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
else:
    st.markdown('<div class="main-header">Settings</div>', unsafe_allow_html=True)

    st.text_input("Project Name", "MicroLLM Studio", key="project_name")
    st.checkbox("Enable Encryption", True, key="encryption")
    st.button("Save Settings", key="save_settings")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#666'>MicroLLM Studio • Stable Build</div>",
    unsafe_allow_html=True
)