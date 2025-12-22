"""
MicroLLM Studio - Streamlit Dashboard
A complete interface for training and managing specialized language models
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time

# Page Configuration
st.set_page_config(
    page_title="MicroLLM Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .status-badge {
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .status-active {
        background-color: #10b981;
        color: white;
    }
    .status-idle {
        background-color: #6b7280;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'models' not in st.session_state:
    st.session_state.models = []
if 'active_training' not in st.session_state:
    st.session_state.active_training = False

# Sidebar Navigation
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/667eea/ffffff?text=MicroLLM", width=200)
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🎓 Training", "🔍 Models", "📊 Analytics", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### System Status")
    system_status = "🟢 Active" if st.session_state.active_training else "⚪ Idle"
    st.markdown(f"**Status:** {system_status}")
    st.markdown(f"**Models:** {len(st.session_state.models)}")
    st.markdown(f"**Training Jobs:** {len(st.session_state.training_history)}")
    
    st.markdown("---")
    st.markdown("### Quick Actions")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    if st.button("📥 Export Report", use_container_width=True):
        st.info("Export functionality coming soon!")

# Main Content
if page == "🏠 Dashboard":
    st.markdown('<p class="main-header">MicroLLM Studio Dashboard</p>', unsafe_allow_html=True)
    st.markdown("**Democratizing Proprietary AI** - Train specialized language models on limited private data")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Active Models",
            value=len(st.session_state.models),
            delta="+2 this week"
        )
    
    with col2:
        st.metric(
            label="Training Jobs",
            value=len(st.session_state.training_history),
            delta="3 completed"
        )
    
    with col3:
        st.metric(
            label="GPU Usage",
            value="45%",
            delta="-5%"
        )
    
    with col4:
        st.metric(
            label="Avg. Accuracy",
            value="87.3%",
            delta="+2.1%"
        )
    
    st.markdown("---")
    
    # Recent Activity & System Overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Training Progress")
        
        # Sample training data
        epochs = list(range(1, 11))
        train_loss = [2.5, 2.1, 1.8, 1.6, 1.4, 1.3, 1.2, 1.1, 1.05, 1.0]
        val_loss = [2.6, 2.2, 1.9, 1.7, 1.5, 1.4, 1.3, 1.2, 1.15, 1.1]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines+markers', 
                                name='Training Loss', line=dict(color='#667eea', width=3)))
        fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers', 
                                name='Validation Loss', line=dict(color='#764ba2', width=3)))
        
        fig.update_layout(
            title="Loss Over Epochs",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Quick Stats")
        
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin-top:0;">ARSLM Core</h4>
            <p style="margin-bottom:0;">✅ Efficiency: 95%</p>
            <p style="margin-bottom:0;">✅ Explainability: High</p>
            <p style="margin-bottom:0;">✅ Security: Enterprise</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Resource Usage")
        st.progress(0.45, text="GPU: 45%")
        st.progress(0.62, text="CPU: 62%")
        st.progress(0.38, text="Memory: 38%")
        st.progress(0.71, text="Storage: 71%")
    
    st.markdown("---")
    
    # Recent Models
    st.subheader("🤖 Recent Models")
    
    if len(st.session_state.models) == 0:
        st.info("No models trained yet. Start by creating a new training job!")
    else:
        for i, model in enumerate(st.session_state.models[-3:]):
            with st.expander(f"📦 {model['name']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Type:** {model['type']}")
                    st.write(f"**Size:** {model['size']}")
                with col2:
                    st.write(f"**Accuracy:** {model['accuracy']}")
                    st.write(f"**Created:** {model['date']}")
                with col3:
                    st.button("🚀 Deploy", key=f"deploy_{i}")
                    st.button("📊 Details", key=f"details_{i}")

elif page == "🎓 Training":
    st.markdown('<p class="main-header">Model Training</p>', unsafe_allow_html=True)
    
    tabs = st.tabs(["New Training", "Training Queue", "History"])
    
    with tabs[0]:
        st.subheader("Create New Training Job")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.text_input("Model Name", placeholder="my-custom-model")
            model_type = st.selectbox(
                "Model Architecture",
                ["ARSLM-Micro (100M)", "ARSLM-Small (300M)", "ARSLM-Medium (1B)", "ARSLM-Large (3B)"]
            )
            dataset_source = st.selectbox(
                "Data Source",
                ["Upload Files", "S3 Bucket", "Local Directory", "Database"]
            )
            
        with col2:
            training_mode = st.selectbox(
                "Training Mode",
                ["Fine-tuning", "From Scratch", "Transfer Learning"]
            )
            batch_size = st.slider("Batch Size", 8, 128, 32)
            learning_rate = st.number_input(
                "Learning Rate", 
                min_value=0.00001, 
                max_value=0.01, 
                value=0.001,
                format="%.5f"
            )
        
        st.markdown("### Advanced Settings")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            epochs = st.number_input("Epochs", 1, 100, 10)
            warmup_steps = st.number_input("Warmup Steps", 0, 1000, 100)
        with col2:
            max_seq_length = st.number_input("Max Sequence Length", 128, 2048, 512)
            gradient_acc = st.number_input("Gradient Accumulation", 1, 16, 4)
        with col3:
            weight_decay = st.number_input("Weight Decay", 0.0, 0.1, 0.01, format="%.3f")
            dropout = st.slider("Dropout", 0.0, 0.5, 0.1)
        
        st.markdown("### Data Configuration")
        uploaded_file = st.file_uploader(
            "Upload Training Data",
            type=['txt', 'json', 'jsonl', 'csv'],
            help="Upload your training dataset"
        )
        
        validation_split = st.slider("Validation Split (%)", 0, 30, 10)
        
        st.markdown("### Security & Privacy")
        col1, col2 = st.columns(2)
        with col1:
            enable_encryption = st.checkbox("Enable Data Encryption", value=True)
            enable_audit = st.checkbox("Enable Audit Logging", value=True)
        with col2:
            enable_privacy = st.checkbox("Enable Differential Privacy", value=False)
            enable_federated = st.checkbox("Federated Learning Mode", value=False)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                if model_name:
                    new_model = {
                        'name': model_name,
                        'type': model_type,
                        'size': model_type.split('(')[1].strip(')'),
                        'accuracy': 'Training...',
                        'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'status': 'training'
                    }
                    st.session_state.models.append(new_model)
                    st.session_state.training_history.append({
                        'model': model_name,
                        'started': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'status': 'running'
                    })
                    st.session_state.active_training = True
                    st.success(f"Training started for {model_name}!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Please provide a model name")
        
        with col2:
            if st.button("💾 Save Configuration", use_container_width=True):
                st.info("Configuration saved!")
        
        with col3:
            st.markdown("💡 **Tip:** Start with ARSLM-Micro for quick experiments")
    
    with tabs[1]:
        st.subheader("Training Queue")
        
        if st.session_state.active_training:
            st.markdown("### Currently Training")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Training in progress... {i+1}%")
                time.sleep(0.01)
            
            st.success("Training simulation complete!")
        else:
            st.info("No active training jobs")
    
    with tabs[2]:
        st.subheader("Training History")
        
        if len(st.session_state.training_history) == 0:
            st.info("No training history available")
        else:
            df = pd.DataFrame(st.session_state.training_history)
            st.dataframe(df, use_container_width=True)

elif page == "🔍 Models":
    st.markdown('<p class="main-header">Model Management</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_query = st.text_input("🔍 Search models", placeholder="Search by name or type...")
    with col2:
        filter_type = st.selectbox("Filter by Type", ["All", "ARSLM-Micro", "ARSLM-Small", "ARSLM-Medium"])
    with col3:
        sort_by = st.selectbox("Sort by", ["Date", "Name", "Accuracy"])
    
    st.markdown("---")
    
    if len(st.session_state.models) == 0:
        st.info("No models available. Create your first model in the Training section!")
    else:
        for i, model in enumerate(st.session_state.models):
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                
                with col1:
                    st.markdown(f"### {model['name']}")
                    st.caption(f"Type: {model['type']}")
                
                with col2:
                    st.metric("Size", model['size'])
                
                with col3:
                    st.metric("Accuracy", model['accuracy'])
                
                with col4:
                    st.button("🚀 Deploy", key=f"deploy_model_{i}")
                    st.button("🗑️ Delete", key=f"delete_model_{i}")
                
                st.markdown("---")

elif page == "📊 Analytics":
    st.markdown('<p class="main-header">Analytics & Insights</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Performance", "Resources", "Comparisons"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance Over Time")
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            accuracy = [70 + i*0.5 + np.random.randn()*2 for i in range(30)]
            
            fig = px.line(x=dates, y=accuracy, labels={'x': 'Date', 'y': 'Accuracy (%)'})
            fig.update_traces(line_color='#667eea', line_width=3)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Training Time Distribution")
            training_times = [45, 67, 89, 34, 56, 78, 90, 45]
            fig = go.Figure(data=[go.Histogram(x=training_times, nbinsx=10)])
            fig.update_traces(marker_color='#764ba2')
            fig.update_layout(xaxis_title="Time (minutes)", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Resource Utilization")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=45,
                title={'text': "GPU Usage"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "#667eea"}}
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=62,
                title={'text': "CPU Usage"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "#764ba2"}}
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=38,
                title={'text': "Memory Usage"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "#667eea"}}
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Model Comparison")
        
        comparison_data = {
            'Model': ['ARSLM-Micro', 'ARSLM-Small', 'ARSLM-Medium'],
            'Accuracy': [82, 87, 91],
            'Training Time (hrs)': [2, 5, 12],
            'Parameters (M)': [100, 300, 1000]
        }
        
        df = pd.DataFrame(comparison_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['Model'], y=df['Accuracy'], name='Accuracy', marker_color='#667eea'))
        st.plotly_chart(fig, use_container_width=True)

else:  # Settings
    st.markdown('<p class="main-header">Settings</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["General", "Security", "API", "Advanced"])
    
    with tab1:
        st.subheader("General Settings")
        
        project_name = st.text_input("Project Name", value="MicroLLM Studio")
        workspace = st.text_input("Workspace Directory", value="/workspace/microllm")
        
        st.markdown("### Notification Settings")
        email_notifications = st.checkbox("Email Notifications", value=True)
        slack_notifications = st.checkbox("Slack Notifications", value=False)
        
        st.markdown("### Display Preferences")
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
        language = st.selectbox("Language", ["English", "Français", "Español"])
    
    with tab2:
        st.subheader("Security Settings")
        
        st.markdown("### Data Protection")
        encryption_enabled = st.checkbox("Enable End-to-End Encryption", value=True)
        audit_logging = st.checkbox("Enable Audit Logging", value=True)
        data_retention = st.slider("Data Retention (days)", 30, 365, 90)
        
        st.markdown("### Access Control")
        require_2fa = st.checkbox("Require Two-Factor Authentication", value=False)
        session_timeout = st.number_input("Session Timeout (minutes)", 15, 480, 60)
    
    with tab3:
        st.subheader("API Configuration")
        
        st.markdown("### API Keys")
        st.text_input("API Key", type="password", placeholder="Enter your API key")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔑 Generate New Key", use_container_width=True):
                st.success("New API key generated!")
        with col2:
            if st.button("📋 Copy Key", use_container_width=True):
                st.info("Key copied to clipboard!")
        
        st.markdown("### API Endpoints")
        st.code("https://api.microllm.studio/v1/train", language="bash")
        st.code("https://api.microllm.studio/v1/models", language="bash")
        st.code("https://api.microllm.studio/v1/inference", language="bash")
    
    with tab4:
        st.subheader("Advanced Settings")
        
        st.markdown("### Performance Tuning")
        max_workers = st.number_input("Max Workers", 1, 16, 4)
        cache_size = st.number_input("Cache Size (GB)", 1, 100, 10)
        
        st.markdown("### Debugging")
        debug_mode = st.checkbox("Enable Debug Mode", value=False)
        verbose_logging = st.checkbox("Verbose Logging", value=False)
        
        st.markdown("### Experimental Features")
        auto_scaling = st.checkbox("Auto-Scaling", value=False)
        distributed_training = st.checkbox("Distributed Training", value=False)
    
    st.markdown("---")
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            st.success("Settings saved successfully!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>MicroLLM Studio v1.0 | Built on ARSLM Framework</p>
    <p>Democratizing Proprietary AI for Enterprise & Regulated Environments</p>
</div>
""", unsafe_allow_html=True)

# Import numpy for random data
import numpy as np