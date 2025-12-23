import streamlit as st

# -------------------
# Initialisation session_state
# -------------------
if "tenant_selected" not in st.session_state:
    st.session_state.tenant_selected = None
if "model_selected" not in st.session_state:
    st.session_state.model_selected = None
if "feedback" not in st.session_state:
    st.session_state.feedback = ""

# -------------------
# Sidebar : sélection du tenant et du modèle
# -------------------
st.sidebar.title("MicroLLM Studio")
tenants = ["Tenant A", "Tenant B", "Tenant C"]
models = ["ARSLM Small", "ARSLM Medium", "ARSLM Large"]

# Sélection tenant
st.session_state.tenant_selected = st.sidebar.selectbox(
    "Select Tenant",
    tenants,
    key="tenant_selectbox"
)

# Sélection modèle
st.session_state.model_selected = st.sidebar.selectbox(
    "Select Model",
    models,
    key="model_selectbox"
)

# Action buttons
st.sidebar.button("Train Model", key="train_btn")
st.sidebar.button("Deploy Model", key="deploy_btn")
st.sidebar.button("Explain Model", key="explain_btn")

# -------------------
# Main Area : affichage
# -------------------
st.title("MicroLLM Studio Dashboard")

st.markdown(f"**Tenant:** {st.session_state.tenant_selected}")
st.markdown(f"**Model:** {st.session_state.model_selected}")

# Zone pour messages et feedback
st.text_area("Feedback / Logs", st.session_state.feedback, key="feedback_area", height=200)

# -------------------
# Exemple interaction backend
# -------------------
if st.button("Simulate Training", key="simulate_train_btn"):
    st.session_state.feedback += f"Training started for {st.session_state.model_selected} under {st.session_state.tenant_selected}\n"
    st.success("Training simulated successfully!")

if st.button("Simulate Deployment", key="simulate_deploy_btn"):
    st.session_state.feedback += f"Deployment started for {st.session_state.model_selected} under {st.session_state.tenant_selected}\n"
    st.success("Deployment simulated successfully!")

if st.button("Simulate Explainability", key="simulate_explain_btn"):
    st.session_state.feedback += f"Explainability run for {st.session_state.model_selected}\n"
    st.info("Explainability simulated successfully!")