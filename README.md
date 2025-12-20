# MicroLLM Studio
Private, explainable, on-prem AI platform.
# 🧠 MicroLLM Studio

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-Proprietary-red.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)

**Enterprise-Grade Privacy-First Language Model Platform**

[Features](#-key-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Architecture](#-architecture) • [API](#-api-reference) • [Security](#-security)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage Examples](#-usage-examples)
- [API Reference](#-api-reference)
- [Training Guide](#-training-guide)
- [Deployment](#-deployment)
- [Security & Privacy](#-security--privacy)
- [Multi-Tenant Setup](#-multi-tenant-setup)
- [Explainability](#-explainability)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

**MicroLLM Studio** is an enterprise-grade language model platform designed for organizations that require:

- 🔒 **Maximum Privacy**: Differential privacy, federated learning, zero data retention
- 🏢 **Multi-Tenancy**: Complete data isolation between organizations
- 🔍 **Explainability**: Full interpretability for regulated industries (healthcare, finance, legal)
- 🏠 **On-Premises Deployment**: Run in your own infrastructure with complete control
- ⚡ **High Performance**: Optimized for edge devices and resource-constrained environments
- 🛡️ **Enterprise Security**: End-to-end encryption, RBAC, comprehensive audit logs

### Why MicroLLM Studio?

Unlike cloud-based LLM services, MicroLLM Studio gives you:

✅ **Data Sovereignty** - Your data never leaves your infrastructure  
✅ **Regulatory Compliance** - GDPR, HIPAA, SOC 2 ready  
✅ **Customization** - Fine-tune models on your proprietary data  
✅ **Cost Control** - No per-token pricing, predictable costs  
✅ **Transparency** - Open architecture, explainable AI  

---

## 🎯 Key Features

### 🧠 Advanced AI Architecture

#### ARSLM (Adaptive Recurrent Self-Learning Memory)
Novel neural architecture that combines the best of:
- **LSTM-style memory gating** for long-term dependencies
- **Adaptive attention mechanisms** for dynamic focus
- **Privacy-preserving gradients** for secure training
- **Low memory footprint** for edge deployment

```python
# Example: Initialize ARSLM model
from core.arslm.model import ARSLMModel

config = {
    'vocab_size': 50000,
    'embedding_dim': 256,
    'hidden_size': 512,
    'num_layers': 3,
    'differential_privacy': True,
    'noise_multiplier': 0.1
}

model = ARSLMModel(config)
```

### 🔒 Privacy & Security

| Feature | Description | Status |
|---------|-------------|--------|
| **Differential Privacy** | ε-δ privacy guarantees during training | ✅ |
| **Federated Learning** | Train without centralizing data | ✅ |
| **End-to-End Encryption** | AES-256 at rest, TLS 1.3 in transit | ✅ |
| **Key Rotation** | Automatic encryption key rotation | ✅ |
| **Audit Logs** | Immutable logs with integrity checks | ✅ |
| **Zero Knowledge** | No access to tenant data | ✅ |

### 🏢 Enterprise Multi-Tenancy

```yaml
Tenant Isolation:
  ✓ Separate database schemas per tenant
  ✓ Dedicated model instances
  ✓ Resource quotas and limits
  ✓ Role-based access control (RBAC)
  ✓ Usage tracking and billing
```

### 🔍 Explainability & Interpretability

```python
# Explain model predictions
from core.arslm.explainability import ExplainabilityAnalyzer

analyzer = ExplainabilityAnalyzer(model, tokenizer)
explanation = analyzer.explain_prediction(
    input_text="The patient shows symptoms of",
    target_token="fever"
)

print(explanation['feature_importance'])  # Token importance scores
print(explanation['attention_analysis'])   # Attention patterns
print(explanation['layer_contributions'])  # Layer-wise analysis
```

**Supported Methods:**
- Integrated Gradients
- Gradient SHAP
- Attention Visualization
- Layer-wise Relevance Propagation
- Bias Detection

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Applications                      │
│          (Web Dashboard, Mobile, API Clients)                │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                      API Gateway                             │
│              (FastAPI + Authentication)                      │
│  ┌──────────┬──────────┬──────────┬──────────┐             │
│  │   Auth   │ Tenants  │Inference │  Admin   │             │
│  │   JWT    │   RBAC   │  Router  │  Panel   │             │
│  └──────────┴──────────┴──────────┴──────────┘             │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   Core Services                              │
│  ┌────────────────┬──────────────────┬──────────────────┐  │
│  │  ARSLM Engine  │  Training Svc    │  Privacy Guard   │  │
│  │  - Inference   │  - Federated     │  - Diff Privacy  │  │
│  │  - Generation  │  - Fine-tuning   │  - Key Rotation  │  │
│  │  - Embedding   │  - Evaluation    │  - Encryption    │  │
│  └────────────────┴──────────────────┴──────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   Data Layer                                 │
│  ┌────────────────┬──────────────────┬──────────────────┐  │
│  │  PostgreSQL    │     Redis        │  Object Storage  │  │
│  │  (Metadata)    │    (Cache)       │  (Models/Data)   │  │
│  └────────────────┴──────────────────┴──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. **Core ARSLM Engine** (`core/arslm/`)
- `arscell.py` - Neural architecture implementation
- `model.py` - Complete language model with generation
- `federated.py` - Federated learning components
- `explainability.py` - Interpretability tools

#### 2. **API Layer** (`api/`)
- `main.py` - FastAPI application
- `auth/` - JWT authentication, RBAC
- `tenants/` - Multi-tenant management
- `inference/` - Inference endpoints
- `audit/` - Audit logging system

#### 3. **Training Pipeline** (`training/`)
- `trainer.py` - Training orchestration
- `dataset_manager.py` - Data loading and preprocessing
- `privacy_guard.py` - Differential privacy implementation

#### 4. **Security** (`security/`)
- `crypto.py` - Encryption operations
- `key_rotation.py` - Automatic key management
- `secrets.py` - Secret management

#### 5. **Dashboard** (`dashboard/`)
- `app.py` - Streamlit web interface
- `explainability_ui.py` - Visualization components
- `tenant_admin.py` - Admin panel

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10 or higher
- **Docker**: 20.10 or higher
- **Docker Compose**: 2.0 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **GPU**: CUDA-capable GPU (optional, for faster training)

### 1. Clone Repository

```bash
git clone https://github.com/your-org/microllm-studio.git
cd microllm-studio
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Copy environment template
cp .env.example .env
```

### 3. Configure Environment Variables

Edit `.env` file:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/microllm
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-super-secret-key-change-this-in-production
JWT_SECRET_KEY=another-secret-key-for-jwt

# Model Configuration
DEFAULT_MODEL_SIZE=base  # small, base, large
ENABLE_DIFFERENTIAL_PRIVACY=true
PRIVACY_EPSILON=1.0
PRIVACY_DELTA=1e-5

# Multi-Tenant
ENABLE_MULTI_TENANT=true
DEFAULT_TENANT_PLAN=standard

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4
```

### 4. Start with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

### 5. Access Applications

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:3000
- **Health Check**: http://localhost:8000/health

### 6. Create First User

```bash
# Using API
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@example.com",
    "password": "SecurePassword123!",
    "role": "admin"
  }'
```

---

## 💻 Installation

### Option 1: Docker (Recommended)

```bash
# Pull pre-built image
docker pull microllm/studio:latest

# Or build from source
docker build -t microllm/studio:latest -f deployment/docker/Dockerfile .

# Run
docker run -d \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e SECRET_KEY=... \
  --name microllm-api \
  microllm/studio:latest
```

### Option 2: Manual Installation

```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y \
  python3.10 \
  python3.10-dev \
  build-essential \
  postgresql-client \
  redis-tools

# Python dependencies
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Database setup
python scripts/init_database.py

# Run API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Option 3: Kubernetes (Production)

```bash
# Deploy to Kubernetes
kubectl apply -f deployment/k8s/namespace.yaml
kubectl apply -f deployment/k8s/secrets.yaml
kubectl apply -f deployment/k8s/configmap.yaml
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
kubectl apply -f deployment/k8s/ingress.yaml

# Check status
kubectl get pods -n microllm-studio
```

---

## 📚 Usage Examples

### Example 1: Text Generation

```python
import requests

API_URL = "http://localhost:8000/api/v1"
TOKEN = "your-jwt-token"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# Generate text
response = requests.post(
    f"{API_URL}/inference/generate",
    headers=headers,
    json={
        "text": "Artificial intelligence will",
        "max_length": 100,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9
    }
)

result = response.json()
print(result['generated_text'])
print(f"Inference time: {result['inference_time_ms']}ms")
```

### Example 2: Fine-Tuning Model

```python
from training.trainer import ARSLMTrainer
from training.dataset_manager import DatasetManager
from core.arslm.model import ARSLMModel
import torch

# Load pre-trained model
model = ARSLMModel.from_checkpoint('models/arslm-base.pt')

# Prepare dataset
dataset_manager = DatasetManager()
train_loader = dataset_manager.create_dataloader(
    data_path='data/custom_training_data.json',
    batch_size=32,
    shuffle=True
)

# Initialize trainer with privacy protection
trainer = ARSLMTrainer(
    model=model,
    train_dataloader=train_loader,
    learning_rate=0.0001,
    use_privacy=True,
    privacy_budget=1.0
)

# Train
history = trainer.train(
    num_epochs=10,
    save_path='checkpoints/custom-model',
    callback=lambda epoch, hist: print(f"Epoch {epoch}: Loss={hist['train_loss'][-1]}")
)

# Save final model
model.save_checkpoint('models/custom-model-final.pt')
```

### Example 3: Federated Learning

```python
from core.arslm.federated import FederatedAggregator, FederatedClient

# Server side: Initialize aggregator
aggregator = FederatedAggregator(
    aggregation_strategy='fedavg',
    min_clients=3,
    privacy_budget=1.0
)

# Client side: Train locally
client = FederatedClient(
    client_id="hospital_1",
    model=model,
    local_epochs=5
)

# Each client trains on local data
update = client.train_local(local_train_data)

# Send update to server (encrypted)
aggregator.add_client_update(
    client_id="hospital_1",
    model_update=update['updates'],
    num_samples=update['num_samples']
)

# Server aggregates updates
global_update = aggregator.aggregate()

# Distribute to all clients
model.load_state_dict(global_update)
```

### Example 4: Model Explainability

```python
from core.arslm.explainability import ExplainabilityAnalyzer

# Initialize analyzer
analyzer = ExplainabilityAnalyzer(model, tokenizer)

# Explain prediction
explanation = analyzer.explain_prediction(
    input_text="The stock market will likely",
    target_token="rise"
)

# Feature importance
for token, importance in explanation['feature_importance'].items():
    print(f"{token}: {importance:.4f}")

# Generate HTML report
report = analyzer.generate_explanation_report(
    input_text="The stock market will likely",
    format='html'
)

with open('explanation_report.html', 'w') as f:
    f.write(report)
```

### Example 5: Multi-Tenant API Usage

```python
# Admin: Create tenant
response = requests.post(
    f"{API_URL}/tenants",
    headers=admin_headers,
    json={
        "name": "Acme Corporation",
        "plan": "enterprise",
        "config": {
            "max_models": 10,
            "enable_federated": True
        }
    }
)

tenant_id = response.json()['tenant_id']

# Tenant user: Use API with tenant context
tenant_headers = {
    "Authorization": f"Bearer {tenant_token}",
    "X-Tenant-ID": tenant_id
}

# All operations isolated to this tenant
response = requests.get(
    f"{API_URL}/models",
    headers=tenant_headers
)
```

---

## 🔌 API Reference

### Authentication

#### POST `/api/v1/auth/login`
Authenticate user and receive JWT token.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "SecurePassword123!"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Inference

#### POST `/api/v1/inference/generate`
Generate text using ARSLM model.

**Request:**
```json
{
  "text": "Once upon a time",
  "max_length": 100,
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.9,
  "model_id": "arslm-base"
}
```

**Response:**
```json
{
  "generated_text": "Once upon a time, in a faraway kingdom...",
  "tokens_generated": 45,
  "inference_time_ms": 42.5,
  "model_id": "arslm-base",
  "confidence": 0.87
}
```

#### POST `/api/v1/inference/explain`
Get explanation for model prediction.

**Request:**
```json
{
  "text": "The patient shows symptoms of",
  "method": "integrated_gradients"
}
```

**Response:**
```json
{
  "input_text": "The patient shows symptoms of",
  "predicted_token": "fever",
  "confidence": 0.85,
  "feature_importance": {
    "patient": 0.35,
    "symptoms": 0.42,
    "shows": 0.08
  },
  "attention_weights": [...]
}
```

### Models

#### GET `/api/v1/models`
List all models for current tenant.

#### POST `/api/v1/models`
Create or deploy new model.

#### DELETE `/api/v1/models/{model_id}`
Delete model.

### Training

#### POST `/api/v1/training/start`
Start training job.

**Request:**
```json
{
  "model_name": "custom-finance-model",
  "dataset_path": "/data/finance_corpus.json",
  "config": {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "use_differential_privacy": true,
    "privacy_epsilon": 1.0
  }
}
```

#### GET `/api/v1/training/{job_id}/status`
Get training job status.

### Full API Documentation

Visit http://localhost:8000/docs for interactive Swagger documentation.

---

## 🎓 Training Guide

### Preparing Your Dataset

```python
# Format: JSONL with text field
# data/training_data.jsonl
{"text": "First training example..."}
{"text": "Second training example..."}
{"text": "Third training example..."}
```

### Training Script

```python
from training.trainer import ARSLMTrainer
from training.dataset_manager import DatasetManager
from core.arslm.model import ARSLMModel

# 1. Load or create model
config = {
    'vocab_size': 50000,
    'embedding_dim': 256,
    'hidden_size': 512,
    'num_layers': 3,
    'dropout': 0.1,
    'differential_privacy': True,
    'noise_multiplier': 0.1
}

model = ARSLMModel(config)

# 2. Prepare data
dm = DatasetManager()
train_loader = dm.create_dataloader(
    'data/training_data.jsonl',
    batch_size=32
)
val_loader = dm.create_dataloader(
    'data/validation_data.jsonl',
    batch_size=32
)

# 3. Configure trainer
trainer = ARSLMTrainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    learning_rate=0.0001,
    use_privacy=True,
    privacy_budget=1.0
)

# 4. Train
history = trainer.train(
    num_epochs=10,
    save_path='checkpoints',
    callback=training_callback
)

# 5. Evaluate
test_loss = trainer._validate()
print(f"Test loss: {test_loss:.4f}")
```

### Privacy Budget Management

```python
from training.privacy_guard import PrivacyGuard

privacy_guard = PrivacyGuard(
    enabled=True,
    epsilon=1.0,  # Privacy budget
    delta=1e-5,   # Failure probability
    max_grad_norm=1.0
)

# During training, check privacy spent
epsilon_spent, delta_spent = privacy_guard.get_privacy_spent(
    steps=1000,
    batch_size=32,
    dataset_size=10000
)

print(f"Privacy spent: ε={epsilon_spent:.2f}, δ={delta_spent:.2e}")
```

---

## 🐳 Deployment

### Production Deployment Checklist

- [ ] Change all default passwords and secrets
- [ ] Enable HTTPS/TLS with valid certificates
- [ ] Configure firewall rules
- [ ] Set up database backups
- [ ] Enable monitoring and alerting
- [ ] Configure log rotation
- [ ] Set resource limits
- [ ] Enable rate limiting
- [ ] Configure CORS properly
- [ ] Review security settings

### Docker Compose (Single Server)

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    image: microllm/studio:1.0.0
    restart: always
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - SECRET_KEY=${SECRET_KEY}
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
```

### Kubernetes (Multi-Server)

```bash
# Deploy to production cluster
kubectl create namespace microllm-prod

# Apply configurations
kubectl apply -f k8s/prod/ -n microllm-prod

# Scale deployment
kubectl scale deployment microllm-api --replicas=5 -n microllm-prod

# Rolling update
kubectl set image deployment/microllm-api \
  api=microllm/studio:1.1.0 \
  -n microllm-prod
```

### Environment-Specific Configurations

```bash
# Development
export ENV=development
export DEBUG=true
export LOG_LEVEL=DEBUG

# Staging
export ENV=staging
export DEBUG=false
export LOG_LEVEL=INFO

# Production
export ENV=production
export DEBUG=false
export LOG_LEVEL=WARNING
export ENABLE_METRICS=true
export ENABLE_TRACING=true
```

---

## 🛡️ Security & Privacy

### Differential Privacy

```python
# Configure privacy parameters
privacy_config = {
    'epsilon': 1.0,      # Privacy budget (lower = more private)
    'delta': 1e-5,       # Failure probability
    'max_grad_norm': 1.0  # Gradient clipping threshold
}

# Privacy is automatically enforced during training
trainer = ARSLMTrainer(
    model=model,
    use_privacy=True,
    privacy_budget=privacy_config['epsilon']
)
```

**Privacy Guarantees:**
- ε = 0.1: Very strong privacy
- ε = 1.0: Strong privacy (recommended)
- ε = 5.0: Moderate privacy
- ε = 10.0: Weak privacy

### Encryption

All s
