# 🏙️ Smart City Monitor

[![Tests](https://github.com/<you>/Neural-Network-Types/actions/workflows/test.yml/badge.svg)](https://github.com/<you>/Neural-Network-Types/actions/workflows/test.yml)
[![Lint](https://github.com/<you>/Neural-Network-Types/actions/workflows/lint.yml/badge.svg)](https://github.com/<you>/Neural-Network-Types/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/<you>/Neural-Network-Types/branch/main/graph/badge.svg)](https://codecov.io/gh/<you>/Neural-Network-Types)
[![🤗 Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Spaces-blue)](https://huggingface.co/spaces/marco-pcg/Neural-Network-Types)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue)](https://ghcr.io/<you>/neural-network-types)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Five neural network architectures. One API. One demo. Fully automated.**

A multi-modal AI system that demonstrates when and why to use each major neural network architecture — built end-to-end with training, inference, serving, containerization, and CI/CD.

🔴 **[Live Demo](https://huggingface.co/spaces/marco-pcg/Neural-Network-Types)** · 📖 **[API Docs](http://localhost:8000/docs)** · 🐳 **[Docker Image](https://ghcr.io/<you>/neural-network-types)**

---

## 📖 Overview

Different problems demand different architectures. Instead of building five disconnected notebooks, this project unifies five models behind a single REST API and an interactive Gradio demo, then ships them with Docker and automated CI/CD.

| # | Architecture | Task | Data Type | Why This Architecture |
|---|--------------|------|-----------|----------------------|
| 01 | **CNN** | Traffic sign classification | Images | Convolutional filters preserve spatial locality and detect visual features. |
| 02 | **LSTM** | Energy demand forecasting | Time series | Gated recurrence captures long-range temporal dependencies. |
| 03 | **GNN** | Fraud ring detection | Graphs | Message passing over edges lets the model reason about relationships. |
| 04 | **Transformer** | Citizen report routing | Text | Self-attention captures full-sentence semantic context. |
| 05 | **Autoencoder** | Water leak detection | Sensor data | Trained only on normal data; high reconstruction error = anomaly. |

---

## 🏗️ Architecture
┌──────────────────────────────────────────────────────────────────┐
│ Smart City Monitor │
├──────────────────────────────────────────────────────────────────┤
│ │
│ Image ──▶ [ CNN ]──────────▶ 🚦 Traffic Sign │
│ Time ──▶ [ LSTM ]─────────▶ ⚡ Energy Forecast │
│ Graph ──▶ [ GNN ]──────────▶ 🕸️ Fraud Detection │
│ Text ──▶ [ Transformer ]──▶ 📨 Report Routing │
│ Sensor──▶ [ Autoencoder ]──▶ 💧 Leak Detection │
│ │
│ ▼ │
│ ┌──────────────────┐ ┌─────────────────┐ │
│ │ FastAPI API │ │ Gradio Demo │ │
│ │ /traffic │ │ (5 tabs) │ │
│ │ /energy │ │ │ │
│ │ /fraud │ │ │ │
│ │ /route │ │ │ │
│ │ /leak │ │ │ │
│ └──────────────────┘ └─────────────────┘ │
│ │
└──────────────────────────────────────────────────────────────────┘
---

## 🚀 Quick Start

### Option 1: Run Locally

```bash
git clone https://github.com/<you>/Neural-Network-Types.git
cd Neural-Network-Types

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Train all five models (or download pretrained from Releases):
```bash

python 01_cnn_traffic.py
python 02_lstm_energy.py
python 03_gnn_fraud.py
python 04_transformer_reports.py
python 05_autoencoder_leaks.py
```
Verify all artifacts loaded correctly:
```bash

python verify_models.py
```
Option 2: Docker
```bash

docker compose up --build
```
Service	URL
REST API + Swagger UI	http://localhost:8000/docs
Gradio Demo (5 tabs)	http://localhost:7860
Option 3: Hugging Face Spaces

Skip the setup entirely and try the hosted demo:

🔴 https://huggingface.co/spaces/marco-pcg/Neural-Network-Types
🧠 Model Details
01 · CNN — Traffic Sign Classification

Recognizes stop signs, yield signs, and speed limits from dashboard camera images.

    Input: 64×64 RGB image

    Architecture: 2 × (Conv2d → ReLU → MaxPool) → 2 × Linear

    Output: Class label + top-3 confidences

    Artifacts: traffic_sign_cnn.pth, class_names.json

02 · LSTM — Energy Demand Forecasting

Predicts tomorrow's electricity consumption from the previous 30 days.

    Input: Sequence of 30 daily kWh values

    Architecture: 2-layer LSTM (hidden=50) → Linear head

    Output: Next-day kWh forecast

    Artifacts: energy_lstm.pth, energy_scaler.pkl, energy_metadata.json

⚠️ The scaler is a required artifact — without it, predictions are off by orders of magnitude.
03 · GNN — Fraud Ring Detection

Flags fraudulent accounts based on their position in a transaction graph.

    Input: Node feature matrix + edge index

    Architecture: 2 × GCNConv → softmax over 2 classes

    Output: Per-node fraud / legit with probability

    Artifacts: fraud_gcn.pth, node_features.json

04 · Transformer — Citizen Report Routing

Routes citizen complaints ("broken pipe on Elm St.") to the right department.

    Input: Free-text report

    Architecture: Pretrained all-MiniLM-L6-v2 embeddings + LogisticRegression

    Output: Department + confidence + full score distribution

    Artifacts: report_classifier.pkl, label_map.json, encoder_config.json

💡 The Sentence Transformer itself is not saved — only its name is recorded and re-downloaded at inference time.
05 · Autoencoder — Water Leak Detection

Detects pipe leaks from pressure sensor readings without ever having seen a leak during training.

    Input: 10 sensor pressure values

    Architecture: 10 → 6 → 3 (bottleneck) → 6 → 10

    Output: Anomaly score, severity (normal / moderate / high)

    Artifacts: leak_autoencoder.pth, anomaly_threshold.json, sensor_config.json

Threshold is computed as mean + 3σ of training reconstruction error (99.7% confidence).
🔌 API Endpoints

Once the API is running (uvicorn api:app --reload), visit http://localhost:8000/docs for the interactive Swagger UI.
Method	Endpoint	Description
GET	/health	Service status
POST	/traffic	Classify an uploaded traffic sign image
POST	/energy	Forecast next-day kWh from history
POST	/fraud	Predict fraud labels for a graph
POST	/route	Route a citizen report to a department
POST	/leak	Detect water leak from sensor readings
Example requests
```bash

# Traffic (image upload)
curl -X POST -F "file=@data/images/stop/test.png" http://localhost:8000/traffic

# Energy forecast
curl -X POST http://localhost:8000/energy \
  -H "Content-Type: application/json" \
  -d '{"history": [1200,1150,1300,1250,1400,1350,1280,1220,1190,1310,
                   1420,1380,1260,1240,1210,1330,1450,1400,1290,1270,
                   1230,1180,1300,1440,1390,1270,1250,1220,1200,1340]}'

# Fraud detection
curl -X POST http://localhost:8000/fraud \
  -H "Content-Type: application/json" \
  -d '{"features": [[1,0.5],[0.5,1],[0.1,0.1],[5,5]],
       "edges": [[0,1,2,3],[1,2,3,0]]}'

# Route a report
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"text": "Broken pipe leaking water on Elm Street."}'

# Leak detection
curl -X POST http://localhost:8000/leak \
  -H "Content-Type: application/json" \
  -d '{"readings": [5,5,0.1,0.1,5,5,0.1,0.1,5,5]}'
```
💻 CLI Usage

A unified CLI exposes all five models:
```bash

# Traffic sign classification
python predict.py traffic data/images/stop/test.png

# Energy forecast (from a JSON file with 30 values)
python predict.py energy history.json

# Fraud detection (from a JSON file with features + edges)
python predict.py fraud graph.json

# Citizen report routing
python predict.py route --text "Broken pipe on Elm Street."
python predict.py route --batch reports.json

# Leak detection
python predict.py leak readings.json
```
🧪 Testing
```bash

pytest tests/ -v --cov=inference --cov-report=term-missing
```
Smoke tests cover all five inference modules — missing artifacts, wrong input shapes, and cold-start loading are all verified.
📁 Project Structure
```text

Neural-Network-Types/
├── models/                          # Trained artifacts (via Git LFS)
│   ├── 01/ traffic_sign_cnn.pth
│   ├── 02/ energy_lstm.pth, energy_scaler.pkl, energy_metadata.json
│   ├── 03/ fraud_gcn.pth, node_features.json
│   ├── 04/ report_classifier.pkl, label_map.json, encoder_config.json
│   └── 05/ leak_autoencoder.pth, anomaly_threshold.json, sensor_config.json
├── inference/                       # One module per architecture
│   ├── cnn_traffic.py
│   ├── lstm_energy.py
│   ├── gnn_fraud.py
│   ├── transformer_reports.py
│   └── autoencoder_leaks.py
├── tests/
│   └── test_inference.py
├── .github/workflows/
│   ├── test.yml                     # CI: run pytest on push
│   ├── lint.yml                     # CI: ruff + black
│   ├── docker.yml                   # CD: publish image to GHCR
│   ├── release.yml                  # CD: attach models to releases
│   └── sync-to-hf.yml               # CD: mirror to Hugging Face Spaces
├── api.py                           # FastAPI backend
├── app.py                           # Gradio demo
├── predict.py                       # Unified CLI
├── verify_models.py                 # Artifact sanity check
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml                   # ruff + black config
├── .gitignore
├── .gitattributes                   # Git LFS tracking
└── README.md
```
🛠️ Tech Stack
```
Layer	Tools
Deep Learning	PyTorch, PyTorch Geometric, Sentence-Transformers
Classical ML	scikit-learn, joblib
Data	NumPy, pandas, Pillow
Serving	FastAPI, Uvicorn
Demo	Gradio
Deployment	Docker, Docker Compose, Hugging Face Spaces
CI/CD	GitHub Actions, Codecov, GHCR
Quality	pytest, ruff, black
```
🔄 CI/CD Pipeline
```
Trigger	Workflow	Action
Push to main / develop	test.yml	Run pytest on Python 3.11 + 3.12
Push to main / develop	lint.yml	Run ruff + black --check
Push to main	sync-to-hf.yml	Mirror code to Hugging Face Spaces
Push tag v*.*.*	docker.yml	Build + push Docker image to GHCR
Publish a release	release.yml	Attach models-vX.Y.Z.zip
```

Model Storage Strategy

Model weights are large and should not live in the Space repository. This project uses:

    Git LFS on GitHub for models/**/*.pth and *.pkl

    Runtime download inside Hugging Face Spaces (weights fetched from a separate model repo)

    GitHub Releases for versioned model bundles (models-v1.0.0.zip)

This keeps the code repository lean and the deployment reproducible.
🗺️ Roadmap

    ☑

    Five architectures with training + inference
    ☑

    Unified FastAPI backend
    ☑

    Interactive Gradio demo
    ☑

    Docker + Docker Compose packaging
    ☑

    GitHub Actions CI/CD
    ☑

    Hugging Face Spaces deployment
    □

    MLflow experiment tracking
    □

    Drift monitoring on live predictions
    □

    gRPC serving variant for high-throughput scenarios

📜 License

MIT — see LICENSE.
🙏 Acknowledgments

    PyTorch Geometric for graph neural network primitives

    Sentence-Transformers for accessible text embeddings

    FastAPI for zero-friction API development

    Gradio for shareable ML demos

<p align="center"> <em>Built as a portfolio project demonstrating when and why to choose each neural network architecture.</em><br> <a href="https://huggingface.co/spaces/marco-pcg/Neural-Network-Types">🔴 Try it live</a> · <a href="https://github.com/<you>/Neural-Network-Types/issues">🐛 Report a bug</a> · <a href="https://github.com/<you>/Neural-Network-Types/discussions">💬 Discuss</a> </p>
