# NetShield IDS — ML-based Intrusion Detection System

> Real-time network intrusion detection using Random Forest + XGBoost on NSL-KDD dataset, with a live alert dashboard.

## Architecture

```
Network Traffic
     │
     ▼
Feature Extractor          ← extract 18 flow features (duration, bytes, flags, rates...)
     │
     ▼
ML Classifier              ← Random Forest / XGBoost trained on NSL-KDD
     │
     ▼
Prediction Engine          ← label + confidence + severity
     │
     ▼
Flask SSE Backend          ← real-time server-sent events
     │
     ▼
Live Dashboard             ← flow table, alert log, charts
```

## Attack Classes

| Label  | Description                          | Severity  |
|--------|--------------------------------------|-----------|
| NORMAL | Legitimate traffic                   | None      |
| DoS    | Denial of Service (SYN flood, etc.)  | Critical  |
| Probe  | Port scan, network reconnaissance    | Medium    |
| R2L    | Remote to Local exploit attempt      | High      |
| U2R    | Privilege escalation attack          | Critical  |

## Quick Start

```bash
# 1. Clone or download project
cd ids_project

# 2. One-command setup (installs deps + trains model + launches dashboard)
bash run.sh

# 3. Open browser
http://localhost:5000
```

## With Real NSL-KDD Dataset

```bash
# Download from https://www.unb.ca/cic/datasets/nsl.html
# Place KDDTrain+.txt and KDDTest+.txt in ./data/
bash run.sh
```

## Manual Steps

```bash
pip install -r requirements.txt

# Train model (auto-downloads synthetic data if NSL-KDD not found)
python src/train_model.py

# Run dashboard
python app.py --port 5000
```

## API Endpoints

| Endpoint           | Method | Description                    |
|--------------------|--------|--------------------------------|
| `/`                | GET    | Live dashboard                 |
| `/api/stream`      | GET    | SSE stream of predictions      |
| `/api/predict`     | POST   | Single prediction from JSON    |
| `/api/stats`       | GET    | Aggregate stats                |
| `/api/alerts`      | GET    | Last 20 attack alerts          |
| `/api/events`      | GET    | Last 50 flow events            |
| `/api/status`      | GET    | Model info                     |

## POST /api/predict — Example

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 0.0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "S0",
    "src_bytes": 0,
    "dst_bytes": 0,
    "serror_rate": 1.0,
    "count": 511
  }'
```

## Features Used (18 total)

| Feature              | Description                              |
|----------------------|------------------------------------------|
| duration             | Connection duration in seconds           |
| protocol_type        | tcp / udp / icmp                         |
| service              | http / ftp / smtp / ssh / dns ...        |
| flag                 | TCP flag state (SF, S0, REJ, RSTO ...)   |
| src_bytes            | Bytes sent from source                   |
| dst_bytes            | Bytes sent to destination                |
| land                 | Same src/dst host:port (1=yes)           |
| wrong_fragment       | Malformed fragment count                 |
| urgent               | TCP urgent flag count                    |
| hot                  | Hot indicators (root access attempts)    |
| num_failed_logins    | Failed login attempts                    |
| logged_in            | User logged in successfully              |
| count                | Connections to same host in 2s window    |
| srv_count            | Connections to same service in 2s window |
| serror_rate          | % connections with SYN errors            |
| same_srv_rate        | % connections to same service            |
| dst_host_count       | Connections to same dest host            |
| dst_host_srv_count   | Connections to same dest host + service  |

## Project Structure

```
ids_project/
├── src/
│   ├── train_model.py      # ML training pipeline
│   ├── feature_extractor.py # Flow feature extraction + simulation
│   └── predictor.py        # Model inference engine
├── models/                 # Saved model artifacts (auto-generated)
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── meta.json
├── data/                   # Place NSL-KDD .txt files here
├── templates/
│   └── dashboard.html      # Real-time dashboard
├── app.py                  # Flask backend + SSE
├── requirements.txt
├── run.sh
└── README.md
```

## Tech Stack

- **ML**: scikit-learn, XGBoost, pandas, numpy
- **Backend**: Flask, SSE (server-sent events)
- **Frontend**: Vanilla JS, Canvas API, JetBrains Mono font
- **Dataset**: NSL-KDD (or synthetic fallback)

## Results (NSL-KDD)

| Model         | Accuracy  |
|---------------|-----------|
| Random Forest | ~98.5%    |
| XGBoost       | ~99.1%    |

*Best model is auto-selected and saved.*

---
Built by Adithiya · SRM Institute of Science and Technology · AI & ML, Semester 5
