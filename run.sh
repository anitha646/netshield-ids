#!/bin/bash
# ─────────────────────────────────────────────────────────────
# NetShield IDS — Setup & Run
# Usage: bash run.sh
# ─────────────────────────────────────────────────────────────
set -e

echo ""
echo "  ╔═══════════════════════════════════╗"
echo "  ║     NetShield IDS v1.0             ║"
echo "  ║  ML-based Intrusion Detection      ║"
echo "  ╚═══════════════════════════════════╝"
echo ""

# ── Install deps ──────────────────────────────────────────────
echo "[1/3] Installing dependencies..."
pip install -q -r requirements.txt

# ── Train model ───────────────────────────────────────────────
echo ""
echo "[2/3] Training ML model (RF + XGBoost on NSL-KDD)..."
cd "$(dirname "$0")"
python src/train_model.py data

# ── Launch dashboard ──────────────────────────────────────────
echo ""
echo "[3/3] Starting IDS dashboard..."
echo ""
echo "  ┌─────────────────────────────────────┐"
echo "  │  Dashboard: http://localhost:5000    │"
echo "  │  Press Ctrl+C to stop               │"
echo "  └─────────────────────────────────────┘"
echo ""
python app.py --port 5000
