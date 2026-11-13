#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo " Starting MLOps Platform Setup"
echo "=========================================================="

# 1. Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[1/6] Creating Python 3.11 virtual environment..."
    python3.11 -m venv .venv
else
    echo "[1/6] Python virtual environment exists."
fi

source .venv/bin/activate
echo "[2/6] Installing dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

# 2. Git & DVC
if [ ! -d ".git" ]; then
    echo "[3/6] Initializing Git repository..."
    git init
    git branch -M main
fi

if [ ! -d ".dvc" ]; then
    echo "[4/6] Initializing DVC..."
    dvc init --no-scm || dvc init
    mkdir -p ../dvc-remote
    dvc remote add -d -f localremote ../dvc-remote
fi

# 3. Kind Kubernetes Cluster
if command -v kind >/dev/null 2>&1; then
    if ! kind get clusters 2>/dev/null | grep -q "^mlops$"; then
        echo "[5/6] Creating kind cluster 'mlops'..."
        kind create cluster --name mlops
    else
        echo "[5/6] Kind cluster 'mlops' already exists."
    fi

    echo "Configuring namespace 'mlops-agent'..."
    kubectl create namespace mlops-agent --dry-run=client -o yaml | kubectl apply -f -
else
    echo "[Warning] 'kind' not found on PATH. Please ensure kind is installed."
fi

# 4. MLflow
echo "[6/6] Verifying MLflow background service..."
if ! curl -s http://localhost:5000/health >/dev/null 2>&1; then
    echo "Starting local MLflow tracking server in background on :5000..."
    mkdir -p mlruns
    nohup .venv/bin/mlflow server \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root ./mlruns \
        --host 0.0.0.0 --port 5000 > mlruns/mlflow.log 2>&1 &
    sleep 3
    echo "MLflow server started."
else
    echo "MLflow server is already running on :5000."
fi

echo "=========================================================="
echo " Setup complete! Next: make seed"
echo "=========================================================="
