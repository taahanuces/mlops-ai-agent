"""Configuration and environment variables for the MLOps AI Agent."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent

# LLM Configuration
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")

# MLflow
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

# Airflow 3
AIRFLOW_BASE_URL: str = os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080")
AIRFLOW_USERNAME: str = os.getenv("AIRFLOW_USERNAME", "admin")
AIRFLOW_PASSWORD: str = os.getenv("AIRFLOW_PASSWORD", "admin")

# Kubernetes
K8S_NAMESPACE: str = os.getenv("K8S_NAMESPACE", "mlops-agent")
APPROVED_NAMESPACES: list[str] = [
    ns.strip() for ns in os.getenv("APPROVED_NAMESPACES", "mlops-agent").split(",") if ns.strip()
]

# Policy
ALLOWED_DAG_IDS: list[str] = [
    dag.strip()
    for dag in os.getenv("ALLOWED_DAG_IDS", "training_pipeline").split(",")
    if dag.strip()
]
MAX_MODEL_AGE_DAYS: int = int(os.getenv("MAX_MODEL_AGE_DAYS", "30"))
MIN_F1: float = float(os.getenv("MIN_F1", "0.85"))
RETRAIN_ON_NEW_DATASET: bool = os.getenv("RETRAIN_ON_NEW_DATASET", "false").lower() in (
    "true",
    "1",
    "yes",
)
TOOL_TIMEOUT_S: int = int(os.getenv("TOOL_TIMEOUT_S", "10"))
CONFIRMATION_TTL_S: int = int(os.getenv("CONFIRMATION_TTL_S", "300"))
