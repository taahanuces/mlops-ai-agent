"""Deployed production model inference service with probes, prediction, and provenance."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List

import joblib
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

# Configuration from environment / ConfigMap
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
MODEL_RUN_ID = os.getenv("MODEL_RUN_ID", "local-run-001")
DATASET_VERSION = os.getenv("DATASET_VERSION", "data-v1.0")
GIT_COMMIT = os.getenv("GIT_COMMIT", "head")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("inference_api")

# Global model container
_model: Any = None


def load_model() -> Any:
    """Load model from file path or raise FileNotFoundError."""
    global _model
    p = Path(MODEL_PATH)
    if not p.is_file():
        logger.error(f"Configured model path does not exist: {MODEL_PATH}")
        _model = None
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    _model = joblib.load(p)
    logger.info(f"Successfully loaded model from {MODEL_PATH}")
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception as e:
        logger.warning(f"Startup model load deferred or failed: {e}")
    yield


app = FastAPI(
    title="MLOps Production Inference API",
    version=APP_VERSION,
    description="Containerized model serving service with health/ready probes and version provenance.",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    features: List[List[float]] = Field(
        ...,
        description="Feature matrix of samples (n_samples, n_features)",
        json_schema_extra={"example": [[5.1, 3.5, 1.4, 0.2]]},
    )


class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]] | None = None
    model_run_id: str
    dataset_version: str


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe indicating process is up and responding."""
    return {"status": "ok", "service": "inference-api"}


@app.get("/ready")
def ready() -> Response:
    """Readiness probe verifying that model is successfully loaded and ready for traffic."""
    global _model
    if _model is None:
        try:
            load_model()
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service not ready: Model could not be loaded from {MODEL_PATH}",
            )
    return Response(content='{"status": "ready"}', media_type="application/json")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Run model inference on submitted feature rows."""
    global _model
    if _model is None:
        try:
            load_model()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not loaded from {MODEL_PATH}",
            )

    try:
        import numpy as np

        X = np.array(request.features)
        raw_preds = _model.predict(X)
        preds = raw_preds.tolist() if hasattr(raw_preds, "tolist") else list(raw_preds)
        probs = None
        if hasattr(_model, "predict_proba"):
            raw_probs = _model.predict_proba(X)
            probs = raw_probs.tolist() if hasattr(raw_probs, "tolist") else list(raw_probs)

        return PredictResponse(
            predictions=preds,
            probabilities=probs,
            model_run_id=MODEL_RUN_ID,
            dataset_version=DATASET_VERSION,
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")


@app.get("/version")
def version() -> dict[str, str]:
    """Return deployed model metadata and training provenance."""
    return {
        "app_version": APP_VERSION,
        "model_run_id": MODEL_RUN_ID,
        "dataset_version": DATASET_VERSION,
        "git_commit": GIT_COMMIT,
        "model_path": MODEL_PATH,
    }
