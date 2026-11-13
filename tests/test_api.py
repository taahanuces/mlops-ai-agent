"""Tests for FastAPI agent endpoints and inference service."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from agent.llm import FakeLLM, LLMResponse
from agent.main import app as agent_app
from agent.main import set_orchestrator
from agent.orchestrator import Orchestrator
from agent.permissions import confirmation_store
from api.main import app as inference_app


@pytest.fixture
def agent_client():
    return TestClient(agent_app)


@pytest.fixture
def inference_client():
    return TestClient(inference_app)


def test_agent_health(agent_client):
    """Verify agent /health returns ok."""
    resp = agent_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_agent_chat_endpoint(agent_client):
    """Verify /chat executes turn and returns structured response with headers."""
    fake_llm = FakeLLM(
        [
            LLMResponse(
                content=json.dumps(
                    {
                        "summary": "Agent responded successfully.",
                        "known_facts": [],
                        "uncertainties": [],
                        "recommended_actions": [],
                        "pending_confirmations": [],
                    }
                )
            )
        ]
    )
    set_orchestrator(Orchestrator(fake_llm))

    resp = agent_client.post("/chat", json={"message": "Hello agent"})
    assert resp.status_code == 200
    data = resp.json()
    assert "request_id" in data
    assert "X-Request-ID" in resp.headers
    assert data["answer"]["summary"] == "Agent responded successfully."


def test_agent_confirm_endpoint(agent_client):
    """Verify /confirm executes pending action once and rejects rerun."""
    executed = False

    def test_cb():
        nonlocal executed
        executed = True
        return {"restarted": True}

    cid = confirmation_store.store_action(
        "k8s_restart_deployment", {}, "Restart test", "kubernetes", test_cb
    )

    resp = agent_client.post(f"/confirm/{cid}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert executed is True

    # Duplicate call should fail
    resp_dup = agent_client.post(f"/confirm/{cid}")
    assert resp_dup.status_code == 400


def test_inference_api_endpoints(inference_client, tmp_path, monkeypatch):
    """Verify inference API /health, /ready (503 when missing), /predict, and /version."""
    # Health probe
    r_health = inference_client.get("/health")
    assert r_health.status_code == 200

    # Ready probe when model missing -> 503
    monkeypatch.setattr("api.main.MODEL_PATH", str(tmp_path / "nonexistent.joblib"))
    monkeypatch.setattr("api.main._model", None)
    r_ready_fail = inference_client.get("/ready")
    assert r_ready_fail.status_code == 503

    # Version probe
    r_ver = inference_client.get("/version")
    assert r_ver.status_code == 200
    assert "app_version" in r_ver.json()
    assert "model_run_id" in r_ver.json()

    # Create dummy model and test ready + predict
    dummy_model = MagicMock()
    dummy_model.predict.return_value = [1]
    dummy_model.predict_proba.return_value = [[0.1, 0.9]]
    monkeypatch.setattr("api.main._model", dummy_model)

    r_ready_ok = inference_client.get("/ready")
    assert r_ready_ok.status_code == 200

    r_predict = inference_client.post("/predict", json={"features": [[1.0, 2.0, 3.0, 4.0]]})
    assert r_predict.status_code == 200
    assert r_predict.json()["predictions"] == [1]
