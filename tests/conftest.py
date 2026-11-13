"""Pytest configuration, fixtures, and mocked clients."""

from __future__ import annotations

import pytest

from agent.llm import FakeLLM
from agent.orchestrator import Orchestrator
from agent.tools.registry import reset_evidence_counter


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Set default test environment variables."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-000")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("K8S_NAMESPACE", "mlops-agent")
    monkeypatch.setenv("APPROVED_NAMESPACES", "mlops-agent")
    monkeypatch.setenv("ALLOWED_DAG_IDS", "training_pipeline")
    monkeypatch.setenv("MIN_F1", "0.85")
    monkeypatch.setenv("RETRAIN_ON_NEW_DATASET", "false")
    reset_evidence_counter()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def orchestrator(fake_llm):
    return Orchestrator(llm_client=fake_llm)
