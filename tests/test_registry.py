"""Tests for tool registry and run_tool execution wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pydantic import BaseModel, Field

from agent.tools.registry import (
    get_all_tool_specs,
    next_evidence_id,
    register_tool,
    reset_evidence_counter,
    run_smoke,
    run_tool,
)


class DummyInput(BaseModel):
    val: int = Field(..., ge=0)


def test_registry_unregistered_tool():
    res = run_tool("non_existent_tool_xyz", {})
    assert res.status == "denied"
    assert "not registered" in res.error


def test_registry_validation_error():
    @register_tool("dummy_val_tool", "desc", "read", input_model=DummyInput)
    def dummy_fn(val: int):
        return {"val": val}

    # Pass invalid argument (negative integer)
    res = run_tool("dummy_val_tool", {"val": -5})
    assert res.status == "invalid_request"
    assert "validation error" in res.error.lower()


def test_registry_plain_return_and_exception():
    @register_tool("dummy_plain_tool", "desc", "read")
    def dummy_plain():
        return {"result": 42}

    res = run_tool("dummy_plain_tool", {})
    assert res.status == "ok"
    assert res.data == {"result": 42}

    @register_tool("dummy_err_tool", "desc", "read")
    def dummy_err():
        raise RuntimeError("boom")

    res_err = run_tool("dummy_err_tool", {})
    assert res_err.status == "error"
    assert "boom" in res_err.error


def test_get_all_tool_specs_and_reset_counter():
    specs = get_all_tool_specs()
    assert len(specs) >= 26

    reset_evidence_counter()
    ev1 = next_evidence_id()
    assert ev1 == "ev-0001"


def test_run_smoke_runner_mocked():
    with patch("agent.tools.registry.run_tool") as mock_run:
        mock_res = MagicMock()
        mock_res.status = "ok"
        mock_res.evidence_id = "ev-test"
        mock_res.duration_ms = 5
        mock_res.error = None
        mock_run.return_value = mock_res

        out = run_smoke()
        assert out["tools_registered"] >= 26
        assert "airflow_list_dags" in out["results"]
