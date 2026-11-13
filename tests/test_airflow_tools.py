"""Tests for Airflow 3 API tools, JWT handling, and trigger gating."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from agent.permissions import confirmation_store
from agent.tools.airflow_tools import (
    _airflow_request,
    _get_auth_token,
    airflow_get_dag_runs,
    airflow_get_failed_tasks,
    airflow_get_task_logs,
    airflow_list_dags,
    airflow_trigger_training_dag,
)


def test_airflow_5xx_service_failure():
    """Verify tool returns unavailable/error when Airflow returns 500 error."""
    with patch(
        "agent.tools.airflow_tools._airflow_request", return_value=(500, "Internal Server Error")
    ):
        res = airflow_list_dags()
        assert res.status == "unavailable"
        assert "500" in res.error


def test_invalid_request_unknown_dag():
    """Verify trigger tool refuses DAGs not in ALLOWED_DAG_IDS."""
    res = airflow_trigger_training_dag(dag_id="malicious_or_unknown_dag")
    assert res.status == "denied"
    assert "not in the allowed list" in res.error


def test_airflow_trigger_requires_confirmation_and_callback():
    """Verify triggering an allow-listed DAG returns pending_confirmation and callback executes."""
    res = airflow_trigger_training_dag(dag_id="training_pipeline")
    assert res.status == "pending_confirmation"
    cid = res.data["confirmation_id"]
    assert cid.startswith("conf-")

    # Now verify the callback in confirmation_store
    with patch(
        "agent.tools.airflow_tools._airflow_request",
        return_value=(201, {"dag_run_id": "test_run_1", "state": "queued"}),
    ):
        action, confirmed, err = confirmation_store.execute_and_consume(cid)
        assert err is None
        assert confirmed["dag_run_id"] == "test_run_1"
        assert confirmed["state"] == "queued"


def test_airflow_get_failed_tasks():
    """Verify parsing of failed task instances."""
    mock_tis = {
        "task_instances": [
            {"task_id": "validate_dataset", "state": "failed", "try_number": 1},
            {"task_id": "preprocess_data", "state": "upstream_failed", "try_number": 0},
            {"task_id": "train_model", "state": "upstream_failed", "try_number": 0},
        ]
    }
    with patch("agent.tools.airflow_tools._airflow_request", return_value=(200, mock_tis)):
        res = airflow_get_failed_tasks(dag_id="training_pipeline", dag_run_id="run-123")
        assert res.status == "ok"
        assert len(res.data["failed_tasks"]) == 3


def test_airflow_list_dags_success_and_timeout():
    mock_data = {
        "dags": [
            {
                "dag_id": "training_pipeline",
                "is_paused": False,
                "description": "test",
                "tags": [{"name": "mlops"}],
            }
        ]
    }
    with patch("agent.tools.airflow_tools._airflow_request", return_value=(200, mock_data)):
        res = airflow_list_dags()
        assert res.status == "ok"
        assert res.data["count"] == 1

    with patch(
        "agent.tools.airflow_tools._airflow_request", side_effect=httpx.TimeoutException("timeout")
    ):
        res = airflow_list_dags()
        assert res.status == "timeout"


def test_airflow_get_dag_runs_success_and_timeout():
    mock_data = {
        "dag_runs": [
            {
                "dag_run_id": "r1",
                "state": "success",
                "start_date": "2026-01-01",
                "end_date": "2026-01-01",
            }
        ]
    }
    with patch("agent.tools.airflow_tools._airflow_request", return_value=(200, mock_data)):
        res = airflow_get_dag_runs(dag_id="training_pipeline", limit=1)
        assert res.status == "ok"
        assert len(res.data["runs"]) == 1

    with patch(
        "agent.tools.airflow_tools._airflow_request", side_effect=httpx.TimeoutException("timeout")
    ):
        res = airflow_get_dag_runs(dag_id="training_pipeline")
        assert res.status == "timeout"


def test_airflow_get_task_logs_formats():
    # String format
    with patch(
        "agent.tools.airflow_tools._airflow_request", return_value=(200, "line 1\nline 2\nline 3")
    ):
        res = airflow_get_task_logs(dag_id="d", dag_run_id="r", task_id="t", tail_lines=2)
        assert res.status == "ok"
        assert "line 2" in res.data["logs"]

    # List of dicts format
    structured_logs = [
        {"timestamp": "2026-01-01T00:00:00Z", "level": "INFO", "event": "Task started"},
        {"timestamp": "2026-01-01T00:00:01Z", "level": "ERROR", "event": "Task failed"},
    ]
    with patch("agent.tools.airflow_tools._airflow_request", return_value=(200, structured_logs)):
        res = airflow_get_task_logs(dag_id="d", dag_run_id="r", task_id="t", tail_lines=5)
        assert res.status == "ok"
        assert "Task failed" in res.data["logs"]


def test_get_auth_token_and_request():
    with patch("httpx.Client") as mock_client_cls:
        client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = client

        # 200 from first endpoint
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"access_token": "mock-token", "expires_in": 3600}
        client.post.return_value = resp

        # Clear token cache
        import agent.tools.airflow_tools as aft

        aft._TOKEN_CACHE = None

        tok = _get_auth_token()
        assert tok == "mock-token"

        # Cached token returned on second call
        tok_cached = _get_auth_token()
        assert tok_cached == "mock-token"

        # Test _airflow_request with token
        resp_req = MagicMock()
        resp_req.status_code = 200
        resp_req.headers = {"content-type": "application/json"}
        resp_req.json.return_value = {"ok": True}
        client.request.return_value = resp_req

        code, body = _airflow_request("GET", "dags")
        assert code == 200
        assert body == {"ok": True}
