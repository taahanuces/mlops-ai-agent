"""Tests for Airflow 3 API tools, JWT handling, and trigger gating."""

from __future__ import annotations

from unittest.mock import patch

from agent.tools.airflow_tools import (
    airflow_get_failed_tasks,
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


def test_airflow_trigger_requires_confirmation():
    """Verify triggering an allow-listed DAG returns pending_confirmation."""
    res = airflow_trigger_training_dag(dag_id="training_pipeline")
    assert res.status == "pending_confirmation"
    assert "confirmation_id" in res.data
    assert res.data["confirmation_id"].startswith("conf-")


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
