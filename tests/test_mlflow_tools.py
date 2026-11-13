"""Tests for MLflow intelligence tools and ranking rules."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.tools.mlflow_tools import (
    mlflow_compare_runs,
    mlflow_find_best_run,
)
from agent.tools.registry import run_tool


def test_invalid_metric_name():
    """Verify tool returns invalid_request when metric is not found in runs."""
    mock_run = MagicMock()
    mock_run.data.metrics = {"accuracy": 0.9}

    mock_exp = MagicMock()
    mock_exp.experiment_id = "0"

    with patch("agent.tools.mlflow_tools._get_mlflow_client") as mock_client:
        client_inst = MagicMock()
        client_inst.search_experiments.return_value = [mock_exp]
        client_inst.search_runs.return_value = [mock_run]
        mock_client.return_value = client_inst

        res = mlflow_find_best_run(metric="non_existent_metric")
        assert res.status == "invalid_request"
        assert "not found" in res.error


def test_best_run_excludes_failed_gate_and_dirty():
    """Ensure runs with quality_gate=failed or dvc_dirty=true are excluded from best candidate."""
    mock_exp = MagicMock()
    mock_exp.experiment_id = "0"

    # Run 1: High F1 (0.95), but gate failed
    r1 = MagicMock()
    r1.info.run_id = "run-1"
    r1.data.metrics = {"f1": 0.95, "train_f1": 0.96}
    r1.data.tags = {"quality_gate": "failed", "dvc_dirty": "false", "run_number": "1"}

    # Run 2: High F1 (0.94), but dirty data
    r2 = MagicMock()
    r2.info.run_id = "run-2"
    r2.data.metrics = {"f1": 0.94, "train_f1": 0.95}
    r2.data.tags = {"quality_gate": "passed", "dvc_dirty": "true", "run_number": "2"}

    # Run 3: Clean, passed gate, F1 0.91 (defensible pick)
    r3 = MagicMock()
    r3.info.run_id = "run-3"
    r3.data.metrics = {"f1": 0.91, "train_f1": 0.92}
    r3.data.tags = {"quality_gate": "passed", "dvc_dirty": "false", "run_number": "3"}

    with patch("agent.tools.mlflow_tools._get_mlflow_client") as mock_client:
        client_inst = MagicMock()
        client_inst.search_experiments.return_value = [mock_exp]
        client_inst.search_runs.return_value = [r1, r2, r3]
        mock_client.return_value = client_inst

        res = mlflow_find_best_run(metric="f1")
        assert res.status == "ok"
        best = res.data["best_run"]
        assert best["run_id"] == "run-3"
        assert len(res.data["excluded_runs"]) == 2
        assert len(res.data["caveats"]) >= 2


def test_mlflow_timeout():
    """Verify tool handles slow or hanging MLflow client with timeout."""
    with patch(
        "agent.tools.mlflow_tools._get_mlflow_client",
        side_effect=TimeoutError("MLflow connection timed out"),
    ):
        res = run_tool("mlflow_list_experiments", {})
        assert res.status == "timeout"
        assert "timed out" in res.error.lower()


def test_mlflow_compare_runs():
    """Verify side-by-side run comparison."""
    r1 = MagicMock()
    r1.data.metrics = {"f1": 0.90}
    r1.data.params = {"n_estimators": 50}
    r1.data.tags = {"dvc_dataset_version": "data-v1.0", "quality_gate": "passed"}

    with patch("agent.tools.mlflow_tools._get_mlflow_client") as mock_client:
        client_inst = MagicMock()
        client_inst.get_run.return_value = r1
        mock_client.return_value = client_inst

        res = mlflow_compare_runs(run_ids=["run-1", "run-2"])
        assert res.status == "ok"
        assert len(res.data["comparison"]) == 2
