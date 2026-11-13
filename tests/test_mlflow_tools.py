"""Tests for MLflow intelligence tools and ranking rules."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.tools.mlflow_tools import (
    mlflow_compare_runs,
    mlflow_find_best_run,
    mlflow_get_model_versions,
    mlflow_get_run_details,
    mlflow_list_experiments,
    mlflow_list_runs,
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
    r1.info.run_id = "run-1"
    r1.data.metrics = {"f1": 0.90}
    r1.data.params = {"n_estimators": 50}
    r1.data.tags = {"quality_gate": "passed"}

    r2 = MagicMock()
    r2.info.run_id = "run-2"
    r2.data.metrics = {"f1": 0.92}
    r2.data.params = {"n_estimators": 100}
    r2.data.tags = {"quality_gate": "passed"}

    with patch("agent.tools.mlflow_tools._get_mlflow_client") as mock_client:
        client_inst = MagicMock()
        client_inst.get_run.side_effect = lambda rid: r1 if rid == "run-1" else r2
        mock_client.return_value = client_inst

        res = mlflow_compare_runs(run_ids=["run-1", "run-2"])
        assert res.status == "ok"
        assert len(res.data["comparison"]) == 2


def test_mlflow_list_experiments():
    mock_exp = MagicMock()
    mock_exp.experiment_id = "1"
    mock_exp.name = "production-training"
    mock_exp.artifact_location = "s3://bucket"
    mock_exp.lifecycle_stage = "active"

    with patch("agent.tools.mlflow_tools._get_mlflow_client") as mock_client:
        client = MagicMock()
        client.search_experiments.return_value = [mock_exp]
        mock_client.return_value = client

        res = mlflow_list_experiments()
        assert res.status == "ok"
        assert len(res.data["experiments"]) == 1
        assert res.data["experiments"][0]["name"] == "production-training"


def test_mlflow_list_runs():
    mock_exp = MagicMock(experiment_id="1")
    mock_run = MagicMock()
    mock_run.info.run_id = "r-1"
    mock_run.info.run_name = "test-run"
    mock_run.info.start_time = 12345
    mock_run.info.status = "FINISHED"
    mock_run.data.metrics = {"f1": 0.91}
    mock_run.data.params = {"n_est": "10"}
    mock_run.data.tags = {"dvc_dataset_version": "data-v2.0"}

    with patch("agent.tools.mlflow_tools._get_mlflow_client") as mock_client:
        client = MagicMock()
        client.search_experiments.return_value = [mock_exp]
        client.search_runs.return_value = [mock_run]
        mock_client.return_value = client

        res = mlflow_list_runs(experiment_id=None, limit=5)
        assert res.status == "ok"
        assert len(res.data["runs"]) == 1
        assert res.data["runs"][0]["run_id"] == "r-1"


def test_mlflow_get_run_details():
    mock_run = MagicMock()
    mock_run.info.run_id = "r-target"
    mock_run.info.status = "FINISHED"
    mock_run.info.start_time = 100
    mock_run.info.end_time = 200
    mock_run.data.metrics = {"f1": 0.93}
    mock_run.data.params = {}
    mock_run.data.tags = {"run_number": "7"}

    mock_art = MagicMock()
    mock_art.path = "model/model.pkl"

    with patch("agent.tools.mlflow_tools._get_mlflow_client") as mock_client:
        client = MagicMock()
        client.get_run.return_value = mock_run
        client.list_artifacts.return_value = [mock_art]
        client.search_experiments.return_value = [MagicMock(experiment_id="1")]
        client.search_runs.return_value = [mock_run]
        mock_client.return_value = client

        # by run_id
        res_by_id = mlflow_get_run_details(run_id="r-target")
        assert res_by_id.status == "ok"
        assert res_by_id.data["run_id"] == "r-target"
        assert res_by_id.data["artifacts"] == ["model/model.pkl"]

        # by run_number
        res_by_num = mlflow_get_run_details(run_number=7)
        assert res_by_num.status == "ok"
        assert res_by_num.data["run_id"] == "r-target"

        # not found
        client.get_run.return_value = None
        client.search_runs.return_value = []
        res_not_found = mlflow_get_run_details(run_number=9999)
        assert res_not_found.status == "invalid_request"


def test_mlflow_get_model_versions():
    mock_v1 = MagicMock(
        version="1", current_stage="Production", aliases=["champion"], run_id="r1", status="READY"
    )
    mock_v2 = MagicMock(
        version="2", current_stage="Staging", aliases=["candidate"], run_id="r2", status="READY"
    )

    with patch("agent.tools.mlflow_tools._get_mlflow_client") as mock_client:
        client = MagicMock()
        client.search_model_versions.return_value = [mock_v1, mock_v2]
        mock_client.return_value = client

        res = mlflow_get_model_versions(name="classification-model")
        assert res.status == "ok"
        assert len(res.data["versions"]) == 2
        assert res.data["versions"][0]["aliases"] == ["champion"]
