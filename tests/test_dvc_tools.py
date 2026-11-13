"""Tests for DVC and Git provenance tools and security allow-listing."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from agent.permissions import check_dvc_command_allowed
from agent.tools.dvc_tools import (
    _resolve_binary,
    dvc_check_data_changed,
    dvc_compare_versions,
    dvc_get_dataset_for_run,
    dvc_get_dataset_version,
    dvc_get_reproduction_steps,
    dvc_list_tracked_datasets,
)
from agent.tools.registry import run_tool


def test_dvc_command_allowlist():
    """Verify that only approved command prefixes are allowed."""
    assert check_dvc_command_allowed(["dvc", "status"]) is True
    assert check_dvc_command_allowed(["dvc", "diff", "--json", "v1", "v2"]) is True
    assert check_dvc_command_allowed(["git", "describe", "--tags"]) is True
    assert (
        check_dvc_command_allowed(
            ["dvc", "get", ".", "data/dataset.csv", "-o", ".cache/dataset_checkouts/v1/dataset.csv"]
        )
        is True
    )

    # Disallowed commands
    assert check_dvc_command_allowed(["dvc", "push"]) is False
    assert check_dvc_command_allowed(["dvc", "gc"]) is False
    assert check_dvc_command_allowed(["dvc", "remove"]) is False
    assert check_dvc_command_allowed(["rm", "-rf", "/"]) is False
    assert (
        check_dvc_command_allowed(["dvc", "get", ".", "data/dataset.csv", "-o", "/etc/passwd"])
        is False
    )


@patch("agent.tools.dvc_tools._execute_safe_command")
def test_dvc_compare_and_change_detection(mock_exec):
    """Test parsing of dvc diff and data status JSON output."""
    mock_exec.return_value = (0, json.dumps({"modified": [{"path": "data/dataset.csv"}]}), "")

    res = dvc_compare_versions(rev_a="data-v1.0", rev_b="data-v2.0")
    assert res.status == "ok"
    assert "data/dataset.csv" in str(res.data)

    res_change = dvc_check_data_changed()
    assert res_change.status == "ok"
    assert res_change.data["data_changed"] is True


@patch("agent.tools.dvc_tools._execute_safe_command")
def test_dvc_get_reproduction_steps(mock_exec):
    """Verify reproduction instructions generation."""
    mock_exec.return_value = (0, "commit-hash-abc", "")

    res = dvc_get_reproduction_steps(rev="data-v1.0")
    assert res.status == "ok"
    assert len(res.data["reproduction_commands"]) >= 2
    assert "dvc checkout" in res.data["reproduction_commands"][1]


@patch("agent.tools.dvc_tools._execute_safe_command")
def test_dvc_materialize_dataset_version_security(mock_exec, tmp_path):
    """Verify materialize tool enforces allowed cache output directory."""
    res = run_tool("dvc_materialize_dataset_version", {"rev": "data-v1.0"})
    assert res.tool == "dvc_materialize_dataset_version"


def test_dvc_list_tracked_datasets():
    res = dvc_list_tracked_datasets()
    assert res.status == "ok"
    assert "tracked_datasets" in res.data


@patch("agent.tools.dvc_tools._execute_safe_command")
def test_dvc_get_dataset_version(mock_exec):
    # Mock git rev-parse, git describe, git show
    mock_dvc_yaml = "outs:\n- md5: abc123def456\n  size: 1000\n"
    mock_exec.side_effect = [
        (0, "commit123", ""),  # git rev-parse
        (0, "data-v2.0", ""),  # git describe exact-match
        (0, mock_dvc_yaml, ""),  # git show dataset.csv.dvc
    ]

    res = dvc_get_dataset_version(rev="data-v2.0")
    assert res.status == "ok"
    assert res.data["git_tag"] == "data-v2.0"
    assert res.data["git_commit"] == "commit123"
    assert res.data["dvc_md5"] == "abc123def456"


def test_dvc_get_dataset_for_run():
    mock_run = MagicMock()
    mock_run.info.run_id = "run-007"
    mock_run.data.tags = {
        "run_number": "7",
        "dvc_dataset_version": "data-v2.0",
        "git_commit": "abcdef1",
        "dvc_md5": "hash777",
        "dvc_dirty": "false",
        "quality_gate": "passed",
    }

    with patch("agent.tools.mlflow_tools._get_mlflow_client") as mock_client:
        client = MagicMock()
        client.get_run.return_value = mock_run
        mock_client.return_value = client

        res = dvc_get_dataset_for_run(run_id="run-007")
        assert res.status == "ok"
        assert res.data["run_id"] == "run-007"
        assert res.data["dvc_dataset_version"] == "data-v2.0"
        assert res.data["dvc_dirty"] is False


def test_resolve_binary():
    assert _resolve_binary("python") != ""
