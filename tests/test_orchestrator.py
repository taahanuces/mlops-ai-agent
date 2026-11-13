"""Tests for LLM tool selection and multi-turn orchestrator execution."""

from __future__ import annotations

import json
from unittest.mock import patch

from agent.llm import LLMResponse, ToolCall
from agent.orchestrator import Orchestrator
from agent.schemas import ToolResult


def test_tool_selection_best_model(fake_llm):
    """Verify tool selection for best model request executes mlflow_find_best_run."""
    fake_llm.set_responses(
        [
            LLMResponse(
                tool_calls=[ToolCall(id="c1", name="mlflow_find_best_run", input={"metric": "f1"})]
            ),
            LLMResponse(
                content=json.dumps(
                    {
                        "summary": "Best model identified.",
                        "known_facts": [
                            {"statement": "Best run found", "evidence_ids": ["ev-0001"]}
                        ],
                        "uncertainties": [],
                        "recommended_actions": [],
                        "pending_confirmations": [],
                    }
                )
            ),
        ]
    )

    with patch("agent.orchestrator.run_tool") as mock_tool:
        mock_tool.return_value = ToolResult(
            evidence_id="ev-0001",
            tool="mlflow_find_best_run",
            source="mlflow",
            status="ok",
            data={"best_run": {"run_id": "run-best-7"}},
        )
        orch = Orchestrator(fake_llm)
        ans, evidence = orch.run("Which is the best model?")

        assert len(evidence) == 1
        assert evidence[0].tool == "mlflow_find_best_run"
        assert len(ans.known_facts) == 1
        assert ans.known_facts[0].evidence_ids == ["ev-0001"]


def test_tool_selection_pod_logs(fake_llm):
    """Verify tool selection for pod logs selects and executes k8s_get_pod_logs."""
    fake_llm.set_responses(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="c2",
                        name="k8s_get_pod_logs",
                        input={"pod_name": "mlops-pod-1", "tail_lines": 50},
                    )
                ]
            ),
            LLMResponse(
                content=json.dumps(
                    {
                        "summary": "Logs retrieved.",
                        "known_facts": [
                            {"statement": "Pod logs inspected", "evidence_ids": ["ev-0001"]}
                        ],
                        "uncertainties": [],
                        "recommended_actions": [],
                        "pending_confirmations": [],
                    }
                )
            ),
        ]
    )

    with patch("agent.orchestrator.run_tool") as mock_tool:
        mock_tool.return_value = ToolResult(
            evidence_id="ev-0001",
            tool="k8s_get_pod_logs",
            source="kubernetes",
            status="ok",
            data={"logs": "Starting inference service... ready"},
        )
        orch = Orchestrator(fake_llm)
        ans, evidence = orch.run("Show me logs for pod mlops-pod-1")

        assert len(evidence) == 1
        assert evidence[0].tool == "k8s_get_pod_logs"
        assert ans.known_facts[0].statement == "Pod logs inspected"


def test_full_tool_registry_loaded_on_import():
    """Regression: the live API must expose all tools without manual module imports."""
    import subprocess
    import sys

    code = (
        "import agent.orchestrator as o; "
        "from agent.tools.registry import TOOL_REGISTRY; "
        "print(len(TOOL_REGISTRY))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip().splitlines()[-1] == "26", out.stderr


def test_correlate_system_context_full_flow():
    from agent.orchestrator import correlate_system_context
    from agent.schemas import ToolResult

    # Case 1: insufficient evidence (< 2 domains) -> returns None
    ev1 = ToolResult(
        evidence_id="ev-1",
        tool="mlflow_find_best_run",
        source="mlflow",
        status="ok",
        data={"best_run": {"run_id": "r1"}},
    )
    assert correlate_system_context([ev1]) is None

    # Case 2: two domains with match
    ev2 = ToolResult(
        evidence_id="ev-2",
        tool="k8s_get_deployed_version",
        source="kubernetes",
        status="ok",
        data={"configmap": {"MODEL_RUN_ID": "r1", "DATASET_VERSION": "data-v2.0"}},
    )
    ev3 = ToolResult(
        evidence_id="ev-3",
        tool="dvc_get_dataset_version",
        source="dvc",
        status="ok",
        data={"git_tag": "data-v2.0"},
    )
    ev4 = ToolResult(
        evidence_id="ev-4",
        tool="airflow_get_dag_runs",
        source="airflow",
        status="ok",
        data={"runs": [{"state": "success"}]},
    )

    corr = correlate_system_context([ev1, ev2, ev3, ev4])
    assert corr is not None
    assert corr.status == "ok"
    assert corr.data["is_deployed_best"] is True
    assert corr.data["dataset_match"] is True
    assert corr.data["latest_dag_state"] == "success"
