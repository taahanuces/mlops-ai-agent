"""Safety, confirmation single-use, evidence guard, and retraining policy tests."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

from agent.llm import LLMResponse, ToolCall
from agent.orchestrator import Orchestrator
from agent.permissions import ConfirmationStore
from agent.schemas import ToolResult


def test_confirmation_single_use_and_expiry():
    """Verify write confirmation is single-use and expires after TTL."""
    store = ConfirmationStore(ttl_seconds=1)
    executed = False

    def dummy_action():
        nonlocal executed
        executed = True
        return "success"

    cid = store.store_action(
        "k8s_restart_deployment", {}, "Restart pod", "kubernetes", dummy_action
    )

    # First execution succeeds
    action, res, err = store.execute_and_consume(cid)
    assert action is not None
    assert res == "success"
    assert executed is True

    # Second execution fails (single use)
    action2, res2, err2 = store.execute_and_consume(cid)
    assert action2 is None
    assert "not found or already consumed" in err2.lower()

    # Test expiry
    cid_exp = store.store_action("test_tool", {}, "Expire test", "agent", dummy_action)
    time.sleep(1.1)
    act_exp, res_exp, err_exp = store.execute_and_consume(cid_exp)
    assert act_exp is None
    assert "expired" in err_exp.lower()


def test_evidence_guard_rejects_uncited_fact(fake_llm):
    """Verify that facts citing nonexistent or failed evidence are moved to uncertainties."""
    # LLM tries to fabricate a fact citing non-existent 'ev-9999'
    fake_llm.set_responses(
        [
            LLMResponse(
                content=json.dumps(
                    {
                        "summary": "Fabricated summary",
                        "known_facts": [
                            {"statement": "Model F1 score is 0.999", "evidence_ids": ["ev-9999"]}
                        ],
                        "uncertainties": [],
                        "recommended_actions": [],
                        "pending_confirmations": [],
                    }
                ),
                tool_calls=[],
            )
        ]
    )

    orch = Orchestrator(fake_llm)
    ans, ev = orch.run("What is the model accuracy?")
    assert len(ans.known_facts) == 0
    assert any("Unverified claim" in u for u in ans.uncertainties)


def test_agent_does_not_invent_results_when_tool_down(fake_llm):
    """Verify no numbers or invented facts are present when tools fail."""
    fake_llm.set_responses(
        [
            # Turn 1: call tool that fails
            LLMResponse(tool_calls=[ToolCall(id="tc-1", name="mlflow_list_experiments", input={})]),
            # Turn 2: Attempt answer
            LLMResponse(
                content=json.dumps(
                    {
                        "summary": "There are 5 experiments with 25 runs.",
                        "known_facts": [
                            {"statement": "5 experiments found", "evidence_ids": ["ev-0001"]}
                        ],
                        "uncertainties": [],
                        "recommended_actions": [],
                        "pending_confirmations": [],
                    }
                )
            ),
        ]
    )

    with patch("agent.orchestrator.run_tool") as mock_run:
        mock_run.return_value = ToolResult(
            evidence_id="ev-0001",
            tool="mlflow_list_experiments",
            source="mlflow",
            status="unavailable",
            error="Connection refused",
        )
        orch = Orchestrator(fake_llm)
        ans, ev = orch.run("How many experiments are in MLflow?")

        assert len(ans.known_facts) == 0
        assert "unavailable" in ans.summary.lower()
        assert any("mlflow_list_experiments" in u for u in ans.uncertainties)


def test_retraining_not_recommended_without_evidence(fake_llm):
    """Verify agent rejects retraining recommendation in healthy state without policy evidence."""
    fake_llm.set_responses(
        [
            LLMResponse(
                content=json.dumps(
                    {
                        "summary": "System is healthy.",
                        "known_facts": [],
                        "uncertainties": [],
                        "recommended_actions": ["Trigger retraining immediately"],
                        "pending_confirmations": [],
                    }
                )
            )
        ]
    )

    orch = Orchestrator(fake_llm)
    ans, ev = orch.run("Check system health and explain whether retraining is necessary.")
    assert not any("trigger retraining immediately" in a.lower() for a in ans.recommended_actions)
    assert any(
        "No evidence that retraining is currently required" in a for a in ans.recommended_actions
    )


def test_retraining_recommended_on_failed_gate(fake_llm):
    """Verify retraining is recommended (not triggered) when evidence shows quality_gate=failed."""
    fake_llm.set_responses(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="tc-1", name="mlflow_find_best_run", input={"metric": "f1"})
                ]
            ),
            LLMResponse(
                content=json.dumps(
                    {
                        "summary": "Quality gate failed.",
                        "known_facts": [],
                        "uncertainties": [],
                        "recommended_actions": ["Recommend retraining due to failed quality gate"],
                        "pending_confirmations": [],
                    }
                )
            ),
        ]
    )

    with patch("agent.orchestrator.run_tool") as mock_run:
        mock_run.return_value = ToolResult(
            evidence_id="ev-0001",
            tool="mlflow_find_best_run",
            source="mlflow",
            status="ok",
            data={"quality_gate": "failed", "best_run": {"quality_gate": "failed"}},
        )
        orch = Orchestrator(fake_llm)
        ans, ev = orch.run("Is retraining needed?")
        assert any("retraining" in a.lower() for a in ans.recommended_actions)


def test_multi_tool_mandatory_scenario(fake_llm):
    """Ensure mandatory scenario calls all systems but never triggers airflow DAG."""
    called_tools = []

    def mock_tool_runner(name, args, req_id=None):
        called_tools.append(name)
        return ToolResult(
            evidence_id=f"ev-{len(called_tools)}",
            tool=name,
            source="agent",
            status="ok",
            data={"status": "ok"},
        )

    # Set up simulated multi-turn tool calling
    fake_llm.set_responses(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="c1", name="mlflow_find_best_run", input={}),
                    ToolCall(id="c2", name="dvc_get_dataset_version", input={}),
                    ToolCall(id="c3", name="k8s_get_deployed_version", input={}),
                    ToolCall(id="c4", name="airflow_get_dag_runs", input={}),
                ]
            ),
            LLMResponse(
                content=json.dumps(
                    {
                        "summary": "All systems healthy.",
                        "known_facts": [
                            {"statement": "Best model evaluated", "evidence_ids": ["ev-1"]}
                        ],
                        "uncertainties": [],
                        "recommended_actions": [],
                        "pending_confirmations": [],
                    }
                )
            ),
        ]
    )

    with patch("agent.orchestrator.run_tool", side_effect=mock_tool_runner):
        orch = Orchestrator(fake_llm)
        ans, ev = orch.run(
            "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
        )

        assert "airflow_trigger_training_dag" not in called_tools
        assert "mlflow_find_best_run" in called_tools
        assert "dvc_get_dataset_version" in called_tools
        assert "k8s_get_deployed_version" in called_tools
        assert "airflow_get_dag_runs" in called_tools


def test_confirmation_store_get_action_and_list_pending():
    store = ConfirmationStore(ttl_seconds=1)
    cid = store.store_action("t", {}, "desc", "agent", lambda: 1)
    act = store.get_action(cid)
    assert act is not None
    assert act.confirmation_id == cid

    pending = store.list_pending()
    assert len(pending) == 1

    time.sleep(1.1)
    assert store.get_action(cid) is None
    assert len(store.list_pending()) == 0


def test_confirmation_store_callback_exception():
    store = ConfirmationStore(ttl_seconds=10)

    def bad_callback():
        raise ValueError("failing action")

    cid = store.store_action("t", {}, "desc", "agent", bad_callback)
    act, res, err = store.execute_and_consume(cid)
    assert res is None
    assert "Execution failed: failing action" in err


def test_permission_checks():
    from agent.permissions import (
        check_airflow_dag_id,
        check_dvc_command_allowed,
        check_kubernetes_namespace,
    )

    assert check_kubernetes_namespace("mlops-agent") is True
    assert check_kubernetes_namespace("kube-system") is False

    assert check_airflow_dag_id("training_pipeline") is True
    assert check_airflow_dag_id("non_existent_dag") is False

    assert check_dvc_command_allowed([]) is False
    assert check_dvc_command_allowed("not a list") is False  # type: ignore
