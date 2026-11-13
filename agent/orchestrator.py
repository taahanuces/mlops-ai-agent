"""Agent orchestrator implementing tool loop, evidence guard, and retraining policy guard."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from agent.config import (
    RETRAIN_ON_NEW_DATASET,
)
from agent.llm import LLMClient
from agent.logging_utils import get_current_request_id, log_request_summary
from agent.schemas import Fact, FinalAnswer, ToolResult
from agent.tools.registry import (
    get_all_tool_specs,
    next_evidence_id,
    run_tool,
)

SYSTEM_PROMPT = """You are an expert MLOps AI Assistant managing a production ML platform with DVC, MLflow, Airflow 3, and Kubernetes.
You answer user questions and perform operational tasks using your tools.

Core Operating Principles:
1. Always ground your answers strictly in tool evidence. Never state an operational fact without citing the corresponding evidence_id.
2. If a tool fails, times out, or reports unavailable, explicitly state that in uncertainties. Do not guess, fill gaps, or invent data.
3. Highest F1 score alone does NOT mean a model is production-ready. You must inspect all evaluation metrics, quality_gate tags, dataset cleanliness (dvc_dirty), and current deployment state.
4. Retraining Policy: Do NOT trigger or recommend retraining unless there is concrete evidence under policy:
   - Model age > MAX_MODEL_AGE_DAYS
   - Quality gate failed in recent DAG runs or experiments
   - Explicit user-provided evidence of concept or data drift.
   A new dataset version alone is reported as a fact; it is a retraining reason only if RETRAIN_ON_NEW_DATASET is true.
   For general queries or the mandatory scenario, NEVER trigger training.
5. All write actions (restarting pods, scaling deployments, triggering DAGs) require confirmation and will return pending_confirmation.
6. Format your final response strictly as a JSON object adhering to this schema:
{
  "summary": "Short 2-3 sentence overview.",
  "known_facts": [
    {"statement": "...", "evidence_ids": ["ev-0001"]}
  ],
  "uncertainties": ["..."],
  "recommended_actions": ["..."],
  "pending_confirmations": ["conf-..."]
}
Only output the raw JSON object for the final answer.
"""


def correlate_system_context(evidence_list: list[ToolResult]) -> ToolResult | None:
    """Pure-Python correlation helper synthesizing findings across DVC, MLflow, K8s, and Airflow."""
    mlflow_best = None
    deployed_version = None
    latest_dataset_tag = None
    latest_dag_run = None

    for ev in evidence_list:
        if ev.status != "ok" or not isinstance(ev.data, dict):
            continue
        if ev.tool == "mlflow_find_best_run":
            mlflow_best = ev.data.get("best_run")
        elif ev.tool == "k8s_get_deployed_version":
            deployed_version = ev.data
        elif ev.tool == "dvc_get_dataset_version":
            latest_dataset_tag = ev.data.get("git_tag")
        elif ev.tool == "airflow_get_dag_runs":
            runs = ev.data.get("runs", [])
            if runs:
                latest_dag_run = runs[0]

    # Only run correlation if at least two domains returned data
    valid_count = sum(
        1 for x in [mlflow_best, deployed_version, latest_dataset_tag, latest_dag_run] if x
    )
    if valid_count < 2:
        return None

    correlation_facts = []
    deployed_run_id = (
        deployed_version.get("configmap", {}).get("MODEL_RUN_ID") if deployed_version else None
    )
    deployed_data_ver = (
        deployed_version.get("configmap", {}).get("DATASET_VERSION") if deployed_version else None
    )

    is_deployed_best = False
    if mlflow_best and deployed_run_id:
        is_deployed_best = mlflow_best.get("run_id") == deployed_run_id
        correlation_facts.append(
            f"Deployed model run ({deployed_run_id}) matches best evaluated MLflow candidate: {is_deployed_best}"
        )

    dataset_match = None
    if latest_dataset_tag and deployed_data_ver:
        dataset_match = latest_dataset_tag == deployed_data_ver
        correlation_facts.append(
            f"Deployed dataset version ({deployed_data_ver}) matches repository latest tag ({latest_dataset_tag}): {dataset_match}"
        )

    ev_id = next_evidence_id()
    req_id = get_current_request_id()
    return ToolResult(
        evidence_id=ev_id,
        tool="correlate_system_context",
        source="agent",
        status="ok",
        data={
            "is_deployed_best": is_deployed_best,
            "dataset_match": dataset_match,
            "deployed_run_id": deployed_run_id,
            "best_run_id": mlflow_best.get("run_id") if mlflow_best else None,
            "latest_data_tag": latest_dataset_tag,
            "latest_dag_state": latest_dag_run.get("state") if latest_dag_run else None,
            "correlation_summary": correlation_facts,
        },
        duration_ms=1,
        request_id=req_id,
    )


class Orchestrator:
    """Manages LLM multi-turn tool interaction, evidence validation, and safety policies."""

    def __init__(self, llm_client: LLMClient, max_turns: int = 8, timeout_s: int = 90):
        self.llm_client = llm_client
        self.max_turns = max_turns
        self.timeout_s = timeout_s

    def run(
        self, user_prompt: str, session_id: str | None = None
    ) -> tuple[FinalAnswer, list[ToolResult]]:
        start_time = time.time()
        req_id = get_current_request_id()
        evidence_list: list[ToolResult] = []
        pending_action_ids: list[str] = []

        # Tool specifications for the LLM
        tool_specs = [
            {
                "name": s.name,
                "description": s.description,
                "input_schema": s.input_schema,
            }
            for s in get_all_tool_specs()
        ]

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]

        final_raw_content = ""
        tools_called_names: list[str] = []

        for turn in range(self.max_turns):
            if time.time() - start_time > self.timeout_s:
                break

            response = self.llm_client.complete(
                messages=messages,
                tools=tool_specs,
                system=SYSTEM_PROMPT,
            )

            # Check if LLM requested tool invocations
            if not response.tool_calls:
                final_raw_content = response.content
                break

            # Execute tool calls
            tool_results_content = []
            for tc in response.tool_calls:
                tools_called_names.append(tc.name)
                # Safeguard against accidental DAG triggers in general/mandatory requests
                if "best" in user_prompt.lower() and "retraining" in user_prompt.lower():
                    if tc.name == "airflow_trigger_training_dag":
                        ev_id = next_evidence_id()
                        res = ToolResult(
                            evidence_id=ev_id,
                            tool=tc.name,
                            source="airflow",
                            status="denied",
                            error="Retraining trigger denied under inspection policy: read-only evaluation query.",
                            duration_ms=0,
                            request_id=req_id,
                        )
                        evidence_list.append(res)
                        tool_results_content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tc.id,
                                "content": json.dumps(res.model_dump()),
                            }
                        )
                        continue

                res = run_tool(tc.name, tc.input, req_id=req_id)
                evidence_list.append(res)
                if res.status == "pending_confirmation" and isinstance(res.data, dict):
                    cid = res.data.get("confirmation_id")
                    if cid:
                        pending_action_ids.append(cid)

                tool_results_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": json.dumps(res.model_dump()),
                    }
                )

            # Append assistant turn and tool results to messages
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
                        for tc in response.tool_calls
                    ],
                }
            )
            messages.append({"role": "user", "content": tool_results_content})

        # Check and add correlation evidence if applicable
        correlation_ev = correlate_system_context(evidence_list)
        if correlation_ev:
            evidence_list.append(correlation_ev)

        # Parse final answer
        final_answer = self._parse_and_validate_answer(
            final_raw_content, evidence_list, pending_action_ids, user_prompt
        )

        total_ms = int((time.time() - start_time) * 1000)
        log_request_summary(len(user_prompt), tools_called_names, total_ms, "success", req_id)
        return final_answer, evidence_list

    def _parse_and_validate_answer(
        self,
        raw_text: str,
        evidence_list: list[ToolResult],
        pending_actions: list[str],
        user_prompt: str,
    ) -> FinalAnswer:
        """Parse LLM JSON, enforce evidence guard, and apply retraining policy guard."""
        valid_ev_ids = {ev.evidence_id for ev in evidence_list if ev.status == "ok"}
        failed_tools = [ev for ev in evidence_list if ev.status != "ok"]

        data: dict[str, Any] = {}
        # Attempt extracting JSON block
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
            except Exception:
                pass

        if not data:
            data = {
                "summary": raw_text.strip() or "Analysis complete based on available evidence.",
                "known_facts": [],
                "uncertainties": [],
                "recommended_actions": [],
                "pending_confirmations": pending_actions,
            }

        # Validate facts via Evidence Guard
        checked_facts: list[Fact] = []
        uncertainties = list(data.get("uncertainties", []))

        for f_raw in data.get("known_facts", []):
            if isinstance(f_raw, dict):
                stmt = f_raw.get("statement", "")
                e_ids = f_raw.get("evidence_ids", [])
                valid_ids = [eid for eid in e_ids if eid in valid_ev_ids]

                if valid_ids:
                    checked_facts.append(Fact(statement=stmt, evidence_ids=valid_ids))
                else:
                    uncertainties.append(f"Unverified claim (no valid tool evidence): {stmt}")

        # Inform of failed tools in uncertainties
        for ft in failed_tools:
            msg = f"Tool '{ft.tool}' returned status '{ft.status}': {ft.error or 'Service down'}"
            if not any(ft.tool in u for u in uncertainties):
                uncertainties.append(msg)

        # Retraining policy guard
        recommended_actions = list(data.get("recommended_actions", []))
        if "retraining" in user_prompt.lower() or "retrain" in user_prompt.lower():
            # Check if retraining conditions are strictly met
            has_failed_gate = any(
                isinstance(ev.data, dict)
                and (
                    ev.data.get("quality_gate") == "failed"
                    or (
                        isinstance(ev.data.get("best_run"), dict)
                        and ev.data["best_run"].get("quality_gate") == "failed"
                    )
                )
                for ev in evidence_list
            )
            data_mismatch = any(
                isinstance(ev.data, dict) and ev.data.get("dataset_match") is False
                for ev in evidence_list
            )
            should_retrain = has_failed_gate or (data_mismatch and RETRAIN_ON_NEW_DATASET)

            if not should_retrain:
                # Ensure no false retraining recommendation
                recommended_actions = [a for a in recommended_actions if "retrain" not in a.lower()]
                recommended_actions.append(
                    "No evidence that retraining is currently required under production policy."
                )

        # If all tools failed, strip all numbers from summary and facts
        all_failed = len(evidence_list) > 0 and len(valid_ev_ids) == 0
        summary = data.get("summary", "")
        if all_failed:
            checked_facts = []
            summary = "Target external systems were unavailable or returned errors. No operational metrics could be retrieved."

        return FinalAnswer(
            summary=summary,
            known_facts=checked_facts,
            uncertainties=uncertainties,
            recommended_actions=recommended_actions,
            pending_confirmations=list(
                set(pending_actions + data.get("pending_confirmations", []))
            ),
        )
