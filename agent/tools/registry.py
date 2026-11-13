"""Tool registry, runner wrapper, and tool metadata definitions."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Callable, Type

from pydantic import BaseModel, ValidationError

from agent.logging_utils import get_current_request_id, log_tool_call
from agent.schemas import ToolResult, ToolSpec

# Registry mapping: name -> (func, input_model, spec)
TOOL_REGISTRY: dict[str, tuple[Callable[..., Any], Type[BaseModel] | None, ToolSpec]] = {}

_evidence_counter = 0


def next_evidence_id() -> str:
    """Generate monotonically increasing evidence ID."""
    global _evidence_counter
    _evidence_counter += 1
    return f"ev-{_evidence_counter:04d}"


def reset_evidence_counter() -> None:
    """Reset evidence counter for testing."""
    global _evidence_counter
    _evidence_counter = 0


def register_tool(
    name: str,
    description: str,
    permission: str,
    input_model: Type[BaseModel] | None = None,
    timeout_s: int = 10,
    source: str = "agent",
) -> Callable:
    """Decorator to register a tool function into the registry."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        schema: dict[str, Any] = (
            input_model.model_json_schema() if input_model else {"type": "object", "properties": {}}
        )
        spec = ToolSpec(
            name=name,
            description=description,
            input_schema=schema,
            permission=permission,  # type: ignore[arg-type]
            timeout_s=timeout_s,
        )
        TOOL_REGISTRY[name] = (fn, input_model, spec)
        return fn

    return decorator


def run_tool(
    name: str,
    args: dict[str, Any] | BaseModel,
    req_id: str | None = None,
) -> ToolResult:
    """Safely execute a tool with timing, exception handling, schema validation, and logging."""
    req = req_id or get_current_request_id()
    ev_id = next_evidence_id()
    start_time = time.time()

    if name not in TOOL_REGISTRY:
        dur = int((time.time() - start_time) * 1000)
        res = ToolResult(
            evidence_id=ev_id,
            tool=name,
            source="agent",
            status="denied",
            error=f"Tool '{name}' is not registered or allowed.",
            duration_ms=dur,
            request_id=req,
        )
        log_tool_call(name, "agent", "denied", dur, ev_id, req)
        return res

    fn, input_model, spec = TOOL_REGISTRY[name]

    # Validate input model
    validated_args: dict[str, Any] = {}
    if input_model:
        try:
            if isinstance(args, BaseModel):
                validated_args = args.model_dump()
            else:
                validated_obj = input_model(**args)
                validated_args = validated_obj.model_dump()
        except ValidationError as val_err:
            dur = int((time.time() - start_time) * 1000)
            res = ToolResult(
                evidence_id=ev_id,
                tool=name,
                source="agent",
                status="invalid_request",
                error=f"Invalid arguments for tool '{name}': {val_err}",
                duration_ms=dur,
                request_id=req,
            )
            log_tool_call(name, "agent", "invalid_request", dur, ev_id, req)
            return res
    elif isinstance(args, dict):
        validated_args = args

    try:
        # Tool call execution
        result = fn(**validated_args)
        dur = int((time.time() - start_time) * 1000)

        if isinstance(result, ToolResult):
            # Already formatted ToolResult from tool
            result.evidence_id = ev_id
            result.request_id = req
            result.duration_ms = dur
            log_tool_call(name, result.source, result.status, dur, ev_id, req)
            return result

        res = ToolResult(
            evidence_id=ev_id,
            tool=name,
            source="agent",
            status="ok",
            data=result,
            duration_ms=dur,
            request_id=req,
        )
        log_tool_call(name, "agent", "ok", dur, ev_id, req)
        return res

    except TimeoutError as te:
        dur = int((time.time() - start_time) * 1000)
        res = ToolResult(
            evidence_id=ev_id,
            tool=name,
            source="agent",
            status="timeout",
            error=f"Tool execution timed out: {te}",
            duration_ms=dur,
            request_id=req,
        )
        log_tool_call(name, "agent", "timeout", dur, ev_id, req)
        return res
    except Exception as exc:
        dur = int((time.time() - start_time) * 1000)
        res = ToolResult(
            evidence_id=ev_id,
            tool=name,
            source="agent",
            status="error",
            error=f"Internal error executing {name}: {str(exc)}",
            duration_ms=dur,
            request_id=req,
        )
        log_tool_call(name, "agent", "error", dur, ev_id, req)
        return res


def get_all_tool_specs() -> list[ToolSpec]:
    """Return specs of all registered tools."""
    return [spec for _, _, spec in TOOL_REGISTRY.values()]


def run_smoke() -> dict[str, Any]:
    """Run every registered tool once against the real systems (writes only stage confirmation)."""
    import json
    from pathlib import Path

    # Use the canonical module so decorators and this runner share one registry
    import agent.tools.airflow_tools  # noqa: F401
    import agent.tools.dvc_tools  # noqa: F401
    import agent.tools.kubernetes_tools  # noqa: F401
    import agent.tools.mlflow_tools  # noqa: F401
    from agent.tools import registry as reg

    champion = Path("reports/evidence/champion_run_id.txt").read_text().strip()
    af_run = json.loads(Path("reports/evidence/airflow_failed_gate_run.json").read_text())[
        "dag_run"
    ]["dag_run_id"]

    runs = reg.run_tool("mlflow_list_runs", {"limit": 5})
    run_rows = (runs.data or {}).get("runs", []) if isinstance(runs.data, dict) else []
    others = [r.get("run_id") for r in run_rows if r.get("run_id") and r.get("run_id") != champion]
    runner_up = others[0] if others else champion

    pod_res = reg.run_tool("k8s_list_pods", {})
    pods = (pod_res.data or {}).get("pods", []) if isinstance(pod_res.data, dict) else []
    pod_name = pods[0].get("name", "unknown") if pods else "unknown"

    cases: dict[str, dict[str, Any]] = {
        "airflow_list_dags": {},
        "airflow_get_dag_runs": {"dag_id": "training_pipeline", "limit": 5},
        "airflow_get_failed_tasks": {"dag_run_id": af_run},
        "airflow_get_task_logs": {"dag_run_id": af_run, "task_id": "quality_gate"},
        "airflow_trigger_training_dag": {"dag_id": "training_pipeline"},
        "dvc_list_tracked_datasets": {},
        "dvc_get_dataset_version": {},
        "dvc_compare_versions": {"rev_a": "data-v1.0", "rev_b": "data-v2.0"},
        "dvc_check_data_changed": {},
        "dvc_get_reproduction_steps": {"rev": "data-v1.0"},
        "dvc_materialize_dataset_version": {"rev": "data-v1.0"},
        "dvc_get_dataset_for_run": {"run_id": champion},
        "k8s_list_pods": {},
        "k8s_list_deployments": {},
        "k8s_get_pod_logs": {"pod_name": pod_name, "tail_lines": 20},
        "k8s_get_services": {},
        "k8s_find_unhealthy_pods": {},
        "k8s_get_deployed_version": {},
        "k8s_restart_deployment": {},
        "k8s_scale_deployment": {"replicas": 2},
        "mlflow_list_experiments": {},
        "mlflow_list_runs": {"limit": 5},
        "mlflow_get_run_details": {"run_id": champion},
        "mlflow_find_best_run": {"metric": "f1", "mode": "max", "limit": 3},
        "mlflow_compare_runs": {"run_ids": [champion, runner_up]},
        "mlflow_get_model_versions": {},
    }
    out: dict[str, Any] = {"tools_registered": len(reg.TOOL_REGISTRY), "results": {}}
    for name, args in cases.items():
        r = reg.run_tool(name, args)
        expected = (
            "pending_confirmation" if reg.TOOL_REGISTRY[name][2].permission == "write" else "ok"
        )
        out["results"][name] = {
            "status": r.status,
            "expected": expected,
            "pass": r.status == expected,
            "evidence_id": r.evidence_id,
            "duration_ms": r.duration_ms,
            "error": r.error,
        }
    out["all_pass"] = all(v["pass"] for v in out["results"].values()) and len(cases) == len(
        reg.TOOL_REGISTRY
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool registry inspector & smoke test")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests on registered tools")
    parser.add_argument("--out", default=None, help="Write smoke JSON to this file")
    args = parser.parse_args()

    if args.smoke:
        import json

        # Re-enter via canonical module name to avoid a duplicate __main__ registry
        from agent.tools.registry import run_smoke as _run_smoke

        result = _run_smoke()
        payload = json.dumps(result, indent=2, default=str)
        if args.out:
            with open(args.out, "w") as fh:
                fh.write(payload + "\n")
        print(payload)
        sys.exit(0 if result["all_pass"] else 1)

    import agent.tools.airflow_tools  # noqa: F401
    import agent.tools.dvc_tools  # noqa: F401
    import agent.tools.kubernetes_tools  # noqa: F401
    import agent.tools.mlflow_tools  # noqa: F401
    from agent.tools import registry as _reg

    print(f"Total tools registered: {len(_reg.TOOL_REGISTRY)}")
    for name, (_, _, spec) in _reg.TOOL_REGISTRY.items():
        print(f" - {name} [{spec.permission}]: {spec.description}")
