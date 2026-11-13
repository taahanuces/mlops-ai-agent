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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool registry inspector & smoke test")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests on registered tools")
    args = parser.parse_args()

    # Import tool modules to populate registry
    try:
        import agent.tools.airflow_tools  # noqa: F401
        import agent.tools.dvc_tools  # noqa: F401
        import agent.tools.kubernetes_tools  # noqa: F401
        import agent.tools.mlflow_tools  # noqa: F401
    except Exception as e:
        print(f"Error importing tools: {e}", file=sys.stderr)

    print(f"Total tools registered: {len(TOOL_REGISTRY)}")
    for name, (_, _, spec) in TOOL_REGISTRY.items():
        print(f" - {name} [{spec.permission}]: {spec.description}")
