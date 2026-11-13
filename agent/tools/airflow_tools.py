"""Airflow 3 API client tools with JWT authentication and confirmation gating."""

from __future__ import annotations

import time
from typing import Any

import httpx
from pydantic import BaseModel, Field

from agent.config import (
    AIRFLOW_BASE_URL,
    AIRFLOW_PASSWORD,
    AIRFLOW_USERNAME,
    ALLOWED_DAG_IDS,
    TOOL_TIMEOUT_S,
)
from agent.permissions import check_airflow_dag_id, confirmation_store
from agent.schemas import ToolResult
from agent.tools.registry import register_tool

# In-memory JWT token cache: (token, expiry_timestamp)
_TOKEN_CACHE: tuple[str, float] | None = None


def _get_auth_token() -> str | None:
    """Obtain or return cached JWT bearer token for Airflow API."""
    global _TOKEN_CACHE
    now = time.time()
    if _TOKEN_CACHE and _TOKEN_CACHE[1] > now + 60:
        return _TOKEN_CACHE[0]

    try:
        # Airflow 3 token endpoint: POST /auth/token or /api/v2/auth/token
        url = f"{AIRFLOW_BASE_URL.rstrip('/')}/auth/token"
        with httpx.Client(timeout=TOOL_TIMEOUT_S) as client:
            resp = client.post(
                url,
                json={"username": AIRFLOW_USERNAME, "password": AIRFLOW_PASSWORD},
            )
            if resp.status_code == 404:
                # Try fallback API path
                resp = client.post(
                    f"{AIRFLOW_BASE_URL.rstrip('/')}/api/v2/auth/token",
                    json={"username": AIRFLOW_USERNAME, "password": AIRFLOW_PASSWORD},
                )
            if resp.status_code == 200:
                data = resp.json()
                token = data.get("access_token") or data.get("token")
                expires_in = data.get("expires_in", 3600)
                if token:
                    _TOKEN_CACHE = (token, now + float(expires_in))
                    return token
    except Exception:
        pass
    return None


def _airflow_request(
    method: str,
    endpoint: str,
    json_data: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    """Execute authenticated HTTP request against Airflow API."""
    token = _get_auth_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    base = AIRFLOW_BASE_URL.rstrip("/")
    url = f"{base}/api/v2/{endpoint.lstrip('/')}"

    with httpx.Client(timeout=TOOL_TIMEOUT_S) as client:
        resp = client.request(
            method=method,
            url=url,
            headers=headers,
            json=json_data,
            params=params,
            auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD) if not token else None,
        )
        try:
            return resp.status_code, resp.json()
        except Exception:
            return resp.status_code, resp.text


# Input models
class ListDagsInput(BaseModel):
    pass


class GetDagRunsInput(BaseModel):
    dag_id: str = Field(default="training_pipeline", description="Target DAG ID")
    limit: int = Field(default=5, ge=1, le=50, description="Max runs to retrieve")


class GetFailedTasksInput(BaseModel):
    dag_id: str = Field(default="training_pipeline", description="Target DAG ID")
    dag_run_id: str = Field(..., description="DAG run ID")


class GetTaskLogsInput(BaseModel):
    dag_id: str = Field(default="training_pipeline", description="Target DAG ID")
    dag_run_id: str = Field(..., description="DAG run ID")
    task_id: str = Field(..., description="Task ID within DAG")
    try_number: int = Field(default=1, description="Task execution attempt number")
    tail_lines: int = Field(default=50, description="Number of tail lines to retrieve")


class TriggerDagInput(BaseModel):
    dag_id: str = Field(default="training_pipeline", description="Allow-listed DAG ID to trigger")
    conf: dict[str, Any] = Field(
        default_factory=dict, description="Optional configuration parameters"
    )


# Tools
@register_tool(
    name="airflow_list_dags",
    description="List Airflow DAGs with their paused states, schedule, and description.",
    permission="read",
    input_model=ListDagsInput,
    source="airflow",
)
def airflow_list_dags(**kwargs: Any) -> ToolResult:
    """List DAGs from Airflow API."""
    try:
        status_code, body = _airflow_request("GET", "dags")
        if status_code >= 400:
            return ToolResult(
                evidence_id="",
                tool="airflow_list_dags",
                source="airflow",
                status="unavailable" if status_code >= 500 else "error",
                error=f"Airflow API error (status {status_code}): {body}",
            )

        dags = body.get("dags", []) if isinstance(body, dict) else []
        summary = [
            {
                "dag_id": d.get("dag_id"),
                "is_paused": d.get("is_paused"),
                "description": d.get("description"),
                "tags": [t.get("name") if isinstance(t, dict) else t for t in d.get("tags", [])],
            }
            for d in dags
        ]
        return ToolResult(
            evidence_id="",
            tool="airflow_list_dags",
            source="airflow",
            status="ok",
            data={"dags": summary, "count": len(summary)},
        )
    except httpx.TimeoutException:
        return ToolResult(
            evidence_id="",
            tool="airflow_list_dags",
            source="airflow",
            status="timeout",
            error="Airflow request timed out",
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="airflow_list_dags",
            source="airflow",
            status="unavailable",
            error=f"Failed to connect to Airflow: {str(e)}",
        )


@register_tool(
    name="airflow_get_dag_runs",
    description="Get recent runs for an Airflow DAG with states, start/end times, and run IDs.",
    permission="read",
    input_model=GetDagRunsInput,
    source="airflow",
)
def airflow_get_dag_runs(
    dag_id: str = "training_pipeline", limit: int = 5, **kwargs: Any
) -> ToolResult:
    """Retrieve DAG runs."""
    try:
        status_code, body = _airflow_request(
            "GET",
            f"dags/{dag_id}/dagRuns",
            params={"limit": limit, "order_by": "-start_date"},
        )
        if status_code >= 400:
            return ToolResult(
                evidence_id="",
                tool="airflow_get_dag_runs",
                source="airflow",
                status="unavailable" if status_code >= 500 else "error",
                error=f"Airflow API error (status {status_code}): {body}",
            )

        runs = body.get("dag_runs", []) if isinstance(body, dict) else []
        data = [
            {
                "dag_run_id": r.get("dag_run_id"),
                "state": r.get("state"),
                "logical_date": r.get("logical_date"),
                "start_date": r.get("start_date"),
                "end_date": r.get("end_date"),
            }
            for r in runs
        ]
        return ToolResult(
            evidence_id="",
            tool="airflow_get_dag_runs",
            source="airflow",
            status="ok",
            data={"dag_id": dag_id, "runs": data, "count": len(data)},
        )
    except httpx.TimeoutException:
        return ToolResult(
            evidence_id="",
            tool="airflow_get_dag_runs",
            source="airflow",
            status="timeout",
            error="Airflow request timed out",
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="airflow_get_dag_runs",
            source="airflow",
            status="unavailable",
            error=f"Airflow connection error: {str(e)}",
        )


@register_tool(
    name="airflow_get_failed_tasks",
    description="Identify failed or upstream_failed task instances in a given DAG run with attempt counts.",
    permission="read",
    input_model=GetFailedTasksInput,
    source="airflow",
)
def airflow_get_failed_tasks(
    dag_id: str = "training_pipeline", dag_run_id: str = "", **kwargs: Any
) -> ToolResult:
    """Fetch task instances in failed or upstream_failed state."""
    try:
        status_code, body = _airflow_request(
            "GET", f"dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"
        )
        if status_code >= 400:
            return ToolResult(
                evidence_id="",
                tool="airflow_get_failed_tasks",
                source="airflow",
                status="unavailable" if status_code >= 500 else "error",
                error=f"Airflow API returned {status_code}: {body}",
            )

        tis = body.get("task_instances", []) if isinstance(body, dict) else []
        failed = [
            {
                "task_id": t.get("task_id"),
                "state": t.get("state"),
                "try_number": t.get("try_number"),
                "start_date": t.get("start_date"),
                "end_date": t.get("end_date"),
            }
            for t in tis
            if t.get("state") in ("failed", "upstream_failed")
        ]
        return ToolResult(
            evidence_id="",
            tool="airflow_get_failed_tasks",
            source="airflow",
            status="ok",
            data={"dag_id": dag_id, "dag_run_id": dag_run_id, "failed_tasks": failed},
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="airflow_get_failed_tasks",
            source="airflow",
            status="unavailable",
            error=f"Failed to query failed tasks: {str(e)}",
        )


@register_tool(
    name="airflow_get_task_logs",
    description="Fetch recent execution logs for a specific task instance in a DAG run.",
    permission="read",
    input_model=GetTaskLogsInput,
    source="airflow",
)
def airflow_get_task_logs(
    dag_id: str = "training_pipeline",
    dag_run_id: str = "",
    task_id: str = "",
    try_number: int = 1,
    tail_lines: int = 50,
    **kwargs: Any,
) -> ToolResult:
    """Fetch and truncate logs for a task instance."""
    try:
        endpoint = f"dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{try_number}"
        status_code, body = _airflow_request("GET", endpoint)
        if status_code >= 400:
            return ToolResult(
                evidence_id="",
                tool="airflow_get_task_logs",
                source="airflow",
                status="unavailable" if status_code >= 500 else "error",
                error=f"Airflow returned status {status_code}: {body}",
            )

        log_content = body if isinstance(body, str) else body.get("content", str(body))
        lines = log_content.splitlines()
        truncated = "\n".join(lines[-tail_lines:]) if len(lines) > tail_lines else log_content

        return ToolResult(
            evidence_id="",
            tool="airflow_get_task_logs",
            source="airflow",
            status="ok",
            data={
                "dag_id": dag_id,
                "dag_run_id": dag_run_id,
                "task_id": task_id,
                "try_number": try_number,
                "total_lines": len(lines),
                "tail_lines": tail_lines,
                "logs": truncated,
            },
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="airflow_get_task_logs",
            source="airflow",
            status="unavailable",
            error=f"Failed to fetch task logs: {str(e)}",
        )


@register_tool(
    name="airflow_trigger_training_dag",
    description="Trigger an Airflow training DAG run. Requires explicit user confirmation before executing.",
    permission="write",
    input_model=TriggerDagInput,
    source="airflow",
)
def airflow_trigger_training_dag(
    dag_id: str = "training_pipeline", conf: dict[str, Any] | None = None, **kwargs: Any
) -> ToolResult:
    """Store trigger action in confirmation store and require confirmation ID."""
    if not check_airflow_dag_id(dag_id):
        return ToolResult(
            evidence_id="",
            tool="airflow_trigger_training_dag",
            source="airflow",
            status="denied",
            error=f"DAG '{dag_id}' is not in the allowed list: {ALLOWED_DAG_IDS}",
        )

    def _execute():
        code, body = _airflow_request(
            "POST", f"dags/{dag_id}/dagRuns", json_data={"conf": conf or {}}
        )
        if code >= 400:
            raise RuntimeError(f"Airflow trigger returned status {code}: {body}")
        return body

    desc = f"Trigger Airflow DAG '{dag_id}' with configuration {conf or {}}"
    cid = confirmation_store.store_action(
        tool_name="airflow_trigger_training_dag",
        arguments={"dag_id": dag_id, "conf": conf or {}},
        description=desc,
        source="airflow",
        callback=_execute,
    )

    return ToolResult(
        evidence_id="",
        tool="airflow_trigger_training_dag",
        source="airflow",
        status="pending_confirmation",
        data={
            "confirmation_id": cid,
            "action": desc,
            "ttl_seconds": 300,
            "instructions": f"Call POST /confirm/{cid} to authorize triggering DAG {dag_id}.",
        },
    )
