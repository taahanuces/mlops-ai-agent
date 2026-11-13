"""FastAPI application entrypoint for the MLOps AI Agent."""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from agent.llm import GeminiClient
from agent.logging_utils import (
    get_current_request_id,
    log_tool_call,
    set_current_request_id,
    setup_logger,
)
from agent.orchestrator import Orchestrator
from agent.permissions import confirmation_store
from agent.schemas import ChatRequest, ChatResponse, ToolResult
from agent.tools.registry import next_evidence_id

logger = setup_logger("agent_api")

app = FastAPI(
    title="MLOps AI Agent API",
    description="Evidence-based conversational agent for DVC, MLflow, Airflow 3, and Kubernetes operations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    set_current_request_id(req_id)
    start_time = time.time()
    response: Response = await call_next(request)
    dur = int((time.time() - start_time) * 1000)
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time-Ms"] = str(dur)
    return response


# Default orchestrator with Gemini client
_orchestrator = Orchestrator(llm_client=GeminiClient())


def get_orchestrator() -> Orchestrator:
    return _orchestrator


def set_orchestrator(orchestrator: Orchestrator) -> None:
    global _orchestrator
    _orchestrator = orchestrator


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for the agent service."""
    return {"status": "ok", "service": "mlops-agent"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    """Submit a conversational request to the agent."""
    req_id = get_current_request_id()
    orchestrator = get_orchestrator()

    answer, evidence = orchestrator.run(
        user_prompt=payload.message,
        session_id=payload.session_id,
    )

    pending_list = [
        act.model_dump()
        for act in confirmation_store.list_pending()
        if act.confirmation_id in answer.pending_confirmations
    ]

    return ChatResponse(
        request_id=req_id,
        answer=answer,
        evidence=evidence,
        pending_actions=pending_list,
    )


@app.post("/confirm/{confirmation_id}", response_model=ToolResult)
def confirm_action(confirmation_id: str) -> ToolResult:
    """Execute a pending write action once after user confirmation."""
    req_id = get_current_request_id()
    ev_id = next_evidence_id()
    start = time.time()

    action, result, error = confirmation_store.execute_and_consume(confirmation_id)
    dur = int((time.time() - start) * 1000)

    if not action:
        res = ToolResult(
            evidence_id=ev_id,
            tool="confirm_action",
            source="agent",
            status="invalid_request",
            error=error or "Confirmation not found or expired.",
            duration_ms=dur,
            request_id=req_id,
        )
        log_tool_call("confirm_action", "agent", "invalid_request", dur, ev_id, req_id)
        raise HTTPException(status_code=400, detail=res.model_dump())

    if error:
        res = ToolResult(
            evidence_id=ev_id,
            tool=action.tool_name,
            source=action.source,
            status="error",
            error=error,
            duration_ms=dur,
            request_id=req_id,
        )
        log_tool_call(action.tool_name, action.source, "error", dur, ev_id, req_id)
        return res

    res = ToolResult(
        evidence_id=ev_id,
        tool=action.tool_name,
        source=action.source,
        status="ok",
        data=result if isinstance(result, (dict, list)) else {"result": result},
        duration_ms=dur,
        request_id=req_id,
    )
    log_tool_call(action.tool_name, action.source, "ok", dur, ev_id, req_id)
    return res


@app.get("/pending", response_model=list[dict[str, Any]])
def list_pending_actions() -> list[dict[str, Any]]:
    """List currently active pending write actions awaiting confirmation."""
    return [a.model_dump() for a in confirmation_store.list_pending()]
