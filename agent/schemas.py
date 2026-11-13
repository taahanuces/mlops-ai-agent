"""Pydantic schemas for the MLOps AI Agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Normalized return envelope for all agent tool executions."""

    evidence_id: str = Field(description="Unique evidence identifier, e.g. ev-0001")
    tool: str = Field(description="Name of the invoked tool")
    source: Literal["dvc", "mlflow", "airflow", "kubernetes", "agent"] = Field(
        description="Underlying system source"
    )
    status: Literal[
        "ok",
        "error",
        "timeout",
        "unavailable",
        "invalid_request",
        "denied",
        "pending_confirmation",
    ] = Field(description="Tool execution status")
    data: dict[str, Any] | list[Any] | None = Field(
        default=None, description="Structured tool output payload"
    )
    error: str | None = Field(default=None, description="Error message if non-ok")
    duration_ms: int = Field(default=0, description="Duration in milliseconds")
    request_id: str = Field(default="", description="Correlated incoming request ID")


class ToolSpec(BaseModel):
    """Specification and metadata for an agent tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    permission: Literal["read", "write", "forbidden"]
    timeout_s: int = 10


class Fact(BaseModel):
    """Fact statement directly cited with successful tool evidence IDs."""

    statement: str
    evidence_ids: list[str] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    """Strictly structured agent response format."""

    summary: str
    known_facts: list[Fact] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    pending_confirmations: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    """Incoming user chat request payload."""

    message: str = Field(..., max_length=2000, description="User prompt")
    session_id: str | None = Field(default=None, description="Optional conversation session ID")


class ChatResponse(BaseModel):
    """API response from agent /chat endpoint."""

    request_id: str
    answer: FinalAnswer
    evidence: list[ToolResult] = Field(default_factory=list)
    pending_actions: list[dict[str, Any]] = Field(default_factory=list)


class PendingAction(BaseModel):
    """Action stored in confirmation store awaiting user authorization."""

    confirmation_id: str
    tool_name: str
    arguments: dict[str, Any]
    description: str
    created_at: float
    ttl_seconds: int = 300
    source: Literal["dvc", "mlflow", "airflow", "kubernetes", "agent"]
