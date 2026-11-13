"""LLMClient protocol, Anthropic direct implementation, and FakeLLM for tests."""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import BaseModel, Field

from agent.config import ANTHROPIC_API_KEY, LLM_MODEL


class ToolCall(BaseModel):
    """Normalized tool call emitted by an LLM."""

    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Normalized LLM response containing text and/or tool invocations."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: str = "end_turn"


class LLMClient(Protocol):
    """Protocol for LLM client implementations."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        """Execute a completion turn."""
        ...


class AnthropicClient:
    """Direct Anthropic API tool use client without heavy frameworks."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or ANTHROPIC_API_KEY
        self.model = model or LLM_MODEL
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        client = self._get_client()

        anthropic_tools = []
        if tools:
            for t in tools:
                anthropic_tools.append(
                    {
                        "name": t["name"],
                        "description": t["description"],
                        "input_schema": t["input_schema"],
                    }
                )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        if system:
            kwargs["system"] = system

        resp = client.messages.create(**kwargs)

        text_parts = []
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        input=block.input if isinstance(block.input, dict) else {},
                    )
                )

        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "end_turn",
        )


class FakeLLM:
    """Scriptable test LLM that emits pre-configured LLMResponses or tool calls."""

    def __init__(self, responses: list[LLMResponse] | None = None):
        self.responses: list[LLMResponse] = list(responses or [])
        self.history: list[dict[str, Any]] = []

    def set_responses(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        self.history.append({"messages": messages, "tools": tools, "system": system})
        if self.responses:
            return self.responses.pop(0)

        # Default fallback response
        return LLMResponse(
            content=json.dumps(
                {
                    "summary": "Default scripted response.",
                    "known_facts": [],
                    "uncertainties": [],
                    "recommended_actions": [],
                    "pending_confirmations": [],
                }
            ),
            tool_calls=[],
        )
