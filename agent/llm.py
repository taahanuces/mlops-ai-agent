"""LLMClient protocol, Google Gemini implementation, and FakeLLM for tests."""

from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Any, Protocol

from pydantic import BaseModel, Field

from agent.config import GEMINI_API_KEY, LLM_MODEL, LLM_TIMEOUT_S


class ToolCall(BaseModel):
    """Normalized tool call emitted by an LLM."""

    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    # Opaque provider token (base64) that must be echoed back with the call (Gemini 3 thinking)
    thought_signature: str | None = None


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


class GeminiClient:
    """Direct Google Gemini (google-genai) function-calling client, no heavy frameworks.

    The orchestrator keeps a provider-neutral message history (text, ``tool_use`` and
    ``tool_result`` blocks); this client converts it to Gemini ``Content`` objects.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model or LLM_MODEL
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=int(LLM_TIMEOUT_S * 1000)),
            )
        return self._client

    @staticmethod
    def _to_contents(messages: list[dict[str, Any]]):
        from google.genai import types

        id_to_name: dict[str, str] = {}
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            raw = m["content"]
            blocks = [{"type": "text", "text": raw}] if isinstance(raw, str) else raw
            parts = []
            for blk in blocks:
                btype = blk.get("type")
                if btype == "text":
                    if blk.get("text"):
                        parts.append(types.Part.from_text(text=blk["text"]))
                elif btype == "tool_use":
                    id_to_name[blk["id"]] = blk["name"]
                    sig = blk.get("thought_signature")
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                name=blk["name"], args=blk.get("input", {})
                            ),
                            thought_signature=base64.b64decode(sig) if sig else None,
                        )
                    )
                elif btype == "tool_result":
                    name = id_to_name.get(blk["tool_use_id"], "unknown_tool")
                    content = blk.get("content", "")
                    try:
                        payload = json.loads(content) if isinstance(content, str) else content
                    except ValueError:
                        payload = content
                    if not isinstance(payload, dict):
                        payload = {"result": payload}
                    parts.append(types.Part.from_function_response(name=name, response=payload))
            if parts:
                contents.append(types.Content(role=role, parts=parts))
        return contents

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        from google.genai import types

        client = self._get_client()

        config_kwargs: dict[str, Any] = {"max_output_tokens": 4096, "temperature": 0.0}
        if system:
            config_kwargs["system_instruction"] = system
        if tools:
            decls = [
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters_json_schema=t["input_schema"],
                )
                for t in tools
            ]
            config_kwargs["tools"] = [types.Tool(function_declarations=decls)]
            config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                disable=True
            )

        contents = self._to_contents(messages)
        gen_config = types.GenerateContentConfig(**config_kwargs)
        resp = None
        for attempt in range(5):
            try:
                resp = client.models.generate_content(
                    model=self.model, contents=contents, config=gen_config
                )
                break
            except Exception as exc:  # retry only transient rate-limit / server errors
                code = getattr(exc, "code", None)
                if code not in (429, 500, 503, 504) or attempt == 4:
                    raise
                time.sleep(min(2**attempt * 2, 20))
        assert resp is not None

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        candidates = resp.candidates or []
        parts = (candidates[0].content.parts or []) if candidates and candidates[0].content else []
        for i, part in enumerate(parts):
            if getattr(part, "function_call", None):
                fc = part.function_call
                sig = getattr(part, "thought_signature", None)
                tool_calls.append(
                    ToolCall(
                        id=fc.id or f"call-{uuid.uuid4().hex[:8]}-{i}",
                        name=fc.name,
                        input=dict(fc.args) if fc.args else {},
                        thought_signature=base64.b64encode(sig).decode() if sig else None,
                    )
                )
            elif getattr(part, "text", None) and not getattr(part, "thought", False):
                text_parts.append(part.text)

        finish = str(candidates[0].finish_reason) if candidates else "none"
        return LLMResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else finish,
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
