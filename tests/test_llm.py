"""Unit tests for GeminiClient conversion logic (no network)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.llm import FakeLLM, GeminiClient, LLMResponse


def test_to_contents_converts_tool_round_trip():
    messages = [
        {"role": "user", "content": "Which dataset did run 7 use?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "mlflow_get_run_details",
                    "input": {"run_number": 7},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "c1", "content": '{"dataset": "data-v2.0"}'}
            ],
        },
    ]
    contents = GeminiClient._to_contents(messages)
    assert [c.role for c in contents] == ["user", "model", "user"]
    call = contents[1].parts[0].function_call
    assert call.name == "mlflow_get_run_details" and dict(call.args) == {"run_number": 7}
    resp = contents[2].parts[0].function_response
    assert resp.name == "mlflow_get_run_details"
    assert resp.response == {"dataset": "data-v2.0"}


def test_to_contents_wraps_non_dict_and_non_json_results():
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "a", "name": "t", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "plain text"},
                {"type": "tool_result", "tool_use_id": "a", "content": "[1, 2]"},
            ],
        },
    ]
    contents = GeminiClient._to_contents(messages)
    results = [p.function_response.response for p in contents[1].parts]
    assert results == [{"result": "plain text"}, {"result": [1, 2]}]


def _fake_response(parts):
    cand = SimpleNamespace(content=SimpleNamespace(parts=parts), finish_reason="STOP")
    return SimpleNamespace(candidates=[cand])


def test_complete_parses_function_calls_and_text():
    client = GeminiClient(api_key="test", model="m")
    fc = SimpleNamespace(id=None, name="k8s_list_pods", args={"namespace": "mlops-agent"})
    parts = [
        SimpleNamespace(function_call=fc, text=None, thought=False),
        SimpleNamespace(function_call=None, text="hello", thought=False),
        SimpleNamespace(function_call=None, text="hidden reasoning", thought=True),
    ]
    mock = MagicMock()
    mock.models.generate_content.return_value = _fake_response(parts)
    client._client = mock
    out = client.complete(
        [{"role": "user", "content": "pods?"}],
        tools=[{"name": "k8s_list_pods", "description": "d", "input_schema": {"type": "object"}}],
        system="sys",
    )
    assert out.stop_reason == "tool_use"
    assert out.tool_calls[0].name == "k8s_list_pods"
    assert out.tool_calls[0].input == {"namespace": "mlops-agent"}
    assert out.tool_calls[0].id
    assert out.content == "hello"


def test_complete_retries_transient_errors(monkeypatch):
    monkeypatch.setattr("agent.llm.time.sleep", lambda s: None)
    client = GeminiClient(api_key="test", model="m")

    class Transient(Exception):
        code = 503

    mock = MagicMock()
    mock.models.generate_content.side_effect = [
        Transient(),
        _fake_response([SimpleNamespace(function_call=None, text="ok", thought=False)]),
    ]
    client._client = mock
    out = client.complete([{"role": "user", "content": "hi"}])
    assert out.content == "ok"
    assert mock.models.generate_content.call_count == 2


def test_complete_does_not_retry_client_errors(monkeypatch):
    monkeypatch.setattr("agent.llm.time.sleep", lambda s: None)
    client = GeminiClient(api_key="test", model="m")

    class Bad(Exception):
        code = 404

    mock = MagicMock()
    mock.models.generate_content.side_effect = Bad()
    client._client = mock
    try:
        client.complete([{"role": "user", "content": "hi"}])
        raise AssertionError("expected exception")
    except Bad:
        pass
    assert mock.models.generate_content.call_count == 1


def test_fake_llm_default_response():
    assert isinstance(FakeLLM().complete([{"role": "user", "content": "x"}]), LLMResponse)


def test_thought_signature_round_trip():
    import base64

    sig = base64.b64encode(b"opaque-sig").decode()
    client = GeminiClient(api_key="test", model="m")
    fc = SimpleNamespace(id=None, name="t", args={})
    part = SimpleNamespace(
        function_call=fc, text=None, thought=False, thought_signature=b"opaque-sig"
    )
    mock = MagicMock()
    mock.models.generate_content.return_value = _fake_response([part])
    client._client = mock
    out = client.complete([{"role": "user", "content": "x"}])
    assert out.tool_calls[0].thought_signature == sig

    contents = GeminiClient._to_contents(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "1",
                        "name": "t",
                        "input": {},
                        "thought_signature": sig,
                    }
                ],
            }
        ]
    )
    assert contents[0].parts[0].thought_signature == b"opaque-sig"
