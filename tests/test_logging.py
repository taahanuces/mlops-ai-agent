"""Tests for structured JSON logging and sensitive credential redaction."""

from __future__ import annotations

import json
import logging

from agent.logging_utils import (
    JSONFormatter,
    redact_sensitive,
    set_current_request_id,
)


def test_logging_fields_and_redaction():
    """Verify sensitive token, password, and secret keys/values are redacted."""
    raw_payload = {
        "api_key": "sk-ant-api03-secret12345678901234567890",
        "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz",
        "normal_field": "public_data",
        "nested": {
            "password": "super_secret_pw",
            "token": "token123",
            "count": 42,
        },
    }

    redacted = redact_sensitive(raw_payload)
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["authorization"] == "[REDACTED]"
    assert redacted["normal_field"] == "public_data"
    assert redacted["nested"]["password"] == "[REDACTED]"
    assert redacted["nested"]["token"] == "[REDACTED]"
    assert redacted["nested"]["count"] == 42


def test_json_formatter_output():
    """Verify log records format into valid JSON with request_id and timing fields."""
    formatter = JSONFormatter()
    set_current_request_id("req-uuid-test-999")

    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Test execution",
        args=(),
        exc_info=None,
    )
    record.tool_log = {
        "tool": "mlflow_find_best_run",
        "source": "mlflow",
        "status": "ok",
        "duration_ms": 125,
        "evidence_id": "ev-0001",
    }

    formatted_str = formatter.format(record)
    parsed = json.loads(formatted_str)

    assert parsed["level"] == "INFO"
    assert parsed["request_id"] == "req-uuid-test-999"
    assert parsed["tool"] == "mlflow_find_best_run"
    assert parsed["duration_ms"] == 125
