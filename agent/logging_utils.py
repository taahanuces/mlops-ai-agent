"""Structured JSON logging with request context tracking and sensitive data redaction."""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import time
from typing import Any

# Context variable for request correlation ID
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")

# Sensitive key patterns
SECRET_KEY_PATTERN = re.compile(
    r"(?i)(token|password|secret|api[_-]?key|authorization|bearer|cookie)"
)

# Sensitive value patterns (JWTs, Anthropic sk- keys, generic secrets)
SECRET_VALUE_PATTERN = re.compile(
    r"(sk-[a-zA-Z0-9\-_]{20,}|eyJ[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]+)"
)


def get_current_request_id() -> str:
    """Retrieve current request ID from contextvar."""
    return request_id_var.get()


def set_current_request_id(req_id: str) -> None:
    """Set current request ID in contextvar."""
    request_id_var.set(req_id)


def redact_sensitive(obj: Any) -> Any:
    """Recursively mask sensitive values and keys in dicts, lists, and strings."""
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for k, v in obj.items():
            if SECRET_KEY_PATTERN.search(str(k)):
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = redact_sensitive(v)
        return cleaned
    elif isinstance(obj, list):
        return [redact_sensitive(item) for item in obj]
    elif isinstance(obj, str):
        return SECRET_VALUE_PATTERN.sub("[REDACTED_SECRET]", obj)
    return obj


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_current_request_id(),
        }

        # Include custom attributes attached to log record
        if hasattr(record, "tool_log") and isinstance(record.tool_log, dict):
            log_entry.update(redact_sensitive(record.tool_log))
        if hasattr(record, "request_log") and isinstance(record.request_log, dict):
            log_entry.update(redact_sensitive(record.request_log))

        return json.dumps(redact_sensitive(log_entry))


def setup_logger(name: str = "mlops_agent") -> logging.Logger:
    """Configures structured JSON logging on stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = setup_logger()


def log_tool_call(
    tool: str,
    source: str,
    status: str,
    duration_ms: int,
    evidence_id: str,
    req_id: str | None = None,
) -> None:
    """Log a single structured tool call entry."""
    req = req_id or get_current_request_id()
    extra = {
        "tool_log": {
            "tool": tool,
            "source": source,
            "status": status,
            "duration_ms": duration_ms,
            "evidence_id": evidence_id,
            "request_id": req,
        }
    }
    logger.info(f"Tool executed: {tool} -> {status}", extra=extra)


def log_request_summary(
    user_message_len: int,
    tools_called: list[str],
    total_ms: int,
    outcome: str,
    req_id: str | None = None,
) -> None:
    """Log a single structured incoming chat request summary."""
    req = req_id or get_current_request_id()
    extra = {
        "request_log": {
            "request_id": req,
            "user_message_len": user_message_len,
            "tools_called": tools_called,
            "total_ms": total_ms,
            "outcome": outcome,
        }
    }
    logger.info(f"Request finished: {outcome}", extra=extra)
