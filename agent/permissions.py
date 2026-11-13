"""Permission model, namespace & allowlist checks, and confirmation store."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable

from agent.config import (
    ALLOWED_DAG_IDS,
    APPROVED_NAMESPACES,
    CONFIRMATION_TTL_S,
)
from agent.schemas import PendingAction

# Approved DVC/Git command prefixes (argv format)
APPROVED_DVC_COMMAND_PREFIXES: list[list[str]] = [
    ["dvc", "status"],
    ["dvc", "data", "status"],
    ["dvc", "diff"],
    ["dvc", "list"],
    ["dvc", "get"],
    ["git", "rev-parse"],
    ["git", "describe"],
    ["git", "tag", "--list"],
    ["git", "tag", "-l"],
    ["git", "log"],
    ["git", "show"],
]


class ConfirmationStore:
    """Thread-safe in-memory store for write actions awaiting user confirmation."""

    def __init__(self, ttl_seconds: int = CONFIRMATION_TTL_S):
        self._lock = threading.Lock()
        self._actions: dict[str, tuple[PendingAction, Callable[[], Any]]] = {}
        self._ttl_seconds = ttl_seconds

    def store_action(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        description: str,
        source: str,
        callback: Callable[[], Any],
    ) -> str:
        confirmation_id = f"conf-{uuid.uuid4().hex[:8]}"
        action = PendingAction(
            confirmation_id=confirmation_id,
            tool_name=tool_name,
            arguments=arguments,
            description=description,
            created_at=time.time(),
            ttl_seconds=self._ttl_seconds,
            source=source,  # type: ignore[arg-type]
        )
        with self._lock:
            self._actions[confirmation_id] = (action, callback)
        return confirmation_id

    def get_action(self, confirmation_id: str) -> PendingAction | None:
        with self._lock:
            entry = self._actions.get(confirmation_id)
            if not entry:
                return None
            action, _ = entry
            if time.time() - action.created_at > action.ttl_seconds:
                del self._actions[confirmation_id]
                return None
            return action

    def execute_and_consume(
        self, confirmation_id: str
    ) -> tuple[PendingAction | None, Any, str | None]:
        """Execute confirmation callback once and remove it. Returns (action, result, error)."""
        with self._lock:
            entry = self._actions.pop(confirmation_id, None)
            if not entry:
                return None, None, "Confirmation ID not found or already consumed."
            action, callback = entry
            if time.time() - action.created_at > action.ttl_seconds:
                return None, None, f"Confirmation ID {confirmation_id} has expired."

        try:
            result = callback()
            return action, result, None
        except Exception as e:
            return action, None, f"Execution failed: {str(e)}"

    def list_pending(self) -> list[PendingAction]:
        now = time.time()
        with self._lock:
            # Clean expired
            expired = [
                cid
                for cid, (act, _) in self._actions.items()
                if now - act.created_at > act.ttl_seconds
            ]
            for cid in expired:
                del self._actions[cid]
            return [act for act, _ in self._actions.values()]


# Global confirmation store instance
confirmation_store = ConfirmationStore()


def check_kubernetes_namespace(namespace: str | None) -> bool:
    """Validate that target namespace is in approved list."""
    if not namespace:
        return True
    return namespace in APPROVED_NAMESPACES


def check_airflow_dag_id(dag_id: str) -> bool:
    """Validate that target DAG is in approved list."""
    return dag_id in ALLOWED_DAG_IDS


def check_dvc_command_allowed(argv: list[str]) -> bool:
    """Validate that command argv begins with an allow-listed prefix."""
    if not argv:
        return False
    for prefix in APPROVED_DVC_COMMAND_PREFIXES:
        if len(argv) >= len(prefix) and argv[: len(prefix)] == prefix:
            # Special check for dvc get: only output inside .cache/dataset_checkouts/
            if argv[:2] == ["dvc", "get"]:
                if "-o" in argv:
                    out_idx = argv.index("-o") + 1
                    if out_idx < len(argv):
                        out_dir = argv[out_idx]
                        if not out_dir.startswith(".cache/dataset_checkouts/"):
                            return False
                elif "--out" in argv:
                    out_idx = argv.index("--out") + 1
                    if out_idx < len(argv):
                        out_dir = argv[out_idx]
                        if not out_dir.startswith(".cache/dataset_checkouts/"):
                            return False
            return True
    return False
