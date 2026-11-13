"""Tests for the interactive CLI client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.cli import main


def test_cli_exit_immediately():
    with patch("httpx.Client") as mock_client_cls, patch("builtins.input", side_effect=["exit"]):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.return_value.status_code = 200

        main()
        mock_client.get.assert_called_once_with("/health")


def test_cli_health_failure():
    with patch("httpx.Client") as mock_client_cls, patch("builtins.input", side_effect=["quit"]):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.return_value.status_code = 503
        mock_client.get.return_value.text = "Service Unavailable"

        main()
        mock_client.get.assert_called_once_with("/health")


def test_cli_health_exception():
    with patch("httpx.Client") as mock_client_cls, patch("builtins.input", side_effect=["quit"]):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.side_effect = Exception("Connection refused")

        main()


def test_cli_confirm_command():
    with (
        patch("httpx.Client") as mock_client_cls,
        patch("builtins.input", side_effect=["confirm conf-123", "exit"]),
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.return_value.status_code = 200
        mock_client.post.return_value.json.return_value = {"status": "ok", "action": "restarted"}

        main()
        mock_client.post.assert_called_once_with("/confirm/conf-123")


def test_cli_chat_flow_and_rendering():
    chat_response = {
        "answer": {
            "summary": "This is a summary.",
            "known_facts": [{"statement": "Model is good", "evidence_ids": ["ev-1"]}],
            "uncertainties": ["Latency is unknown"],
            "recommended_actions": ["Deploy model"],
            "pending_confirmations": ["conf-999"],
        }
    }
    with (
        patch("httpx.Client") as mock_client_cls,
        patch("builtins.input", side_effect=["", "Hello agent", "exit"]),
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.return_value.status_code = 200

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = chat_response
        mock_client.post.return_value = mock_resp

        main()
        mock_client.post.assert_called_once_with("/chat", json={"message": "Hello agent"})


def test_cli_chat_error_response():
    with (
        patch("httpx.Client") as mock_client_cls,
        patch("builtins.input", side_effect=["error test", "exit"]),
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.return_value.status_code = 200

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal error"
        mock_client.post.return_value = mock_resp

        main()


def test_cli_keyboard_interrupt():
    with (
        patch("httpx.Client") as mock_client_cls,
        patch("builtins.input", side_effect=KeyboardInterrupt),
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.return_value.status_code = 200

        main()
