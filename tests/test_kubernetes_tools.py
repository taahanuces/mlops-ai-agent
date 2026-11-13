"""Tests for Kubernetes diagnostic and operational tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.tools.kubernetes_tools import (
    k8s_find_unhealthy_pods,
    k8s_restart_deployment,
    k8s_scale_deployment,
)


def test_k8s_write_rejected_wrong_namespace():
    """Verify write attempt in unauthorized namespace (e.g. kube-system) is denied."""
    res = k8s_restart_deployment(deployment_name="mlops-inference", namespace="kube-system")
    assert res.status == "denied"
    assert "not authorized" in res.error


def test_k8s_write_requires_confirmation():
    """Verify restart action returns pending_confirmation with valid confirmation ID."""
    res = k8s_restart_deployment(deployment_name="mlops-inference", namespace="mlops-agent")
    assert res.status == "pending_confirmation"
    assert "confirmation_id" in res.data
    assert res.data["confirmation_id"].startswith("conf-")


def test_k8s_scale_bounds():
    """Verify scale tool enforces 1 to 5 replica bounds."""
    # Attempt 0 replicas -> denied
    res_zero = k8s_scale_deployment(replicas=0)
    assert res_zero.status == "denied"

    # Attempt 10 replicas -> denied
    res_ten = k8s_scale_deployment(replicas=10)
    assert res_ten.status == "denied"


def test_k8s_find_unhealthy_pods():
    """Test identification of CrashLoopBackOff pods and warning events."""
    mock_pod = MagicMock()
    mock_pod.metadata.name = "mlops-inference-badpod"
    mock_pod.status.phase = "Running"
    mock_cs = MagicMock()
    mock_cs.ready = False
    mock_cs.restart_count = 5
    mock_cs.state.waiting.reason = "CrashLoopBackOff"
    mock_cs.state.waiting.message = "back-off restarting failed container"
    mock_cs.last_state.terminated = None
    mock_pod.status.container_statuses = [mock_cs]

    mock_event = MagicMock()
    mock_event.involved_object.name = "mlops-inference-badpod"
    mock_event.type = "Warning"
    mock_event.reason = "Failed"
    mock_event.message = "Readiness probe failed: HTTP 503"

    with patch("agent.tools.kubernetes_tools._get_k8s_clients") as mock_clients:
        core_api = MagicMock()
        core_api.list_namespaced_pod.return_value.items = [mock_pod]
        core_api.list_namespaced_event.return_value.items = [mock_event]
        mock_clients.return_value = (core_api, MagicMock())

        res = k8s_find_unhealthy_pods(namespace="mlops-agent")
        assert res.status == "ok"
        assert res.data["count"] == 1
        pod_info = res.data["unhealthy_pods"][0]
        assert pod_info["pod_name"] == "mlops-inference-badpod"
        assert pod_info["restarts"] == 5
