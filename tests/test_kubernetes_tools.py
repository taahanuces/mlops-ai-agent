"""Tests for Kubernetes diagnostic and operational tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.permissions import confirmation_store
from agent.tools.kubernetes_tools import (
    k8s_find_unhealthy_pods,
    k8s_get_deployed_version,
    k8s_get_pod_logs,
    k8s_get_services,
    k8s_list_deployments,
    k8s_list_pods,
    k8s_restart_deployment,
    k8s_scale_deployment,
)


def test_k8s_write_rejected_wrong_namespace():
    """Verify write attempt in unauthorized namespace (e.g. kube-system) is denied."""
    res = k8s_restart_deployment(deployment_name="mlops-inference", namespace="kube-system")
    assert res.status == "denied"
    assert "not authorized" in res.error

    res_scale = k8s_scale_deployment(replicas=2, namespace="kube-system")
    assert res_scale.status == "denied"


def test_k8s_write_requires_confirmation_and_callback():
    """Verify restart and scale return pending_confirmation and their callbacks execute."""
    # Restart
    res = k8s_restart_deployment(deployment_name="mlops-inference", namespace="mlops-agent")
    assert res.status == "pending_confirmation"
    cid = res.data["confirmation_id"]
    assert cid.startswith("conf-")

    with patch("agent.tools.kubernetes_tools._get_k8s_clients") as mock_clients:
        apps_api = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_api)
        _, result, err = confirmation_store.execute_and_consume(cid)
        assert err is None
        assert result["status"] == "restarted"
        assert apps_api.patch_namespaced_deployment.called

    # Scale
    res_scale = k8s_scale_deployment(
        deployment_name="mlops-inference", replicas=3, namespace="mlops-agent"
    )
    assert res_scale.status == "pending_confirmation"
    cid_scale = res_scale.data["confirmation_id"]

    with patch("agent.tools.kubernetes_tools._get_k8s_clients") as mock_clients:
        apps_api = MagicMock()
        mock_clients.return_value = (MagicMock(), apps_api)
        _, result_scale, err = confirmation_store.execute_and_consume(cid_scale)
        assert err is None
        assert result_scale["scaled_replicas"] == 3
        assert apps_api.patch_namespaced_deployment_scale.called


def test_k8s_scale_bounds():
    """Verify scale tool enforces 1 to 5 replica bounds."""
    res_zero = k8s_scale_deployment(replicas=0)
    assert res_zero.status == "denied"

    res_ten = k8s_scale_deployment(replicas=10)
    assert res_ten.status == "denied"


def test_k8s_list_pods():
    mock_pod = MagicMock()
    mock_pod.metadata.name = "pod-1"
    mock_pod.status.phase = "Running"
    mock_pod.status.container_statuses = [MagicMock(restart_count=1, ready=True)]
    mock_pod.spec.node_name = "node-1"
    mock_pod.metadata.creation_timestamp = None

    with patch("agent.tools.kubernetes_tools._get_k8s_clients") as mock_clients:
        core_api = MagicMock()
        core_api.list_namespaced_pod.return_value.items = [mock_pod]
        mock_clients.return_value = (core_api, MagicMock())

        res = k8s_list_pods(namespace="mlops-agent")
        assert res.status == "ok"
        assert len(res.data["pods"]) == 1
        assert res.data["pods"][0]["ready"] is True

        # Error path
        core_api.list_namespaced_pod.side_effect = Exception("k8s api down")
        res_err = k8s_list_pods(namespace="mlops-agent")
        assert res_err.status == "unavailable"


def test_k8s_list_deployments():
    mock_dep = MagicMock()
    mock_dep.metadata.name = "mlops-inference"
    mock_dep.spec.replicas = 2
    mock_dep.status.ready_replicas = 2
    mock_dep.status.available_replicas = 2
    mock_dep.spec.template.spec.containers = [MagicMock(image="mlops-inference:1.0.0")]
    mock_dep.status.conditions = []

    with patch("agent.tools.kubernetes_tools._get_k8s_clients") as mock_clients:
        apps_api = MagicMock()
        apps_api.list_namespaced_deployment.return_value.items = [mock_dep]
        mock_clients.return_value = (MagicMock(), apps_api)

        res = k8s_list_deployments(namespace="mlops-agent")
        assert res.status == "ok"
        assert len(res.data["deployments"]) == 1
        assert res.data["deployments"][0]["image"] == "mlops-inference:1.0.0"


def test_k8s_get_pod_logs():
    with patch("agent.tools.kubernetes_tools._get_k8s_clients") as mock_clients:
        core_api = MagicMock()
        core_api.read_namespaced_pod_log.return_value = "line 1\nline 2"
        mock_clients.return_value = (core_api, MagicMock())

        res = k8s_get_pod_logs(pod_name="pod-1", namespace="mlops-agent")
        assert res.status == "ok"
        assert "line 1" in res.data["logs"]

        core_api.read_namespaced_pod_log.side_effect = Exception("pod not found")
        res_err = k8s_get_pod_logs(pod_name="bad-pod", namespace="mlops-agent")
        assert res_err.status == "error"


def test_k8s_get_services():
    mock_svc = MagicMock()
    mock_svc.metadata.name = "mlops-inference"
    mock_svc.spec.type = "ClusterIP"
    mock_svc.spec.cluster_ip = "10.96.0.1"
    mock_svc.spec.ports = [MagicMock(port=80, target_port=8000, protocol="TCP")]

    mock_ep = MagicMock()
    mock_ep.metadata.name = "mlops-inference"
    sub = MagicMock()
    sub.addresses = [MagicMock(), MagicMock()]
    sub.not_ready_addresses = []
    mock_ep.subsets = [sub]

    with patch("agent.tools.kubernetes_tools._get_k8s_clients") as mock_clients:
        core_api = MagicMock()
        core_api.list_namespaced_service.return_value.items = [mock_svc]
        core_api.read_namespaced_endpoints.return_value = mock_ep
        mock_clients.return_value = (core_api, MagicMock())

        res = k8s_get_services(namespace="mlops-agent")
        assert res.status == "ok"
        assert len(res.data["services"]) == 1
        assert res.data["services"][0]["ready_endpoints"] == 2


def test_k8s_get_deployed_version():
    mock_dep = MagicMock()
    mock_dep.metadata.name = "mlops-inference"
    mock_dep.spec.template.spec.containers = [MagicMock(image="mlops-inference:1.0.0")]
    mock_dep.metadata.labels = {"app.kubernetes.io/version": "1.0.0"}

    mock_cm = MagicMock()
    mock_cm.data = {
        "APP_VERSION": "1.0.0",
        "MODEL_RUN_ID": "run-xyz",
        "DATASET_VERSION": "data-v2.0",
    }

    with patch("agent.tools.kubernetes_tools._get_k8s_clients") as mock_clients:
        core_api = MagicMock()
        core_api.read_namespaced_config_map.return_value = mock_cm
        apps_api = MagicMock()
        apps_api.read_namespaced_deployment.return_value = mock_dep
        mock_clients.return_value = (core_api, apps_api)

        res = k8s_get_deployed_version(namespace="mlops-agent")
        assert res.status == "ok"
        assert res.data["configmap"]["MODEL_RUN_ID"] == "run-xyz"
        assert res.data["configmap"]["DATASET_VERSION"] == "data-v2.0"


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
