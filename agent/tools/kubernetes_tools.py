"""Kubernetes diagnostic and operational tools with namespace security checks."""

from __future__ import annotations

import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from agent.config import APPROVED_NAMESPACES, K8S_NAMESPACE
from agent.permissions import check_kubernetes_namespace, confirmation_store
from agent.schemas import ToolResult
from agent.tools.registry import register_tool


def _get_k8s_clients():
    """Initialize official Kubernetes API clients."""
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()
    return client.CoreV1Api(), client.AppsV1Api()


# Input models
class ListPodsInput(BaseModel):
    namespace: Optional[str] = Field(default=None, description="Kubernetes namespace")


class ListDeploymentsInput(BaseModel):
    namespace: Optional[str] = Field(default=None, description="Kubernetes namespace")


class GetPodLogsInput(BaseModel):
    pod_name: str = Field(..., description="Target pod name")
    namespace: Optional[str] = Field(default=None, description="Kubernetes namespace")
    tail_lines: int = Field(default=50, ge=1, le=500, description="Number of tail lines")
    previous: bool = Field(
        default=False, description="Whether to fetch previously crashed container logs"
    )


class GetServicesInput(BaseModel):
    namespace: Optional[str] = Field(default=None, description="Kubernetes namespace")


class FindUnhealthyPodsInput(BaseModel):
    namespace: Optional[str] = Field(default=None, description="Kubernetes namespace")


class GetDeployedVersionInput(BaseModel):
    namespace: Optional[str] = Field(default=None, description="Kubernetes namespace")
    deployment_name: str = Field(default="mlops-inference", description="Deployment name")


class RestartDeploymentInput(BaseModel):
    deployment_name: str = Field(default="mlops-inference", description="Target deployment")
    namespace: Optional[str] = Field(default=None, description="Kubernetes namespace")


class ScaleDeploymentInput(BaseModel):
    deployment_name: str = Field(default="mlops-inference", description="Target deployment")
    replicas: int = Field(..., ge=1, le=5, description="Desired replicas between 1 and 5")
    namespace: Optional[str] = Field(default=None, description="Kubernetes namespace")


# Tools
@register_tool(
    name="k8s_list_pods",
    description="List all pods in the approved namespace with phase, readiness, restarts, and node.",
    permission="read",
    input_model=ListPodsInput,
    source="kubernetes",
)
def k8s_list_pods(namespace: Optional[str] = None, **kwargs: Any) -> ToolResult:
    """List pods in approved namespace."""
    ns = namespace or K8S_NAMESPACE
    if not check_kubernetes_namespace(ns):
        return ToolResult(
            evidence_id="",
            tool="k8s_list_pods",
            source="kubernetes",
            status="denied",
            error=f"Namespace '{ns}' is not authorized. Approved: {APPROVED_NAMESPACES}",
        )

    try:
        core_api, _ = _get_k8s_clients()
        pods = core_api.list_namespaced_pod(ns)
        pod_list = []
        for pod in pods.items:
            restarts = 0
            ready = False
            if pod.status.container_statuses:
                restarts = sum(cs.restart_count for cs in pod.status.container_statuses)
                ready = all(cs.ready for cs in pod.status.container_statuses)

            pod_list.append(
                {
                    "name": pod.metadata.name,
                    "phase": pod.status.phase,
                    "ready": ready,
                    "restarts": restarts,
                    "node": pod.spec.node_name,
                    "created_at": (
                        pod.metadata.creation_timestamp.isoformat()
                        if pod.metadata.creation_timestamp
                        else None
                    ),
                }
            )

        return ToolResult(
            evidence_id="",
            tool="k8s_list_pods",
            source="kubernetes",
            status="ok",
            data={"namespace": ns, "pods": pod_list, "count": len(pod_list)},
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="k8s_list_pods",
            source="kubernetes",
            status="unavailable",
            error=f"Kubernetes API unreachable: {str(e)}",
        )


@register_tool(
    name="k8s_list_deployments",
    description="List deployments in the approved namespace with desired/ready/available replicas and container image tags.",
    permission="read",
    input_model=ListDeploymentsInput,
    source="kubernetes",
)
def k8s_list_deployments(namespace: Optional[str] = None, **kwargs: Any) -> ToolResult:
    """List deployments."""
    ns = namespace or K8S_NAMESPACE
    if not check_kubernetes_namespace(ns):
        return ToolResult(
            evidence_id="",
            tool="k8s_list_deployments",
            source="kubernetes",
            status="denied",
            error=f"Namespace '{ns}' is not authorized. Approved: {APPROVED_NAMESPACES}",
        )

    try:
        _, apps_api = _get_k8s_clients()
        deps = apps_api.list_namespaced_deployment(ns)
        res = []
        for d in deps.items:
            img = (
                d.spec.template.spec.containers[0].image
                if d.spec.template.spec.containers
                else None
            )
            res.append(
                {
                    "name": d.metadata.name,
                    "replicas_desired": d.spec.replicas,
                    "replicas_ready": d.status.ready_replicas or 0,
                    "replicas_available": d.status.available_replicas or 0,
                    "image": img,
                    "conditions": [
                        {"type": c.type, "status": c.status, "reason": c.reason}
                        for c in (d.status.conditions or [])
                    ],
                }
            )
        return ToolResult(
            evidence_id="",
            tool="k8s_list_deployments",
            source="kubernetes",
            status="ok",
            data={"namespace": ns, "deployments": res},
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="k8s_list_deployments",
            source="kubernetes",
            status="unavailable",
            error=f"Kubernetes API error: {str(e)}",
        )


@register_tool(
    name="k8s_get_pod_logs",
    description="Retrieve execution or crash logs from a specific pod in the approved namespace.",
    permission="read",
    input_model=GetPodLogsInput,
    source="kubernetes",
)
def k8s_get_pod_logs(
    pod_name: str,
    namespace: Optional[str] = None,
    tail_lines: int = 50,
    previous: bool = False,
    **kwargs: Any,
) -> ToolResult:
    """Fetch logs from pod."""
    ns = namespace or K8S_NAMESPACE
    if not check_kubernetes_namespace(ns):
        return ToolResult(
            evidence_id="",
            tool="k8s_get_pod_logs",
            source="kubernetes",
            status="denied",
            error=f"Namespace '{ns}' is not authorized.",
        )

    try:
        core_api, _ = _get_k8s_clients()
        raw_logs = core_api.read_namespaced_pod_log(
            name=pod_name,
            namespace=ns,
            tail_lines=tail_lines,
            previous=previous,
        )
        return ToolResult(
            evidence_id="",
            tool="k8s_get_pod_logs",
            source="kubernetes",
            status="ok",
            data={
                "pod_name": pod_name,
                "namespace": ns,
                "previous": previous,
                "logs": raw_logs,
            },
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="k8s_get_pod_logs",
            source="kubernetes",
            status="error",
            error=f"Failed to fetch logs for pod {pod_name}: {str(e)}",
        )


@register_tool(
    name="k8s_get_services",
    description="List Kubernetes services and their endpoint readiness counts in the approved namespace.",
    permission="read",
    input_model=GetServicesInput,
    source="kubernetes",
)
def k8s_get_services(namespace: Optional[str] = None, **kwargs: Any) -> ToolResult:
    """List services and endpoint status."""
    ns = namespace or K8S_NAMESPACE
    if not check_kubernetes_namespace(ns):
        return ToolResult(
            evidence_id="",
            tool="k8s_get_services",
            source="kubernetes",
            status="denied",
            error=f"Namespace '{ns}' is not authorized.",
        )

    try:
        core_api, _ = _get_k8s_clients()
        services = core_api.list_namespaced_service(ns)
        res = []
        for svc in services.items:
            endpoints_count = 0
            try:
                ep = core_api.read_namespaced_endpoints(svc.metadata.name, ns)
                if ep.subsets:
                    for s in ep.subsets:
                        if s.addresses:
                            endpoints_count += len(s.addresses)
            except Exception:
                pass

            res.append(
                {
                    "name": svc.metadata.name,
                    "type": svc.spec.type,
                    "cluster_ip": svc.spec.cluster_ip,
                    "ports": [
                        {"port": p.port, "target_port": str(p.target_port)} for p in svc.spec.ports
                    ],
                    "ready_endpoints": endpoints_count,
                }
            )

        return ToolResult(
            evidence_id="",
            tool="k8s_get_services",
            source="kubernetes",
            status="ok",
            data={"namespace": ns, "services": res},
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="k8s_get_services",
            source="kubernetes",
            status="unavailable",
            error=f"Failed to fetch services: {str(e)}",
        )


@register_tool(
    name="k8s_find_unhealthy_pods",
    description="Inspect pods in approved namespace for non-ready states, CrashLoopBackOff, ImagePullBackOff, restarts, and warning events.",
    permission="read",
    input_model=FindUnhealthyPodsInput,
    source="kubernetes",
)
def k8s_find_unhealthy_pods(namespace: Optional[str] = None, **kwargs: Any) -> ToolResult:
    """Identify unhealthy pods and correlate recent warning events."""
    ns = namespace or K8S_NAMESPACE
    if not check_kubernetes_namespace(ns):
        return ToolResult(
            evidence_id="",
            tool="k8s_find_unhealthy_pods",
            source="kubernetes",
            status="denied",
            error=f"Namespace '{ns}' is not authorized.",
        )

    try:
        core_api, _ = _get_k8s_clients()
        pods = core_api.list_namespaced_pod(ns)
        events = core_api.list_namespaced_event(ns)

        unhealthy = []
        for pod in pods.items:
            is_healthy = True
            reasons = []
            restarts = 0
            last_term_state = None

            if pod.status.phase != "Running":
                is_healthy = False
                reasons.append(f"Phase is {pod.status.phase}")

            if pod.status.container_statuses:
                for cs in pod.status.container_statuses:
                    restarts += cs.restart_count
                    if not cs.ready:
                        is_healthy = False
                    if cs.state.waiting:
                        is_healthy = False
                        reasons.append(
                            f"Waiting: {cs.state.waiting.reason} ({cs.state.waiting.message})"
                        )
                    if cs.last_state and cs.last_state.terminated:
                        last_term_state = {
                            "exit_code": cs.last_state.terminated.exit_code,
                            "reason": cs.last_state.terminated.reason,
                            "message": cs.last_state.terminated.message,
                        }

            if not is_healthy or restarts > 0:
                # Find pod events
                pod_events = [
                    f"{ev.reason}: {ev.message}"
                    for ev in events.items
                    if ev.involved_object.name == pod.metadata.name and ev.type == "Warning"
                ]

                unhealthy.append(
                    {
                        "pod_name": pod.metadata.name,
                        "phase": pod.status.phase,
                        "restarts": restarts,
                        "reasons": reasons,
                        "last_termination": last_term_state,
                        "warning_events": pod_events[-5:],
                    }
                )

        return ToolResult(
            evidence_id="",
            tool="k8s_find_unhealthy_pods",
            source="kubernetes",
            status="ok",
            data={
                "namespace": ns,
                "unhealthy_pods": unhealthy,
                "count": len(unhealthy),
            },
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="k8s_find_unhealthy_pods",
            source="kubernetes",
            status="unavailable",
            error=f"Failed to inspect pod health: {str(e)}",
        )


@register_tool(
    name="k8s_get_deployed_version",
    description="Get deployed version info: image tag, deployment version label, ConfigMap values, and /version endpoint data if available.",
    permission="read",
    input_model=GetDeployedVersionInput,
    source="kubernetes",
)
def k8s_get_deployed_version(
    namespace: Optional[str] = None,
    deployment_name: str = "mlops-inference",
    **kwargs: Any,
) -> ToolResult:
    """Retrieve version information across Deployment, ConfigMap, and running pods."""
    ns = namespace or K8S_NAMESPACE
    if not check_kubernetes_namespace(ns):
        return ToolResult(
            evidence_id="",
            tool="k8s_get_deployed_version",
            source="kubernetes",
            status="denied",
            error=f"Namespace '{ns}' is not authorized.",
        )

    try:
        core_api, apps_api = _get_k8s_clients()
        dep = apps_api.read_namespaced_deployment(deployment_name, ns)
        img = (
            dep.spec.template.spec.containers[0].image
            if dep.spec.template.spec.containers
            else None
        )
        labels = dep.metadata.labels or {}

        # Read ConfigMap if exists
        cm_data = {}
        try:
            cm = core_api.read_namespaced_config_map("mlops-inference-config", ns)
            cm_data = cm.data or {}
        except Exception:
            pass

        return ToolResult(
            evidence_id="",
            tool="k8s_get_deployed_version",
            source="kubernetes",
            status="ok",
            data={
                "deployment": deployment_name,
                "namespace": ns,
                "image": img,
                "version_label": labels.get("app.kubernetes.io/version"),
                "configmap": {
                    "APP_VERSION": cm_data.get("APP_VERSION"),
                    "MODEL_RUN_ID": cm_data.get("MODEL_RUN_ID"),
                    "DATASET_VERSION": cm_data.get("DATASET_VERSION"),
                },
            },
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="k8s_get_deployed_version",
            source="kubernetes",
            status="unavailable",
            error=f"Failed to fetch deployed version: {str(e)}",
        )


@register_tool(
    name="k8s_restart_deployment",
    description="Restart a Kubernetes deployment by patching its restartedAt annotation. Requires explicit user confirmation.",
    permission="write",
    input_model=RestartDeploymentInput,
    source="kubernetes",
)
def k8s_restart_deployment(
    deployment_name: str = "mlops-inference",
    namespace: Optional[str] = None,
    **kwargs: Any,
) -> ToolResult:
    """Store restart action in confirmation store and return confirmation ID."""
    ns = namespace or K8S_NAMESPACE
    if not check_kubernetes_namespace(ns):
        return ToolResult(
            evidence_id="",
            tool="k8s_restart_deployment",
            source="kubernetes",
            status="denied",
            error=f"Namespace '{ns}' is not authorized. Approved: {APPROVED_NAMESPACES}",
        )

    def _execute():
        _, apps_api = _get_k8s_clients()
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        body = {
            "spec": {
                "template": {
                    "metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": now_str}}
                }
            }
        }
        apps_api.patch_namespaced_deployment(deployment_name, ns, body)
        return {
            "deployment": deployment_name,
            "namespace": ns,
            "restarted_at": now_str,
            "status": "restarted",
        }

    desc = f"Restart deployment '{deployment_name}' in namespace '{ns}'"
    cid = confirmation_store.store_action(
        tool_name="k8s_restart_deployment",
        arguments={"deployment_name": deployment_name, "namespace": ns},
        description=desc,
        source="kubernetes",
        callback=_execute,
    )

    return ToolResult(
        evidence_id="",
        tool="k8s_restart_deployment",
        source="kubernetes",
        status="pending_confirmation",
        data={
            "confirmation_id": cid,
            "action": desc,
            "ttl_seconds": 300,
            "instructions": f"Call POST /confirm/{cid} to authorize restart of deployment {deployment_name}.",
        },
    )


@register_tool(
    name="k8s_scale_deployment",
    description="Scale a Kubernetes deployment to desired replicas (1-5). Requires explicit user confirmation.",
    permission="write",
    input_model=ScaleDeploymentInput,
    source="kubernetes",
)
def k8s_scale_deployment(
    deployment_name: str = "mlops-inference",
    replicas: int = 2,
    namespace: Optional[str] = None,
    **kwargs: Any,
) -> ToolResult:
    """Store scale action in confirmation store with replica bound 1-5."""
    ns = namespace or K8S_NAMESPACE
    if not check_kubernetes_namespace(ns):
        return ToolResult(
            evidence_id="",
            tool="k8s_scale_deployment",
            source="kubernetes",
            status="denied",
            error=f"Namespace '{ns}' is not authorized.",
        )

    if replicas < 1 or replicas > 5:
        return ToolResult(
            evidence_id="",
            tool="k8s_scale_deployment",
            source="kubernetes",
            status="denied",
            error="Scaling replica count must be bounded between 1 and 5.",
        )

    def _execute():
        _, apps_api = _get_k8s_clients()
        body = {"spec": {"replicas": replicas}}
        apps_api.patch_namespaced_deployment_scale(deployment_name, ns, body)
        return {
            "deployment": deployment_name,
            "namespace": ns,
            "scaled_replicas": replicas,
        }

    desc = f"Scale deployment '{deployment_name}' in namespace '{ns}' to {replicas} replicas"
    cid = confirmation_store.store_action(
        tool_name="k8s_scale_deployment",
        arguments={"deployment_name": deployment_name, "replicas": replicas, "namespace": ns},
        description=desc,
        source="kubernetes",
        callback=_execute,
    )

    return ToolResult(
        evidence_id="",
        tool="k8s_scale_deployment",
        source="kubernetes",
        status="pending_confirmation",
        data={
            "confirmation_id": cid,
            "action": desc,
            "ttl_seconds": 300,
            "instructions": f"Call POST /confirm/{cid} to authorize scaling deployment {deployment_name} to {replicas}.",
        },
    )
