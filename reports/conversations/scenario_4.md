# Scenario 4 — Kubernetes Incident Diagnosis and Root Cause Analysis

**User Request:**
> Why is my prediction API unavailable in Kubernetes?

## Agent Summary
The prediction API is partially unavailable because one of the pods in the 'mlops-inference' deployment is failing its readiness check due to a missing model file.

## Known Facts (Tool-Grounded)
- The 'mlops-inference' deployment has 2 desired replicas, but only 2 are currently ready, with a total of 3 pods present in the 'mlops-agent' namespace. `[ev-0005, ev-0006]`
- Pod 'mlops-inference-84c74cb8db-s5z4f' is in a 'Running' phase but is not 'ready'. `[ev-0006]`
- Logs for the non-ready pod indicate a failure to load the model: 'Configured model path does not exist: /app/models/nonexistent.joblib'. `[ev-0007]`

## Uncertainties & Service Failures
- It is unclear why this specific pod was created with an incorrect model path configuration compared to the other two healthy pods.
- Tool 'k8s_find_unhealthy_pods' returned status 'denied': Namespace 'default' is not authorized.
- Tool 'k8s_get_services' returned status 'denied': Namespace 'default' is not authorized.

## Recommended Actions
- Investigate the deployment configuration or recent updates that might have introduced the incorrect model path.
- Consider deleting the unhealthy pod to allow the deployment controller to recreate it with the correct configuration.

## Supporting Evidence Log
| Evidence ID | Tool | Source | Status | Duration (ms) |
|---|---|---|---|---|
| `ev-0003` | `k8s_find_unhealthy_pods` | `kubernetes` | `denied` | 0 |
| `ev-0004` | `k8s_get_services` | `kubernetes` | `denied` | 0 |
| `ev-0005` | `k8s_list_deployments` | `kubernetes` | `ok` | 452 |
| `ev-0006` | `k8s_list_pods` | `kubernetes` | `ok` | 25 |
| `ev-0007` | `k8s_get_pod_logs` | `kubernetes` | `ok` | 33 |

