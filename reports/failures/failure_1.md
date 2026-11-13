# Failure Investigation 1 — Kubernetes Configuration Drift & Model Path Error

## 1. Setup & Injected Fault
- **Fault Type:** Configuration Drift (`MODEL_PATH` mismatch in ConfigMap).
- **Injection Command:** `bash scripts/inject_failure.sh break k8s-config`
- **Mechanism:** Patched ConfigMap `mlops-inference-config` setting `MODEL_PATH=/app/models/nonexistent.joblib` and triggered a rolling restart via `kubectl rollout restart deployment/mlops-inference`.
- **Target Namespace:** `mlops-agent`

## 2. Observed Symptoms
- Pods transitioned into an unhealthy state (`0/1` containers ready).
- Pod `mlops-inference-84c74cb8db-s5z4f` reported `Running` phase but failed the readiness check `/ready`.
- HTTP 503 Service Unavailable returned by `/ready` endpoint with payload: `Configured model path does not exist: /app/models/nonexistent.joblib`.
- Kubernetes service `mlops-inference` endpoint count dropped.

## 3. Agent Diagnostic Transcript
The user asked: *"Why is my prediction API unavailable in Kubernetes?"* (Scenario 4).

```json
{
  "summary": "The prediction API is partially unavailable because one of the pods in the 'mlops-inference' deployment is failing its readiness check due to a missing model file.",
  "known_facts": [
    {
      "statement": "The 'mlops-inference' deployment has 2 desired replicas, but only 2 are currently ready, with a total of 3 pods present in the 'mlops-agent' namespace.",
      "evidence_ids": ["ev-0005", "ev-0006"]
    },
    {
      "statement": "Pod 'mlops-inference-84c74cb8db-s5z4f' is in a 'Running' phase but is not 'ready'.",
      "evidence_ids": ["ev-0006"]
    },
    {
      "statement": "Logs for the non-ready pod indicate a failure to load the model: 'Configured model path does not exist: /app/models/nonexistent.joblib'.",
      "evidence_ids": ["ev-0007"]
    }
  ],
  "uncertainties": [
    "It is unclear why this specific pod was created with an incorrect model path configuration compared to the other two healthy pods.",
    "Tool 'k8s_find_unhealthy_pods' returned status 'denied': Namespace 'default' is not authorized.",
    "Tool 'k8s_get_services' returned status 'denied': Namespace 'default' is not authorized."
  ],
  "recommended_actions": [
    "Investigate the deployment configuration or recent updates that might have introduced the incorrect model path.",
    "Consider deleting the unhealthy pod to allow the deployment controller to recreate it with the correct configuration."
  ]
}
```

## 4. Supporting Tool Evidence Log
| Evidence ID | Tool | Source | Status | Duration (ms) |
|---|---|---|---|---|
| `ev-0003` | `k8s_find_unhealthy_pods` | `kubernetes` | `denied` | 0 |
| `ev-0004` | `k8s_get_services` | `kubernetes` | `denied` | 0 |
| `ev-0005` | `k8s_list_deployments` | `kubernetes` | `ok` | 452 |
| `ev-0006` | `k8s_list_pods` | `kubernetes` | `ok` | 25 |
| `ev-0007` | `k8s_get_pod_logs` | `kubernetes` | `ok` | 33 |

## 5. Root Cause Analysis
The application startup and readiness handler loads the model artifact specified in the environment variable `MODEL_PATH` injected from ConfigMap `mlops-inference-config`. When this was altered to `/app/models/nonexistent.joblib`, the joblib loader threw a `FileNotFoundError`, preventing the container readiness probe (`/ready`) from succeeding. Kubernetes readiness controller detected the failed probe and prevented the pod from receiving traffic.

## 6. Resolution & Prevention
- **Resolution:** Restored the valid model path via `bash scripts/inject_failure.sh fix k8s-config`, which restored `MODEL_PATH=/app/models/model.joblib` and waited for a clean rollout status.
- **Prevention:** Implement a Kubernetes validating webhook or admission controller to verify that ConfigMap model path values match valid artifact registries before admitting deployment updates.
