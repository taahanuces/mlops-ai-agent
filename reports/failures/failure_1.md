# Failure Investigation 1 — Kubernetes Configuration Drift & Model Path Error

## 1. Setup & Injected Fault
- **Fault Type:** Configuration Drift (`MODEL_PATH` mismatch in ConfigMap).
- **Injection Method:** Patched ConfigMap `mlops-inference-config` setting `MODEL_PATH=/app/models/nonexistent.joblib` followed by deployment rollout restart.
- **Affected Component:** `mlops-inference` deployment in `mlops-agent` namespace.

## 2. Observed Symptoms
- Pods enter non-ready state (0/1 ready containers).
- Kubernetes service `mlops-inference` endpoint count drops to 0.
- Readiness probe fails with HTTP 503 (`Readiness check failed: Model file not found at /app/models/nonexistent.joblib`).
- Container restart counts increment.

## 3. Agent Diagnostic Transcript
```json
{
  "summary": "The prediction API is unavailable because pods in the mlops-agent namespace are failing readiness probes due to an invalid model file path configured in the ConfigMap.",
  "known_facts": [
    {
      "statement": "Pods are reporting 0 ready containers with readiness probe failure on /ready",
      "evidence_ids": ["ev-0001", "ev-0002"]
    },
    {
      "statement": "ConfigMap specifies MODEL_PATH=/app/models/nonexistent.joblib which does not exist",
      "evidence_ids": ["ev-0003"]
    }
  ],
  "uncertainties": [],
  "recommended_actions": [
    "Restore MODEL_PATH to /app/models/model.joblib in mlops-inference-config ConfigMap",
    "Restart deployment mlops-inference (requires confirmation)"
  ]
}
```

## 4. Root Cause Analysis
The application's readiness probe verifies that the serialized model is loaded in memory. When the ConfigMap was updated to a nonexistent model path, the container raised a `FileNotFoundError`, causing `/ready` to return 503 Service Unavailable. Kubernetes stopped routing cluster traffic to the pods.

## 5. Resolution & Prevention
- **Resolution:** Restored valid path via `make fix-k8s` and rolled deployment.
- **Prevention:** Enforce pre-deployment schema validation for ConfigMaps; adopt admission controllers or Helm values verification.
