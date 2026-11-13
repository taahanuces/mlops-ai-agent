# Scenario 5 — Mandatory Multi-System Operational Correlated Scenario

**User Request:**
> Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.

## Agent Summary
The best performing model is currently deployed in production, and no retraining is necessary at this time.

## Known Facts (Tool-Grounded)
- The best recent model (run_id: 8ddbb08febcc4e62b4e7e85762681a45) achieved an F1 score of 0.9267 and passed the quality gate. `[ev-0008]`
- The best model was trained on dataset version 'data-v2.0'. `[ev-0008]`
- The current production deployment (mlops-inference) is using the same model run_id (8ddbb08febcc4e62b4e7e85762681a45) and dataset version (data-v2.0). `[ev-0009]`

## Uncertainties & Service Failures
- None identified.

## Recommended Actions
- No action required; the production environment is already running the best-validated model.
- No evidence that retraining is currently required under production policy.

## Supporting Evidence Log
| Evidence ID | Tool | Source | Status | Duration (ms) |
|---|---|---|---|---|
| `ev-0008` | `mlflow_find_best_run` | `mlflow` | `ok` | 1057 |
| `ev-0009` | `k8s_get_deployed_version` | `kubernetes` | `ok` | 27 |
| `ev-0010` | `correlate_system_context` | `agent` | `ok` | 1 |

