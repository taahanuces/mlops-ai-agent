# Architecture Specification — MLOps AI Agent

This document details the architectural design, security mechanisms, provenance chain, and operational flow of the Production-Ready MLOps AI Agent.

---

## 1. System Overview & Component Topology

```
+----------------------------------------------------------------------------------+
|                                User Interaction                                  |
|         FastAPI REST Interface (/chat, /confirm/{id}) / CLI Terminal             |
+----------------------------------------+-----------------------------------------+
                                         |
                                         v
+----------------------------------------------------------------------------------+
|                           MLOps AI Agent Orchestrator                            |
|  - Gemini Function-Calling Loop (Max 8 turns)                                         |
|  - In-Memory Confirmation Store (5-min TTL, Single-Use)                          |
|  - Context Correlation Engine                                                    |
|  - Evidence Guard (Grounding & Hallucination Prevention)                         |
|  - Retraining Policy Guard                                                       |
|  - Structured Redacted JSON Logging (Request ID correlation)                     |
+----+-------------------+--------------------+--------------------+---------------+
     |                   |                    |                    |
     v                   v                    v                    v
+---------+         +---------+          +---------+          +---------+
|   DVC   |         | MLflow  |          | Airflow |          |   K8s   |
| Toolset |         | Toolset |          | Toolset |          | Toolset |
+----+----+         +----+----+          +----+----+          +----+----+
     |                   |                    |                    |
     v                   v                    v                    v
[Subprocess]      [MlflowClient]        [HTTPX / JWT]     [Official K8s]
Safe Argv List    SQLite Backend        Airflow 3 API     API Client
Git & .dvc        Artifact Store        Auth & Runs       Pods, Probes,
Provenance        Registry & Runs       5-Stage DAG       Deployments
```

---

## 2. Provenance Chain & Correlation Model

The agent bridges the operational gap between training and deployment by tracing an unbroken chain of custody:

1. **Dataset Versioning (DVC & Git):**
   - Datasets are tracked via DVC (`dataset.csv.dvc`) and committed to Git with semantic tags (`data-v1.0`, `data-v1.1`, `data-v2.0`).
   - Every file change updates the file MD5 hash recorded in Git.
2. **Training & Experiment Intelligence (MLflow):**
   - `training/train.py` logs explicit provenance tags to MLflow on every run:
     - `dvc_dataset_version`: Git data tag (e.g., `data-v2.0`)
     - `git_commit`: Short Git commit SHA
     - `dvc_md5`: MD5 hash of the dataset
     - `dvc_dirty`: Boolean flag indicating if training data was modified vs. `.dvc` tracking
     - `run_number`: Integer for human queries (e.g., "experiment 7")
     - `quality_gate`: `passed` or `failed`
3. **Deployment Metadata (Kubernetes):**
   - The deployment mounts a `ConfigMap` providing `MODEL_RUN_ID`, `APP_VERSION`, and `DATASET_VERSION`.
   - The `/version` endpoint exposes these identifiers live.
4. **Context Correlation Engine:**
   - Synthesizes live deployment identifiers with MLflow registry and DVC repository tags to verify if the running pod matches the best evaluated model.

---

## 3. Security & Permission Architecture

The agent enforces a multi-tier permission matrix implemented in Python:

| Permission Level | Tools | Security Enforcement |
|---|---|---|
| **Read** | `dvc_*`, `mlflow_*`, `airflow_list_*`, `airflow_get_*`, `k8s_list_*`, `k8s_get_*`, `k8s_find_*` | Executed immediately. Namespace and command allow-lists enforced. |
| **Write** | `airflow_trigger_training_dag`, `k8s_restart_deployment`, `k8s_scale_deployment` | Returns `pending_confirmation` with a unique `confirmation_id`. Requires explicit user invocation of `POST /confirm/{id}`. Stored in memory with a 5-minute TTL and single-use consumption. |
| **Forbidden** | Arbitrary shell execution, `dvc push/gc/repro`, write operations outside `APPROVED_NAMESPACES`, unregistered DAGs | Immediately blocked and logged as `denied`. |

### Specific Isolation Mechanisms:
- **DVC Command Allow-List:** Subprocess calls require `shell=False` and must begin with strictly approved argv tokens (e.g., `["dvc", "status"]`, `["git", "describe"]`).
- **Kubernetes Namespace Guard:** All K8s queries validate that `namespace in APPROVED_NAMESPACES` prior to API execution.
- **Airflow DAG Allow-List:** Triggers restricted to `ALLOWED_DAG_IDS` (e.g., `training_pipeline`).
- **Secrets Redaction:** Log filters mask authorization headers, API keys (`sk-*`), passwords, and JWT tokens across all log levels.

---

## 4. Complete Tool Registry

| Tool Name | Domain | Permission | Description | Timeout |
|---|---|---|---|---|
| `dvc_list_tracked_datasets` | DVC | Read | List DVC tracked files with paths and MD5 hashes | 10s |
| `dvc_get_dataset_version` | DVC | Read | Retrieve Git tag, commit hash, and DVC MD5 | 10s |
| `dvc_compare_versions` | DVC | Read | Run `dvc diff` between two Git revisions | 10s |
| `dvc_check_data_changed` | DVC | Read | Verify if working copy dataset has uncommitted changes | 10s |
| `dvc_get_reproduction_steps` | DVC | Read | Return exact shell reproduction commands | 10s |
| `dvc_materialize_dataset_version` | DVC | Read | Materialize dataset to local cache and check MD5 | 15s |
| `dvc_get_dataset_for_run` | DVC | Read | Resolve provenance tags for an MLflow run | 10s |
| `mlflow_list_experiments` | MLflow | Read | List experiments and IDs | 10s |
| `mlflow_list_runs` | MLflow | Read | List recent runs with metrics and tags | 10s |
| `mlflow_get_run_details` | MLflow | Read | Fetch metrics, params, tags, and artifact list | 10s |
| `mlflow_find_best_run` | MLflow | Read | Deterministic ranking excluding dirty data and failed gates | 10s |
| `mlflow_compare_runs` | MLflow | Read | Side-by-side run comparison | 10s |
| `mlflow_get_model_versions` | MLflow | Read | Retrieve registry models, aliases, and stages | 10s |
| `airflow_list_dags` | Airflow | Read | List DAGs and paused states | 10s |
| `airflow_get_dag_runs` | Airflow | Read | Query recent DAG runs and states | 10s |
| `airflow_get_failed_tasks` | Airflow | Read | List failed/upstream_failed tasks in a DAG run | 10s |
| `airflow_get_task_logs` | Airflow | Read | Retrieve truncated execution logs for task instance | 10s |
| `airflow_trigger_training_dag` | Airflow | Write | Trigger DAG run (requires user confirmation) | 10s |
| `k8s_list_pods` | K8s | Read | List pods, readiness, and restart counts | 10s |
| `k8s_list_deployments` | K8s | Read | List deployments, replicas, and images | 10s |
| `k8s_get_pod_logs` | K8s | Read | Fetch container logs (supporting previous crashes) | 10s |
| `k8s_get_services` | K8s | Read | List services and ready endpoint counts | 10s |
| `k8s_find_unhealthy_pods` | K8s | Read | Detect CrashLoopBackOff, waiting reasons, events | 10s |
| `k8s_get_deployed_version` | K8s | Read | Fetch deployed image, ConfigMap, and pod version | 10s |
| `k8s_restart_deployment` | K8s | Write | Rollout restart deployment (requires confirmation) | 10s |
| `k8s_scale_deployment` | K8s | Write | Scale deployment between 1-5 (requires confirmation) | 10s |

---

## 5. Evidence Guard & Hallucination Defense

The agent enforces code-level verification on every LLM answer turn:
1. **Citation Verification:** Every entry in `known_facts` must reference an `evidence_id` from a successful (`status="ok"`) tool execution in that request.
2. **Uncertainty Demotion:** If the LLM generates a fact without valid evidence, the fact is stripped and demoted to `uncertainties`.
3. **Failure Propagation:** Non-OK tool statuses (`unavailable`, `timeout`, `error`) are automatically added to `uncertainties`.
4. **Complete Outage Protection:** When target external systems are down, numerical claims are purged, and the agent explicitly discloses system unavailability.
