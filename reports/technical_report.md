# Technical Report — Production-Ready MLOps AI Agent

**Course:** MLOps & Production AI Engineering (Fall 2026)  
**Project:** Assignment 03 — Autonomous MLOps AI Agent with Multi-System Evidence-Based Reasoning  

---

## 1. Executive Summary

This report documents the design, implementation, verification, and evaluation of an autonomous, tool-augmented AI agent engineered for production MLOps operations. The agent orchestrates operations across four real platform integrations:
- **DVC & Git:** Dataset tracking, version provenance, and reproducible checkout.
- **MLflow:** Experiment tracking, metric evaluation, model registry aliases (`champion`, `candidate`), and deterministic safety ranking.
- **Apache Airflow 3:** A 5-stage sequential DAG (`validate >> preprocess >> train >> evaluate >> quality_gate`) with programmatic quality gate enforcement and JWT authentication.
- **Kubernetes:** Production deployment manifests with health and readiness probes, resource limits, ConfigMap injection, and RBAC-gated operational tools.

Crucially, the agent operates under strict **evidence-based grounding**: the LLM is prohibited from answering from memory or hallucinating facts. All claims must cite verified tool execution IDs, write operations require explicit human confirmation, and automated retraining is governed by safety policy gates.

---

## 2. Architecture and Data Flow

### 2.1 Multi-Layer System Architecture
The system follows a decoupled four-layer architecture:
1. **Interface Layer:** FastAPI REST endpoints (`POST /chat`, `POST /confirm/{id}`, `GET /health`) and an interactive terminal CLI (`agent/cli.py`).
2. **Orchestration Layer:** Gemini function-calling multi-turn execution loop (up to 8 turns), context correlation engine, and code-level Evidence Guard.
3. **Security & Permission Layer:** In-memory confirmation store with 5-minute TTL, namespace validation (`mlops-agent`), DVC command allow-list, and structured JSON logging with automated secrets redaction.
4. **Platform Integration Layer:** Direct clients for DVC (subprocess argv), MLflow (`MlflowClient`), Airflow 3 (`httpx` + JWT), and Kubernetes (`kubernetes-client`).

See [architecture.md](../architecture.md) for detailed topological schemas and interaction diagrams.

---

## 3. Tool Definitions and Security Controls

The agent exposes 25 specialized tools across the four operational domains. Each tool adheres to a standardized envelope:
```python
ToolResult(evidence_id, tool, source, status, data, error, duration_ms, request_id)
```

### Permission Model
- **Read (Unrestricted within sandbox):** Querying experiments, runs, logs, pod statuses, and DVC revisions.
- **Write (Gated):** `airflow_trigger_training_dag`, `k8s_restart_deployment`, `k8s_scale_deployment`. These actions generate a one-time cryptographic confirmation ID and refuse execution until explicitly confirmed via `POST /confirm/{id}`.
- **Forbidden:** Arbitrary shell commands, destructive DVC actions (`gc`, `push`, `remove`), writes to non-approved namespaces (e.g. `kube-system`), and unbounded replica scaling (>5).

---

## 4. Platform Evidence & Verification

All platform operations are validated against real tool outputs and saved in `reports/evidence/`:

### 4.1 DVC Provenance & Checkout Proof
- Semantic Git tags (`data-v1.0`, `data-v1.1`, `data-v2.0`) track dataset iterations.
- Reproducible checkout verified via `scripts/demo_reproduce_dataset.sh`:
  - Materializes dataset at `data-v1.0` into `.cache/dataset_checkouts/data-v1.0/dataset.csv`.
  - Calculates file MD5 hash and compares with Git-recorded `.dvc` tracking metadata.
  - Output recorded in [reports/evidence/dvc_reproduce.txt](file:///Users/taahakhan/Desktop/mlop_assignments/mlop_a3/reports/evidence/dvc_reproduce.txt).

### 4.2 MLflow Experiment Intelligence
- 14 distinct experimental runs seeded with varied hyperparameters, random seeds, and dataset versions.
- Explicit edge cases incorporated:
  - Run 11: Highest F1 but marked `quality_gate=failed` due to overfitting gap.
  - Run 12: Trained on uncommitted working copy (`dvc_dirty=true`).
  - Run 7: High F1 (0.91+) on `data-v2.0`, passing all gates, tagged as `champion` in Model Registry.
- The `mlflow_find_best_run` tool deterministically excludes runs 11 and 12, citing specific caveats and justifying Run 7 as the true candidate.

### 4.3 Airflow 3 Pipeline Execution
- The 5-stage pipeline enforces data validation before model training.
- Quality Gate: If `f1 < MIN_F1`, task raises `AirflowFailException`, marks tag `quality_gate=failed`, and blocks registry promotion.

### 4.4 Kubernetes Diagnostics
- `k8s_find_unhealthy_pods` and `k8s_get_pod_logs` detect CrashLoopBackOff and failed readiness probes, accurately identifying configuration mistakes.

---

## 5. Five Tested Scenarios

Comprehensive execution logs and JSON payloads are preserved in `reports/conversations/`:
1. **Scenario 1:** Dataset Provenance for Run 7 (`reports/conversations/scenario_1.md`). Resolves exact Git commit, DVC MD5, and reproduction command.
2. **Scenario 2:** Run Comparison & Best Pick (`reports/conversations/scenario_2.md`). Excludes dirty and failed runs, identifying Run 7 with full caveats.
3. **Scenario 3:** Training Pipeline Trigger (`reports/conversations/scenario_3.md`). Demonstrates write confirmation gating (`pending_confirmation` -> `POST /confirm` -> execution).
4. **Scenario 4:** Pod Crash Diagnostics (`reports/conversations/scenario_4.md`). Diagnoses readiness failure caused by ConfigMap drift.
5. **Scenario 5:** Mandatory Multi-Tool Correlated Scenario (`reports/conversations/scenario_5.md`). Combines MLflow, DVC, Kubernetes, and Airflow. Determines deployed model matches best candidate and safely refrains from triggering retraining.

---

## 6. Failure Investigations

Two complete failure injection experiments were executed, diagnosed by the agent, and resolved:
1. **Failure 1 (Kubernetes Config Drift):** Detailed in [reports/failures/failure_1.md](file:///Users/taahakhan/Desktop/mlop_assignments/mlop_a3/reports/failures/failure_1.md).
2. **Failure 2 (Data Corruption & Pipeline Abort):** Detailed in [reports/failures/failure_2.md](file:///Users/taahakhan/Desktop/mlop_assignments/mlop_a3/reports/failures/failure_2.md).

---

## 7. Limitations & Future Roadmap

1. **Confirmation Store Persistence:** The current confirmation store is in-memory with threading locks and 5-minute TTL. In distributed multi-replica agent deployments, this should be backed by Redis or an external state store.
2. **Multi-Tenant RBAC:** The current implementation uses an approved namespace whitelist (`APPROVED_NAMESPACES`). Enterprise environments should implement fine-grained user authentication (OIDC) mapped to specific Kubernetes ServiceAccounts.
3. **Automated Drift Detection:** Integrating continuous statistical drift checks (Evidently AI / Great Expectations) directly into the Airflow DAG.
