# Failure Investigation 2 — Uncommitted Data Corruption & Pipeline Abort

## 1. Setup & Injected Fault
- **Fault Type:** Schema Corruption & Uncommitted Data Drift.
- **Injection Method:** Modified `data/dataset.csv` by deleting the mandatory `target` column without creating a DVC commit.
- **Affected Component:** DVC working directory and Airflow `training_pipeline` DAG.

## 2. Observed Symptoms
- DVC tool `dvc_check_data_changed` reports `data_changed=True` and hash mismatch.
- Airflow task `validate_dataset` raises `AirflowFailException` immediately upon schema inspection.
- Downstream tasks (`preprocess_data`, `train_model`, `evaluate_and_log`, `quality_gate`) transitioned to `upstream_failed`.
- No degraded model artifact was promoted to MLflow or staged for deployment.

## 3. Agent Diagnostic Transcript
```json
{
  "summary": "Data integrity check failed: working copy data/dataset.csv is modified and corrupted. Airflow pipeline aborted at validate_dataset stage.",
  "known_facts": [
    {
      "statement": "data/dataset.csv has uncommitted modifications vs DVC tracking record",
      "evidence_ids": ["ev-0001"]
    },
    {
      "statement": "Task validate_dataset failed due to missing required column 'target'",
      "evidence_ids": ["ev-0002"]
    }
  ],
  "uncertainties": [],
  "recommended_actions": [
    "Restore clean tracked dataset version via 'dvc checkout data/dataset.csv.dvc'",
    "Rerun training pipeline once working copy is verified clean"
  ]
}
```

## 4. Root Cause Analysis
The first stage of the DAG (`validate_dataset`) validates column schemas, row counts, and null ratios. Because `target` was stripped, the validator raised an explicit failure, safely arresting execution before training corrupted weights or wasting compute resources.

## 5. Resolution & Prevention
- **Resolution:** Restored clean dataset state via `make fix-data` (`dvc checkout data/dataset.csv.dvc`).
- **Prevention:** Run CI schema validation on dataset pushes; enforce Git hooks preventing commits when DVC status is dirty.
