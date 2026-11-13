# Failure Investigation 2 — Data Corruption & DVC Drift

## 1. Setup & Injected Fault
- **Fault Type:** Silent data corruption / schema drift (untracked modification).
- **Injection Command:** `bash scripts/inject_failure.sh break data-corrupt`
- **Mechanism:** Stripped the required `target` label column from `data/dataset.csv` without committing to DVC or Git, causing hash mismatch and validation failure.
- **Affected Component:** Local dataset `data/dataset.csv` tracked by `data/dataset.csv.dvc`.

## 2. Observed Symptoms
- `dvc status` reported `data/dataset.csv` as modified.
- Local computed MD5 hash (`4a35b0d031aedac187e371d92f0d378b`) differed from the DVC-locked hash (`aa9e5c02f7804d01c3dda0314bc45102`).
- Dataset validation (`validate_dataset()`) failed schema verification due to missing `target` column.
- Training pipeline blocked: Airflow stage 1 (`validate_dataset`) raised an immediate `AirflowFailException`, marking downstream stages `upstream_failed`.

## 3. Agent Diagnostic Transcript
The user asked: *"Check data integrity: has the dataset file changed or become corrupted compared to the DVC tracking?"*

```json
{
  "summary": "The dataset file 'data/dataset.csv' has been modified and is no longer consistent with the DVC tracking.",
  "known_facts": [
    {
      "statement": "The dataset file 'data/dataset.csv' is modified compared to the DVC tracked version.",
      "evidence_ids": ["ev-0011"]
    },
    {
      "statement": "The local MD5 hash (4a35b0d031aedac187e371d92f0d378b) does not match the tracked MD5 hash (aa9e5c02f7804d01c3dda0314bc45102).",
      "evidence_ids": ["ev-0011"]
    },
    {
      "statement": "Associated validation dependencies and reports are also affected by these modifications.",
      "evidence_ids": ["ev-0011"]
    }
  ],
  "uncertainties": [
    "The nature of the modifications (e.g., accidental corruption vs. intentional data update) is unknown."
  ],
  "recommended_actions": [
    "Investigate the changes to 'data/dataset.csv' using 'git diff' or 'dvc diff'.",
    "If the changes are intentional, commit them to DVC to update the tracking.",
    "If the changes are accidental, revert the file to the tracked version using 'dvc checkout'."
  ]
}
```

## 4. Supporting Tool Evidence Log
| Evidence ID | Tool | Source | Status | Duration (ms) |
|---|---|---|---|---|
| `ev-0011` | `dvc_check_data_changed` | `dvc` | `ok` | 410 |

Raw Evidence Details:
```json
{
  "data_changed": true,
  "local_md5": "4a35b0d031aedac187e371d92f0d378b",
  "tracked_md5": "aa9e5c02f7804d01c3dda0314bc45102",
  "dvc_status": {
    "validate": [
      {
        "changed deps": {
          "data/dataset.csv": "modified",
          "training/validate.py": "modified"
        }
      },
      {
        "changed outs": {
          ".cache/validation_report.json": "deleted"
        }
      }
    ],
    "data/dataset.csv.dvc": [
      {
        "changed outs": {
          "data/dataset.csv": "modified"
        }
      }
    ]
  }
}
```

## 5. Root Cause Analysis
The local workspace file `data/dataset.csv` was modified directly on disk. Because DVC tracks content-addressed data using MD5 checksums recorded in pointer files (`.dvc`), any in-place edit changes the cryptographic hash. The tool `dvc_check_data_changed` detected this divergence before any model could be trained on unverified data.

## 6. Resolution & Prevention
- **Resolution:** Ran `bash scripts/inject_failure.sh fix data-corrupt` which executed `dvc checkout data/dataset.csv.dvc`, cleanly restoring the verified byte-for-byte dataset from the DVC cache. Subsequent `dvc status` confirmed: *"Data and pipelines are up to date."*
- **Prevention:** Enforce pre-commit Git hooks running `dvc status` to prevent committing without tracking data changes; run DVC integrity verification in CI prior to running training pipelines.
