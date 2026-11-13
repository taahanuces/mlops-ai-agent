# Scenario 1 — Dataset Provenance for Specific Experiment

**User Request:**
> Which dataset version was used for experiment 7? Provide Git commit and reproduction commands.

## Agent Summary
Experiment 7 used dataset version 'data-v2.0' associated with Git commit 'ab6f399'.

## Known Facts (Tool-Grounded)
- Experiment 7 (run_id: 8ddbb08febcc4e62b4e7e85762681a45) used dataset version 'data-v2.0' and Git commit 'ab6f399'. `[ev-0005]`
- The reproduction commands for Git commit 'ab6f399' are: 'git checkout ab6f399 -- data/dataset.csv.dvc' followed by 'dvc checkout data/dataset.csv.dvc', or 'dvc get . data/dataset.csv --rev ab6f399 -o .cache/dataset_checkouts/ab6f399/'. `[ev-0006]`

## Uncertainties & Service Failures
- None identified.

## Recommended Actions
- None.

## Supporting Evidence Log
| Evidence ID | Tool | Source | Status | Duration (ms) |
|---|---|---|---|---|
| `ev-0005` | `mlflow_get_run_details` | `mlflow` | `ok` | 43 |
| `ev-0006` | `dvc_get_reproduction_steps` | `dvc` | `ok` | 20 |

