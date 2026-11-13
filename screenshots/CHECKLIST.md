# Screenshot & Human Evidence Checklist

This checklist defines the evidence artifacts that require manual capture or external human intervention, as detailed in `PROJECT_SPEC.md` Section 12.

| Evidence Category | System | Required Capture / Action | Status | Destination / Notes |
|---|---|---|---|---|
| **DVC Provenance** | Terminal | Terminal capture showing `dvc status`, `git tag -l "data-*"`, `dvc diff`, and execution of `scripts/demo_reproduce_dataset.sh` | [ ] Pending | `screenshots/dvc_provenance.png` & `reports/evidence/dvc_reproduce.txt` |
| **MLflow Intelligence** | MLflow UI | UI capture of experiment run table showing provenance tags (`dvc_dataset_version`, `git_commit`, `dvc_md5`, `quality_gate`), the run compare view, and registered model versions (`champion`, `candidate`) | [ ] Pending | `screenshots/mlflow_runs.png`, `screenshots/mlflow_compare.png`, `screenshots/mlflow_registry.png` |
| **Airflow 3 Pipeline** | Airflow UI | UI capture of `training_pipeline` DAG graph, successful run execution with green tasks, and deliberate failed validation/gate run with task logs | [ ] Pending | `screenshots/airflow_dag_success.png`, `screenshots/airflow_dag_failed.png` |
| **Kubernetes Diagnostics** | Kubernetes / Terminal | Terminal or dashboard capture showing 2/2 pods Ready in `mlops-agent` namespace, `kubectl describe deployment`, services, and diagnosed broken state | [ ] Pending | `screenshots/k8s_pods_ready.png`, `screenshots/k8s_failure_diagnosis.png` |
| **Agent Interactions** | Agent API / CLI | Captures of the 5 demo conversations via CLI or FastAPI Swagger UI, explicitly showcasing tool evidence citations and the write confirmation flow | [ ] Pending | `screenshots/agent_scenario_1_to_5.png` |
| **Continuous Integration** | GitHub Actions | Screenshot of passing GitHub Actions CI workflow (linting, testing, coverage) on pull request | [ ] Pending | `screenshots/github_actions_ci.png` |

## Human Prerequisites Checklist
- [ ] `GEMINI_API_KEY` configured in `.env` (for live Gemini API calls during demo; tests use mocked LLM)
- [ ] Docker Desktop running with 6–8 GB allocated memory
- [ ] GitHub repository remote connected for PR & Actions validation
