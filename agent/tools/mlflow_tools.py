"""MLflow tools using MlflowClient directly with robust deterministic ranking."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from agent.config import MLFLOW_TRACKING_URI
from agent.schemas import ToolResult
from agent.tools.registry import register_tool


def _get_mlflow_client():
    """Create and return an MlflowClient configured with tracking URI."""
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


# Input models
class ListExperimentsInput(BaseModel):
    pass


class ListRunsInput(BaseModel):
    experiment_id: Optional[str] = Field(default=None, description="Optional experiment ID")
    limit: int = Field(default=10, ge=1, le=100, description="Number of recent runs to return")


class GetRunDetailsInput(BaseModel):
    run_id: Optional[str] = Field(default=None, description="MLflow run ID")
    run_number: Optional[int] = Field(default=None, description="Experiment run number integer")


class FindBestRunInput(BaseModel):
    metric: str = Field(default="f1", description="Target metric name to evaluate")
    mode: Literal["max", "min"] = Field(default="max", description="Optimization direction")
    limit: int = Field(default=20, ge=1, le=100, description="Max recent runs to consider")


class CompareRunsInput(BaseModel):
    run_ids: list[str] = Field(
        ..., min_length=2, max_length=10, description="List of run IDs to compare"
    )


class GetModelVersionsInput(BaseModel):
    name: str = Field(default="classification-model", description="Registered model name")


# Tools
@register_tool(
    name="mlflow_list_experiments",
    description="List all experiments in MLflow with their IDs, names, artifact locations, and lifecycle stages.",
    permission="read",
    input_model=ListExperimentsInput,
    source="mlflow",
)
def mlflow_list_experiments(**kwargs: Any) -> ToolResult:
    """Retrieve all experiments via MlflowClient."""
    try:
        client = _get_mlflow_client()
        experiments = client.search_experiments()
        data = [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
            }
            for exp in experiments
        ]
        return ToolResult(
            evidence_id="",
            tool="mlflow_list_experiments",
            source="mlflow",
            status="ok",
            data={"experiments": data, "count": len(data)},
        )
    except TimeoutError as te:
        return ToolResult(
            evidence_id="",
            tool="mlflow_list_experiments",
            source="mlflow",
            status="timeout",
            error=f"MLflow request timed out: {te}",
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="mlflow_list_experiments",
            source="mlflow",
            status="unavailable",
            error=f"MLflow service unavailable: {str(e)}",
        )


@register_tool(
    name="mlflow_list_runs",
    description="List recent MLflow runs ordered by start time descending, with metrics, parameters, and tags.",
    permission="read",
    input_model=ListRunsInput,
    source="mlflow",
)
def mlflow_list_runs(
    experiment_id: Optional[str] = None, limit: int = 10, **kwargs: Any
) -> ToolResult:
    """List recent runs for an experiment."""
    try:
        client = _get_mlflow_client()
        exp_id = experiment_id
        if not exp_id:
            exps = client.search_experiments()
            if not exps:
                return ToolResult(
                    evidence_id="",
                    tool="mlflow_list_runs",
                    source="mlflow",
                    status="ok",
                    data={"runs": [], "count": 0},
                )
            exp_id = exps[0].experiment_id

        runs = client.search_runs(
            experiment_ids=[exp_id],
            order_by=["start_time DESC"],
            max_results=limit,
        )

        formatted = []
        for r in runs:
            formatted.append(
                {
                    "run_id": r.info.run_id,
                    "run_name": r.info.run_name,
                    "start_time": r.info.start_time,
                    "status": r.info.status,
                    "metrics": r.data.metrics,
                    "params": r.data.params,
                    "tags": {
                        k: v
                        for k, v in r.data.tags.items()
                        if not k.startswith("mlflow.") or k == "mlflow.runName"
                    },
                }
            )

        return ToolResult(
            evidence_id="",
            tool="mlflow_list_runs",
            source="mlflow",
            status="ok",
            data={"runs": formatted, "count": len(formatted)},
        )
    except TimeoutError as te:
        return ToolResult(
            evidence_id="",
            tool="mlflow_list_runs",
            source="mlflow",
            status="timeout",
            error=f"MLflow request timed out: {te}",
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="mlflow_list_runs",
            source="mlflow",
            status="unavailable",
            error=f"Failed to query MLflow runs: {str(e)}",
        )


@register_tool(
    name="mlflow_get_run_details",
    description="Retrieve comprehensive details for an MLflow run by run_id or integer run_number.",
    permission="read",
    input_model=GetRunDetailsInput,
    source="mlflow",
)
def mlflow_get_run_details(
    run_id: Optional[str] = None, run_number: Optional[int] = None, **kwargs: Any
) -> ToolResult:
    """Fetch metrics, parameters, tags, and artifact listings for a specific run."""
    try:
        client = _get_mlflow_client()
        target_run = None
        if run_id:
            target_run = client.get_run(run_id)
        elif run_number is not None:
            exps = client.search_experiments()
            exp_ids = [exp.experiment_id for exp in exps]
            runs = client.search_runs(
                experiment_ids=exp_ids,
                filter_string=f"tags.run_number = '{run_number}'",
            )
            if runs:
                target_run = runs[0]

        if not target_run:
            return ToolResult(
                evidence_id="",
                tool="mlflow_get_run_details",
                source="mlflow",
                status="invalid_request",
                error=f"No run found for run_id={run_id}, run_number={run_number}",
            )

        rid = target_run.info.run_id
        artifacts = []
        try:
            art_list = client.list_artifacts(rid)
            artifacts = [a.path for a in art_list]
        except Exception:
            artifacts = ["<could not list artifacts>"]

        return ToolResult(
            evidence_id="",
            tool="mlflow_get_run_details",
            source="mlflow",
            status="ok",
            data={
                "run_id": rid,
                "status": target_run.info.status,
                "start_time": target_run.info.start_time,
                "end_time": target_run.info.end_time,
                "metrics": target_run.data.metrics,
                "params": target_run.data.params,
                "tags": target_run.data.tags,
                "artifacts": artifacts,
            },
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="mlflow_get_run_details",
            source="mlflow",
            status="unavailable",
            error=f"Error accessing MLflow run: {str(e)}",
        )


@register_tool(
    name="mlflow_find_best_run",
    description="Find the best recent training run by metric with deterministic safety ranking and caveats.",
    permission="read",
    input_model=FindBestRunInput,
    source="mlflow",
)
def mlflow_find_best_run(
    metric: str = "f1", mode: Literal["max", "min"] = "max", limit: int = 20, **kwargs: Any
) -> ToolResult:
    """Find top run by metric while filtering/reporting dirty data and failed gates."""
    try:
        client = _get_mlflow_client()
        exps = client.search_experiments()
        if not exps:
            return ToolResult(
                evidence_id="",
                tool="mlflow_find_best_run",
                source="mlflow",
                status="ok",
                data={"best_run": None, "caveats": ["No experiments found in MLflow"]},
            )

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id for exp in exps],
            order_by=["start_time DESC"],
            max_results=limit,
        )

        if not runs:
            return ToolResult(
                evidence_id="",
                tool="mlflow_find_best_run",
                source="mlflow",
                status="ok",
                data={"best_run": None, "caveats": ["No runs found in MLflow"]},
            )

        # Check if metric exists in any run
        has_metric = any(metric in r.data.metrics for r in runs)
        if not has_metric:
            return ToolResult(
                evidence_id="",
                tool="mlflow_find_best_run",
                source="mlflow",
                status="invalid_request",
                error=f"Metric '{metric}' not found in any evaluated runs.",
            )

        valid_candidates = []
        excluded_runs = []

        for r in runs:
            m_val = r.data.metrics.get(metric)
            if m_val is None:
                continue

            gate = r.data.tags.get("quality_gate", "not_evaluated")
            dirty = r.data.tags.get("dvc_dirty", "false").lower() == "true"
            train_f1 = r.data.metrics.get("train_f1")
            test_f1 = r.data.metrics.get("f1")
            overfit = False
            if train_f1 is not None and test_f1 is not None:
                if train_f1 - test_f1 > 0.15:  # significant overfit gap
                    overfit = True

            reasons = []
            if gate == "failed":
                reasons.append("Quality gate failed")
            if dirty:
                reasons.append("Trained on uncommitted/dirty DVC dataset")
            if overfit:
                reasons.append(
                    f"Significant train/test F1 gap (train={train_f1:.3f}, test={test_f1:.3f})"
                )

            item = {
                "run_id": r.info.run_id,
                "run_number": r.data.tags.get("run_number"),
                "metric_value": m_val,
                "metrics": r.data.metrics,
                "params": r.data.params,
                "tags": r.data.tags,
                "quality_gate": gate,
                "dvc_dirty": dirty,
                "dvc_dataset_version": r.data.tags.get("dvc_dataset_version"),
                "exclusion_reasons": reasons,
            }

            if reasons:
                excluded_runs.append(item)
            else:
                valid_candidates.append(item)

        reverse = mode == "max"
        valid_candidates.sort(key=lambda x: x["metric_value"], reverse=reverse)
        excluded_runs.sort(key=lambda x: x["metric_value"], reverse=reverse)

        best_candidate = valid_candidates[0] if valid_candidates else None

        caveats = []
        if excluded_runs:
            for exc in excluded_runs:
                caveats.append(
                    f"Run {exc['run_number'] or exc['run_id']} had {metric}={exc['metric_value']:.4f} but was excluded: {', '.join(exc['exclusion_reasons'])}."
                )

        if not best_candidate and excluded_runs:
            caveats.append("All top runs violated quality gates or data provenance rules.")

        return ToolResult(
            evidence_id="",
            tool="mlflow_find_best_run",
            source="mlflow",
            status="ok",
            data={
                "metric": metric,
                "mode": mode,
                "best_run": best_candidate,
                "excluded_runs": excluded_runs,
                "caveats": caveats,
            },
        )
    except TimeoutError as te:
        return ToolResult(
            evidence_id="",
            tool="mlflow_find_best_run",
            source="mlflow",
            status="timeout",
            error=f"MLflow search timed out: {te}",
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="mlflow_find_best_run",
            source="mlflow",
            status="unavailable",
            error=f"MLflow search failed: {str(e)}",
        )


@register_tool(
    name="mlflow_compare_runs",
    description="Compare side-by-side metrics, parameters, and dataset provenance across multiple MLflow runs.",
    permission="read",
    input_model=CompareRunsInput,
    source="mlflow",
)
def mlflow_compare_runs(run_ids: list[str], **kwargs: Any) -> ToolResult:
    """Compare multiple runs side-by-side."""
    try:
        client = _get_mlflow_client()
        comparison = []
        for rid in run_ids:
            try:
                run = client.get_run(rid)
                comparison.append(
                    {
                        "run_id": rid,
                        "run_number": run.data.tags.get("run_number"),
                        "metrics": run.data.metrics,
                        "params": run.data.params,
                        "dvc_dataset_version": run.data.tags.get("dvc_dataset_version"),
                        "quality_gate": run.data.tags.get("quality_gate"),
                        "dvc_dirty": run.data.tags.get("dvc_dirty"),
                    }
                )
            except Exception as e:
                comparison.append({"run_id": rid, "error": str(e)})

        return ToolResult(
            evidence_id="",
            tool="mlflow_compare_runs",
            source="mlflow",
            status="ok",
            data={"comparison": comparison},
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="mlflow_compare_runs",
            source="mlflow",
            status="unavailable",
            error=f"Error comparing runs: {str(e)}",
        )


@register_tool(
    name="mlflow_get_model_versions",
    description="List registered model versions, current stages or aliases (champion, candidate), and source run IDs.",
    permission="read",
    input_model=GetModelVersionsInput,
    source="mlflow",
)
def mlflow_get_model_versions(name: str = "classification-model", **kwargs: Any) -> ToolResult:
    """Query MLflow Model Registry for registered versions and aliases."""
    try:
        client = _get_mlflow_client()
        versions = client.search_model_versions(f"name='{name}'")
        if not versions:
            return ToolResult(
                evidence_id="",
                tool="mlflow_get_model_versions",
                source="mlflow",
                status="ok",
                data={
                    "model_name": name,
                    "versions": [],
                    "message": "No model versions registered",
                },
            )

        res = [
            {
                "version": v.version,
                "current_stage": v.current_stage,
                "aliases": v.aliases if hasattr(v, "aliases") else [],
                "source_run_id": v.run_id,
                "status": v.status,
            }
            for v in versions
        ]
        return ToolResult(
            evidence_id="",
            tool="mlflow_get_model_versions",
            source="mlflow",
            status="ok",
            data={"model_name": name, "versions": res},
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="mlflow_get_model_versions",
            source="mlflow",
            status="unavailable",
            error=f"Model registry unavailable: {str(e)}",
        )
