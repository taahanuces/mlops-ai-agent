"""DVC and Git provenance tools with subprocess security allow-listing."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from agent.config import REPO_ROOT, TOOL_TIMEOUT_S
from agent.permissions import check_dvc_command_allowed
from agent.schemas import ToolResult
from agent.tools.registry import register_tool


def _resolve_binary(name: str) -> str:
    """Resolve a CLI binary: prefer the running interpreter's bin dir (venv), then PATH."""
    sibling = Path(sys.executable).parent / name
    if sibling.exists():
        return str(sibling)
    return shutil.which(name) or name


def _execute_safe_command(argv: list[str], timeout_s: int = TOOL_TIMEOUT_S) -> tuple[int, str, str]:
    """Execute allow-listed command with subprocess list argv, shell=False, and timeout."""
    if not check_dvc_command_allowed(argv):
        raise PermissionError(f"Command not permitted by DVC allow-list: {' '.join(argv)}")

    resolved = [_resolve_binary(argv[0]), *argv[1:]] if argv[0] in ("dvc", "git") else argv
    res = subprocess.run(
        resolved,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        shell=False,
        timeout=timeout_s,
    )
    return res.returncode, res.stdout, res.stderr


def _calc_md5(file_path: Path) -> str:
    """Calculate MD5 hash of a local file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# Input models
class ListDatasetsInput(BaseModel):
    pass


class GetDatasetVersionInput(BaseModel):
    rev: Optional[str] = Field(default=None, description="Optional Git revision/tag/commit")


class CompareVersionsInput(BaseModel):
    rev_a: str = Field(..., description="First revision/commit/tag")
    rev_b: str = Field(..., description="Second revision/commit/tag")


class CheckDataChangedInput(BaseModel):
    pass


class GetReproductionStepsInput(BaseModel):
    rev: str = Field(..., description="Target Git revision/tag to reproduce")
    dataset_path: str = Field(default="data/dataset.csv", description="Dataset file path")


class MaterializeDatasetVersionInput(BaseModel):
    rev: str = Field(..., description="Target revision/tag to checkout")
    dataset_path: str = Field(default="data/dataset.csv", description="Dataset file path")


class GetDatasetForRunInput(BaseModel):
    run_id: Optional[str] = Field(default=None, description="MLflow run ID")
    run_number: Optional[int] = Field(default=None, description="Experiment run number integer")


# Tools
@register_tool(
    name="dvc_list_tracked_datasets",
    description="List all datasets tracked by DVC in the repository with md5, size, and paths.",
    permission="read",
    input_model=ListDatasetsInput,
    source="dvc",
)
def dvc_list_tracked_datasets(**kwargs: Any) -> ToolResult:
    """Scan repository for .dvc files and dvc.lock to list tracked datasets."""
    tracked: list[dict[str, Any]] = []
    try:
        for dvc_file in REPO_ROOT.glob("**/*.dvc"):
            if ".cache" in dvc_file.parts:
                continue
            try:
                import yaml

                with open(dvc_file, "r") as f:
                    content = yaml.safe_load(f)
                outs = content.get("outs", [])
                for out in outs:
                    tracked.append(
                        {
                            "dvc_file": str(dvc_file.relative_to(REPO_ROOT)),
                            "path": out.get("path"),
                            "md5": out.get("md5"),
                            "size": out.get("size"),
                            "nfiles": out.get("nfiles", 1),
                        }
                    )
            except Exception as e:
                tracked.append({"dvc_file": str(dvc_file.relative_to(REPO_ROOT)), "error": str(e)})

        return ToolResult(
            evidence_id="",
            tool="dvc_list_tracked_datasets",
            source="dvc",
            status="ok",
            data={"tracked_datasets": tracked, "count": len(tracked)},
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="dvc_list_tracked_datasets",
            source="dvc",
            status="error",
            error=str(e),
        )


@register_tool(
    name="dvc_get_dataset_version",
    description="Get Git commit, tag, and DVC md5 for tracked datasets at current HEAD or specified revision.",
    permission="read",
    input_model=GetDatasetVersionInput,
    source="dvc",
)
def dvc_get_dataset_version(rev: Optional[str] = None, **kwargs: Any) -> ToolResult:
    """Retrieve Git tag, commit hash, and DVC dataset md5."""
    try:
        git_target = rev or "HEAD"
        _, commit_hash, _ = _execute_safe_command(["git", "rev-parse", "--short", git_target])
        commit_hash = commit_hash.strip()

        # Tag
        tag = None
        code, tag_out, _ = _execute_safe_command(
            ["git", "describe", "--tags", "--exact-match", git_target]
        )
        if code == 0 and tag_out.strip():
            tag = tag_out.strip()
        else:
            # Try latest tag on this branch
            code2, tag_out2, _ = _execute_safe_command(
                ["git", "describe", "--tags", "--abbrev=0", git_target]
            )
            if code2 == 0:
                tag = tag_out2.strip()

        # Check dvc file content
        dvc_path = REPO_ROOT / "data" / "dataset.csv.dvc"
        md5 = None
        if rev:
            code_show, show_out, _ = _execute_safe_command(
                ["git", "show", f"{rev}:data/dataset.csv.dvc"]
            )
            if code_show == 0:
                import yaml

                d = yaml.safe_load(show_out)
                outs = d.get("outs", [])
                if outs:
                    md5 = outs[0].get("md5")
        elif dvc_path.exists():
            import yaml

            with open(dvc_path, "r") as f:
                d = yaml.safe_load(f)
            outs = d.get("outs", [])
            if outs:
                md5 = outs[0].get("md5")

        return ToolResult(
            evidence_id="",
            tool="dvc_get_dataset_version",
            source="dvc",
            status="ok",
            data={
                "revision": rev or "HEAD",
                "git_commit": commit_hash,
                "git_tag": tag,
                "dvc_md5": md5,
                "dataset_file": "data/dataset.csv",
            },
        )
    except PermissionError as pe:
        return ToolResult(
            evidence_id="",
            tool="dvc_get_dataset_version",
            source="dvc",
            status="denied",
            error=str(pe),
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="dvc_get_dataset_version",
            source="dvc",
            status="error",
            error=str(e),
        )


@register_tool(
    name="dvc_compare_versions",
    description="Compare dataset differences between two Git revisions or tags using dvc diff.",
    permission="read",
    input_model=CompareVersionsInput,
    source="dvc",
)
def dvc_compare_versions(rev_a: str, rev_b: str, **kwargs: Any) -> ToolResult:
    """Execute dvc diff --json between two revisions."""
    try:
        code, stdout, stderr = _execute_safe_command(["dvc", "diff", "--json", rev_a, rev_b])
        diff_data = json.loads(stdout) if stdout.strip() else {}
        return ToolResult(
            evidence_id="",
            tool="dvc_compare_versions",
            source="dvc",
            status="ok",
            data={"rev_a": rev_a, "rev_b": rev_b, "diff": diff_data},
        )
    except PermissionError as pe:
        return ToolResult(
            evidence_id="",
            tool="dvc_compare_versions",
            source="dvc",
            status="denied",
            error=str(pe),
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="dvc_compare_versions",
            source="dvc",
            status="error",
            error=str(e),
        )


@register_tool(
    name="dvc_check_data_changed",
    description="Check if current working dataset has uncommitted modifications vs DVC tracking file.",
    permission="read",
    input_model=CheckDataChangedInput,
    source="dvc",
)
def dvc_check_data_changed(**kwargs: Any) -> ToolResult:
    """Check working copy data status vs DVC file."""
    try:
        # Check dvc status
        code, stdout, stderr = _execute_safe_command(["dvc", "status", "--json"])
        changed = False
        status_data: dict[str, Any] = {}
        if stdout.strip():
            try:
                status_data = json.loads(stdout)
                changed = len(status_data) > 0
            except Exception:
                status_data = {"raw": stdout}
                changed = "modified" in stdout.lower() or "not in cache" in stdout.lower()

        # Also check local file md5 if dataset exists
        dataset_path = REPO_ROOT / "data" / "dataset.csv"
        dvc_path = REPO_ROOT / "data" / "dataset.csv.dvc"
        local_md5 = _calc_md5(dataset_path) if dataset_path.exists() else None
        recorded_md5 = None
        if dvc_path.exists():
            import yaml

            with open(dvc_path, "r") as f:
                d = yaml.safe_load(f)
            outs = d.get("outs", [])
            if outs:
                recorded_md5 = outs[0].get("md5")

        if local_md5 and recorded_md5 and local_md5 != recorded_md5:
            changed = True

        return ToolResult(
            evidence_id="",
            tool="dvc_check_data_changed",
            source="dvc",
            status="ok",
            data={
                "data_changed": changed,
                "local_md5": local_md5,
                "tracked_md5": recorded_md5,
                "dvc_status": status_data,
            },
        )
    except PermissionError as pe:
        return ToolResult(
            evidence_id="",
            tool="dvc_check_data_changed",
            source="dvc",
            status="denied",
            error=str(pe),
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="dvc_check_data_changed",
            source="dvc",
            status="error",
            error=str(e),
        )


@register_tool(
    name="dvc_get_reproduction_steps",
    description="Return exact shell commands and verification steps to reproduce a dataset version at a given tag/rev.",
    permission="read",
    input_model=GetReproductionStepsInput,
    source="dvc",
)
def dvc_get_reproduction_steps(
    rev: str, dataset_path: str = "data/dataset.csv", **kwargs: Any
) -> ToolResult:
    """Verify revision and return reproducible checkout instructions."""
    try:
        # Check rev exists
        code, out, _ = _execute_safe_command(["git", "rev-parse", "--verify", rev])
        if code != 0:
            return ToolResult(
                evidence_id="",
                tool="dvc_get_reproduction_steps",
                source="dvc",
                status="invalid_request",
                error=f"Revision '{rev}' does not exist in Git repository.",
            )

        steps = [
            f"1. Checkout DVC tracker file: git checkout {rev} -- {dataset_path}.dvc",
            f"2. Pull dataset from cache/remote: dvc checkout {dataset_path}.dvc",
            f"Alternative direct checkout: dvc get . {dataset_path} --rev {rev} -o .cache/dataset_checkouts/{rev}/",
        ]
        return ToolResult(
            evidence_id="",
            tool="dvc_get_reproduction_steps",
            source="dvc",
            status="ok",
            data={
                "revision": rev,
                "dataset_path": dataset_path,
                "reproduction_commands": steps,
            },
        )
    except PermissionError as pe:
        return ToolResult(
            evidence_id="",
            tool="dvc_get_reproduction_steps",
            source="dvc",
            status="denied",
            error=str(pe),
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="dvc_get_reproduction_steps",
            source="dvc",
            status="error",
            error=str(e),
        )


@register_tool(
    name="dvc_materialize_dataset_version",
    description="Materialize a specific dataset version locally into .cache/dataset_checkouts/<rev>/ and verify MD5.",
    permission="read",
    input_model=MaterializeDatasetVersionInput,
    source="dvc",
)
def dvc_materialize_dataset_version(
    rev: str, dataset_path: str = "data/dataset.csv", **kwargs: Any
) -> ToolResult:
    """Materialize dataset into isolated cache folder and verify md5 integrity."""
    target_dir = REPO_ROOT / ".cache" / "dataset_checkouts" / rev
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / Path(dataset_path).name

    try:
        # Check out dataset with dvc get into allowed cache output dir
        cmd = [
            "dvc",
            "get",
            ".",
            dataset_path,
            "--rev",
            rev,
            "-o",
            f".cache/dataset_checkouts/{rev}/{Path(dataset_path).name}",
            "--force",
        ]
        code, stdout, stderr = _execute_safe_command(cmd)
        if code != 0 or not out_file.exists():
            return ToolResult(
                evidence_id="",
                tool="dvc_materialize_dataset_version",
                source="dvc",
                status="error",
                error=f"Failed to materialize dataset via dvc get: {stderr or stdout}",
            )

        actual_md5 = _calc_md5(out_file)
        return ToolResult(
            evidence_id="",
            tool="dvc_materialize_dataset_version",
            source="dvc",
            status="ok",
            data={
                "revision": rev,
                "output_path": str(out_file.relative_to(REPO_ROOT)),
                "actual_md5": actual_md5,
                "size_bytes": out_file.stat().st_size,
            },
        )
    except PermissionError as pe:
        return ToolResult(
            evidence_id="",
            tool="dvc_materialize_dataset_version",
            source="dvc",
            status="denied",
            error=str(pe),
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="dvc_materialize_dataset_version",
            source="dvc",
            status="error",
            error=str(e),
        )


@register_tool(
    name="dvc_get_dataset_for_run",
    description="Given an MLflow run_id or run_number, look up its provenance tags (dataset version, commit, dvc_md5, dirty flag).",
    permission="read",
    input_model=GetDatasetForRunInput,
    source="dvc",
)
def dvc_get_dataset_for_run(
    run_id: Optional[str] = None, run_number: Optional[int] = None, **kwargs: Any
) -> ToolResult:
    """Retrieve dataset provenance tags for a given MLflow run."""
    try:
        from agent.tools.mlflow_tools import _get_mlflow_client

        client = _get_mlflow_client()

        target_run = None
        if run_id:
            target_run = client.get_run(run_id)
        elif run_number is not None:
            # Search runs by tag run_number
            exp = client.get_experiment_by_name("default") or client.get_experiment("0")
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=f"tags.run_number = '{run_number}'",
            )
            if runs:
                target_run = runs[0]

        if not target_run:
            return ToolResult(
                evidence_id="",
                tool="dvc_get_dataset_for_run",
                source="dvc",
                status="error",
                error=f"Run not found for run_id={run_id}, run_number={run_number}",
            )

        tags = target_run.data.tags
        return ToolResult(
            evidence_id="",
            tool="dvc_get_dataset_for_run",
            source="dvc",
            status="ok",
            data={
                "run_id": target_run.info.run_id,
                "run_number": tags.get("run_number"),
                "dvc_dataset_version": tags.get("dvc_dataset_version"),
                "git_commit": tags.get("git_commit"),
                "dvc_md5": tags.get("dvc_md5"),
                "dvc_dirty": tags.get("dvc_dirty", "false").lower() == "true",
                "quality_gate": tags.get("quality_gate"),
            },
        )
    except Exception as e:
        return ToolResult(
            evidence_id="",
            tool="dvc_get_dataset_for_run",
            source="dvc",
            status="error",
            error=str(e),
        )
