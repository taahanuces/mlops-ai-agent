"""Model training script with full DVC provenance tracking and MLflow logging."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

from training.evaluate import evaluate_model
from training.preprocess import load_and_preprocess
from training.validate import validate_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_git_tag() -> str:
    try:
        res = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "data-v1.0"


def get_dvc_provenance(dataset_file: Path) -> tuple[str, bool]:
    """Retrieve tracked MD5 from .dvc file and check dirty flag."""
    dvc_path = dataset_file.with_name(dataset_file.name + ".dvc")
    md5 = "none"
    dirty = False

    if dvc_path.exists():
        try:
            import yaml

            with open(dvc_path, "r") as f:
                content = yaml.safe_load(f)
            outs = content.get("outs", [])
            if outs:
                md5 = outs[0].get("md5", "none")
        except Exception:
            pass

    if dataset_file.exists():
        hasher = hashlib.md5()
        with open(dataset_file, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        current_md5 = hasher.hexdigest()
        if md5 != "none" and current_md5 != md5:
            dirty = True

    return md5, dirty


def train_and_log(
    data_path: str = "data/dataset.csv",
    n_estimators: int = 50,
    max_depth: int | None = 5,
    seed: int = 42,
    dataset_version: str | None = None,
    run_number: int | None = None,
    min_f1: float = 0.85,
    airflow_run_id: str | None = None,
    export_path: str | None = None,
    experiment_name: str = "production-training",
) -> dict[str, Any]:
    dataset_file = REPO_ROOT / data_path

    # Step 1: Validate dataset
    val_report = validate_dataset(dataset_file)
    if not val_report["valid"]:
        raise ValueError(f"Dataset validation failed: {val_report['errors']}")

    # Step 2: Preprocess
    X_train, X_test, y_train, y_test, features = load_and_preprocess(
        dataset_file, random_state=seed
    )

    # Step 3: Train model
    start_time = time.time()
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    train_duration = round(time.time() - start_time, 4)

    # Step 4: Evaluate
    metrics = evaluate_model(clf, X_train, y_train, X_test, y_test)
    metrics["train_time_s"] = train_duration

    # Step 5: Quality Gate
    passed_gate = metrics["f1"] >= min_f1
    gate_status = "passed" if passed_gate else "failed"

    # Step 6: Provenance tags
    git_commit = get_git_commit()
    tag_ver = dataset_version or get_git_tag()
    dvc_md5, dvc_dirty = get_dvc_provenance(dataset_file)

    # MLflow tracking
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Determine run number if not provided
        r_num = run_number
        if r_num is None:
            r_num = int(time.time()) % 10000

        # Log parameters
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": str(max_depth),
                "random_seed": seed,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "features_count": len(features),
            }
        )

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log Provenance Tags (Section 6.1)
        tags = {
            "dvc_dataset_version": tag_ver,
            "git_commit": git_commit,
            "dvc_md5": dvc_md5,
            "dvc_dirty": str(dvc_dirty).lower(),
            "run_number": str(r_num),
            "quality_gate": gate_status,
        }
        if airflow_run_id:
            tags["airflow_run_id"] = airflow_run_id

        mlflow.set_tags(tags)

        # Log sklearn model artifact
        mlflow.sklearn.log_model(
            clf,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        # Save metrics artifact
        metrics_file = REPO_ROOT / ".cache" / f"metrics_{run_id}.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump({**metrics, **tags}, f, indent=2)
        mlflow.log_artifact(str(metrics_file), artifact_path="evaluation")

        # Export model locally if requested
        if export_path:
            out_p = REPO_ROOT / export_path
            out_p.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, out_p)

    return {
        "run_id": run_id,
        "run_number": r_num,
        "metrics": metrics,
        "tags": tags,
        "quality_gate": gate_status,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model and log to MLflow with DVC provenance"
    )
    parser.add_argument("--data", default="data/dataset.csv")
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--run-number", type=int, default=None)
    parser.add_argument("--min-f1", type=float, default=0.85)
    parser.add_argument("--airflow-run-id", default=None)
    parser.add_argument("--export", default=None)
    args = parser.parse_args()

    res = train_and_log(
        data_path=args.data,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        seed=args.seed,
        dataset_version=args.dataset_version,
        run_number=args.run_number,
        min_f1=args.min_f1,
        airflow_run_id=args.airflow_run_id,
        export_path=args.export,
    )
    print(f"Training completed successfully. Run: {res['run_id']} (Number: {res['run_number']})")
    print(f"Metrics: {res['metrics']}")
    print(f"Quality Gate: {res['quality_gate']}")
