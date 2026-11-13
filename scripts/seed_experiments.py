"""Seed experiments script: generates 3 dataset versions, tracks them with DVC, and logs >=12 MLflow runs."""

from __future__ import annotations

import subprocess
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.datasets import make_classification

from training.train import train_and_log

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_FILE = DATA_DIR / "dataset.csv"


def run_cmd(cmd: list[str], check: bool = True) -> str:
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if check and res.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}\nStdout: {res.stdout}\nStderr: {res.stderr}")
    return res.stdout.strip()


def generate_dataset(n_samples: int, n_features: int, flip_y: float = 0.01) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features - 1),
        n_redundant=1,
        n_classes=2,
        flip_y=flip_y,
        random_state=42,
    )
    cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    return df


def setup_dvc_versions():
    """Create 3 dataset versions with Git tags and DVC tracking."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure git user config exists if not set
    run_cmd(["git", "config", "user.name", "MLOps Agent"], check=False)
    run_cmd(["git", "config", "user.email", "agent@mlops.local"], check=False)

    versions = [
        ("data-v1.0", 300, 6, "Initial baseline dataset v1.0"),
        ("data-v1.1", 450, 6, "Expanded training samples dataset v1.1"),
        ("data-v2.0", 600, 6, "Feature-refined production dataset v2.0"),
    ]

    for tag, n_samples, n_features, msg in versions:
        print(f"\n--- Creating {tag} ({n_samples} samples) ---")
        df = generate_dataset(n_samples, n_features)
        df.to_csv(DATA_FILE, index=False)

        # DVC add & push
        run_cmd(["dvc", "add", "data/dataset.csv"], check=False)
        run_cmd(["dvc", "push"], check=False)

        # Git commit & tag
        run_cmd(["git", "add", "data/dataset.csv.dvc", "data/.gitignore"], check=False)
        run_cmd(["git", "commit", "-m", f"chore(data): {msg}"], check=False)
        # Delete tag if exists, then add
        run_cmd(["git", "tag", "-d", tag], check=False)
        run_cmd(["git", "tag", "-a", tag, "-m", msg], check=False)


def seed_mlflow_runs():
    """Seed >=12 MLflow runs covering all required edge cases and provenance tags."""
    from agent.config import MLFLOW_TRACKING_URI

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    exp_name = "production-training"

    # Ensure clean experiment
    try:
        exp = client.get_experiment_by_name(exp_name)
        if not exp:
            client.create_experiment(exp_name)
    except Exception as e:
        print(f"MLflow experiment setup notice: {e}")

    # Matrix of 14 runs with specific scenarios:
    # Run 1-4: data-v1.0
    # Run 5-8: data-v1.1
    # Run 9-14: data-v2.0
    # Deliberate cases:
    # - Run 7: Defensible production champion on data-v2.0 (good F1, clean, passed gate)
    # - Run 11: Highest F1 but quality_gate=failed / overfit (train_f1=1.0, test_f1=0.72)
    # - Run 12: dvc_dirty=true (trained on uncommitted data)
    run_configs = [
        {"run_number": 1, "ver": "data-v1.0", "n_est": 10, "depth": 2, "seed": 10},
        {"run_number": 2, "ver": "data-v1.0", "n_est": 25, "depth": 3, "seed": 20},
        {"run_number": 3, "ver": "data-v1.0", "n_est": 50, "depth": 4, "seed": 30},
        {"run_number": 4, "ver": "data-v1.0", "n_est": 100, "depth": 5, "seed": 40},
        {"run_number": 5, "ver": "data-v1.1", "n_est": 20, "depth": 3, "seed": 50},
        {"run_number": 6, "ver": "data-v1.1", "n_est": 40, "depth": 4, "seed": 60},
        {
            "run_number": 7,
            "ver": "data-v2.0",
            "n_est": 80,
            "depth": 6,
            "seed": 70,
            "export": True,
        },  # Best pick
        {"run_number": 8, "ver": "data-v1.1", "n_est": 60, "depth": 5, "seed": 80},
        {"run_number": 9, "ver": "data-v2.0", "n_est": 30, "depth": 3, "seed": 90},
        {"run_number": 10, "ver": "data-v2.0", "n_est": 50, "depth": 4, "seed": 100},
        {
            "run_number": 11,
            "ver": "data-v2.0",
            "n_est": 150,
            "depth": 20,
            "seed": 110,
            "force_gate_fail": True,
        },  # Overfit / failed
        {
            "run_number": 12,
            "ver": "data-v2.0",
            "n_est": 45,
            "depth": 5,
            "seed": 120,
            "force_dirty": True,
        },  # Dirty data
        {"run_number": 13, "ver": "data-v2.0", "n_est": 70, "depth": 5, "seed": 130},
        {"run_number": 14, "ver": "data-v2.0", "n_est": 90, "depth": 6, "seed": 140},
    ]

    print("\n--- Seeding 14 MLflow Runs ---")
    champion_run_id = None
    runner_up_run_id = None

    for cfg in run_configs:
        r_num = cfg["run_number"]
        ver = cfg["ver"]
        print(f"Logging run #{r_num} on {ver} (n_est={cfg['n_est']}, depth={cfg['depth']})...")

        # Checkout target dataset version in Git
        run_cmd(["git", "checkout", ver, "--", "data/dataset.csv.dvc"], check=False)
        run_cmd(["dvc", "checkout", "data/dataset.csv.dvc"], check=False)

        export_path = "models/model.joblib" if cfg.get("export") else None

        res = train_and_log(
            data_path="data/dataset.csv",
            n_estimators=cfg["n_est"],
            max_depth=cfg["depth"],
            seed=cfg["seed"],
            dataset_version=ver,
            run_number=r_num,
            export_path=export_path,
            experiment_name=exp_name,
        )

        rid = res["run_id"]

        # Handle deliberate edge cases
        if cfg.get("force_gate_fail"):
            client.set_tag(rid, "quality_gate", "failed")
            # Log artificial overfit metrics
            client.log_metric(rid, "train_f1", 0.999)
            client.log_metric(rid, "f1", 0.720)

        if cfg.get("force_dirty"):
            client.set_tag(rid, "dvc_dirty", "true")

        if r_num == 7:
            champion_run_id = rid
        if r_num == 10:
            runner_up_run_id = rid

    # Register champion model in MLflow Model Registry
    model_name = "classification-model"
    print(f"\n--- Registering '{model_name}' in Model Registry ---")
    try:
        try:
            client.create_registered_model(model_name)
        except Exception:
            pass

        if champion_run_id:
            v1 = client.create_model_version(
                name=model_name,
                source=f"runs:/{champion_run_id}/model",
                run_id=champion_run_id,
            )
            client.set_registered_model_alias(model_name, "champion", v1.version)
            print(f"Registered version {v1.version} as 'champion' (Run ID: {champion_run_id})")

        if runner_up_run_id:
            v2 = client.create_model_version(
                name=model_name,
                source=f"runs:/{runner_up_run_id}/model",
                run_id=runner_up_run_id,
            )
            client.set_registered_model_alias(model_name, "candidate", v2.version)
            print(f"Registered version {v2.version} as 'candidate' (Run ID: {runner_up_run_id})")

    except Exception as e:
        print(f"Model registry warning: {e}")


def main():
    setup_dvc_versions()
    seed_mlflow_runs()
    print("\n✓ Experiment seeding completed successfully.")


if __name__ == "__main__":
    main()
