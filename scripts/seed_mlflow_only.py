"""Run only the MLflow seeding — DVC versions already exist."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from training.train import train_and_log

REPO_ROOT = Path(__file__).resolve().parent.parent
DVC = str(REPO_ROOT / ".venv" / "bin" / "dvc")
GIT = "git"


def run_cmd(cmd: list[str]) -> str:
    # Replace bare 'dvc' or 'git' with full paths
    resolved = []
    for tok in cmd:
        if tok == "dvc":
            resolved.append(DVC)
        elif tok == "git":
            resolved.append(GIT)
        else:
            resolved.append(tok)
    res = subprocess.run(resolved, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [warn] {' '.join(cmd)} → rc={res.returncode}: {res.stderr.strip()[:200]}")
    return res.stdout.strip()


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    exp_name = "production-training"
    try:
        exp = client.get_experiment_by_name(exp_name)
        if not exp:
            client.create_experiment(exp_name)
            print(f"Created experiment: {exp_name}")
        else:
            print(f"Using existing experiment: {exp_name} (id={exp.experiment_id})")
    except Exception as e:
        print(f"Experiment setup notice: {e}")

    run_configs = [
        {"run_number": 1, "ver": "data-v1.0", "n_est": 10, "depth": 2, "seed": 10},
        {"run_number": 2, "ver": "data-v1.0", "n_est": 25, "depth": 3, "seed": 20},
        {"run_number": 3, "ver": "data-v1.0", "n_est": 50, "depth": 4, "seed": 30},
        {"run_number": 4, "ver": "data-v1.0", "n_est": 100, "depth": 5, "seed": 40},
        {"run_number": 5, "ver": "data-v1.1", "n_est": 20, "depth": 3, "seed": 50},
        {"run_number": 6, "ver": "data-v1.1", "n_est": 40, "depth": 4, "seed": 60},
        {"run_number": 7, "ver": "data-v2.0", "n_est": 80, "depth": 6, "seed": 70, "export": True},
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
        },
        {
            "run_number": 12,
            "ver": "data-v2.0",
            "n_est": 45,
            "depth": 5,
            "seed": 120,
            "force_dirty": True,
        },
        {"run_number": 13, "ver": "data-v2.0", "n_est": 70, "depth": 5, "seed": 130},
        {"run_number": 14, "ver": "data-v2.0", "n_est": 90, "depth": 6, "seed": 140},
    ]

    champion_run_id = None
    runner_up_run_id = None

    for cfg in run_configs:
        r_num = cfg["run_number"]
        ver = cfg["ver"]
        print(f"\nRun #{r_num:02d} | {ver} | n_est={cfg['n_est']}, depth={cfg['depth']} ...")

        # Checkout correct .dvc pointer file and restore data
        run_cmd(["git", "checkout", ver, "--", "data/dataset.csv.dvc"])
        run_cmd(["dvc", "checkout", "data/dataset.csv.dvc"])

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
        print(
            f"  → run_id={rid}  f1={res['metrics'].get('f1', '?'):.4f}  gate={res['quality_gate']}"
        )

        if cfg.get("force_gate_fail"):
            client.set_tag(rid, "quality_gate", "failed")
            client.log_metric(rid, "train_f1", 0.999)
            client.log_metric(rid, "f1", 0.720)
            print("  → forced gate=failed (overfit scenario)")

        if cfg.get("force_dirty"):
            client.set_tag(rid, "dvc_dirty", "true")
            print("  → forced dvc_dirty=true")

        if r_num == 7:
            champion_run_id = rid
        if r_num == 10:
            runner_up_run_id = rid

    # Restore HEAD dataset after all runs
    run_cmd(["git", "checkout", "HEAD", "--", "data/dataset.csv.dvc"])
    run_cmd(["dvc", "checkout", "data/dataset.csv.dvc"])
    print("\nRestored HEAD dataset (data-v2.0 / 600 samples)")

    # Register models
    model_name = "classification-model"
    print(f"\nRegistering '{model_name}' in Model Registry ...")
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
            print(f"  champion → version {v1.version}  (run {champion_run_id})")

        if runner_up_run_id:
            v2 = client.create_model_version(
                name=model_name,
                source=f"runs:/{runner_up_run_id}/model",
                run_id=runner_up_run_id,
            )
            client.set_registered_model_alias(model_name, "candidate", v2.version)
            print(f"  candidate → version {v2.version}  (run {runner_up_run_id})")

    except Exception as e:
        print(f"Model registry warning: {e}")

    # Save champion run_id for ConfigMap
    champion_file = REPO_ROOT / "reports" / "evidence" / "champion_run_id.txt"
    champion_file.parent.mkdir(parents=True, exist_ok=True)
    if champion_run_id:
        champion_file.write_text(champion_run_id)
        print(f"\nChampion run_id saved → {champion_file}")

    print("\n✓ MLflow seeding complete — 14 runs logged.")


if __name__ == "__main__":
    main()
