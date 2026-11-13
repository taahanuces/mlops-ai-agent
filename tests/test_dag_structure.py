"""Tests for Airflow 3 DAG structure, task ordering, and default arguments."""

from __future__ import annotations

from dags.training_pipeline import dag


def test_dag_structure():
    """Verify DAG attributes, task count, sequential dependency order, and retries."""
    assert dag.dag_id == "training_pipeline"
    assert "mlops" in dag.tags
    assert dag.default_args.get("retries", 0) >= 1

    # Task IDs
    task_map = {t.task_id: t for t in dag.tasks}
    expected_tasks = [
        "validate_dataset",
        "preprocess_data",
        "train_model",
        "evaluate_and_log",
        "quality_gate",
    ]
    for tid in expected_tasks:
        assert tid in task_map, f"Missing required task {tid}"

    # Verify dependency chain
    val_task = task_map["validate_dataset"]
    prep_task = task_map["preprocess_data"]
    train_task = task_map["train_model"]
    eval_task = task_map["evaluate_and_log"]
    gate_task = task_map["quality_gate"]

    assert prep_task in val_task.downstream_list
    assert train_task in prep_task.downstream_list
    assert eval_task in train_task.downstream_list
    assert gate_task in eval_task.downstream_list
