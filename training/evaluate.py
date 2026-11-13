"""Model evaluation module computing standard classification metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate trained model on train and test partitions."""
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_f1 = float(f1_score(y_train, train_preds, average="weighted"))
    test_f1 = float(f1_score(y_test, test_preds, average="weighted"))
    acc = float(accuracy_score(y_test, test_preds))
    prec = float(precision_score(y_test, test_preds, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, test_preds, average="weighted", zero_division=0))

    metrics: dict[str, float] = {
        "f1": round(test_f1, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "train_f1": round(train_f1, 4),
    }

    # Binary ROC-AUC if applicable
    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, probs))
            metrics["roc_auc"] = round(auc, 4)
        except Exception:
            pass

    return metrics
