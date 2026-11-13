"""Dataset validation module checking schema, null ratios, label balance, and minimum rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def validate_dataset(
    file_path: str | Path,
    min_rows: int = 50,
    max_null_ratio: float = 0.05,
    target_column: str = "target",
) -> dict[str, Any]:
    """Validate tabular dataset integrity and quality standards."""
    p = Path(file_path)
    if not p.is_file():
        return {
            "valid": False,
            "error": f"File does not exist: {file_path}",
            "row_count": 0,
        }

    df = pd.read_csv(p)
    rows, cols = df.shape

    errors = []
    if rows < min_rows:
        errors.append(f"Row count {rows} below required minimum of {min_rows}")

    if target_column not in df.columns:
        errors.append(f"Missing required target column '{target_column}'")

    null_counts = df.isnull().sum()
    null_ratio = float(null_counts.sum()) / (rows * cols) if rows * cols > 0 else 1.0
    if null_ratio > max_null_ratio:
        errors.append(f"Null ratio {null_ratio:.4f} exceeds threshold {max_null_ratio}")

    class_distribution = {}
    if target_column in df.columns:
        class_counts = df[target_column].value_counts().to_dict()
        class_distribution = {str(k): int(v) for k, v in class_counts.items()}
        # Check label balance
        min_class_ratio = min(class_counts.values()) / rows if rows > 0 else 0
        if min_class_ratio < 0.1:
            errors.append(f"Severe class imbalance: minority class is only {min_class_ratio:.2%}")

    report = {
        "valid": len(errors) == 0,
        "errors": errors,
        "row_count": rows,
        "column_count": cols,
        "columns": list(df.columns),
        "null_ratio": round(null_ratio, 4),
        "class_distribution": class_distribution,
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate tabular dataset")
    parser.add_argument("--data", default="data/dataset.csv", help="Path to CSV dataset")
    parser.add_argument("--min-rows", type=int, default=50, help="Minimum rows required")
    args = parser.parse_args()

    result = validate_dataset(args.data, min_rows=args.min_rows)
    print(f"Validation Report: {result}")
    if not result["valid"]:
        sys.exit(1)
