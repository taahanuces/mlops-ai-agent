"""Data preprocessing module for feature matrix extraction and splitting."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess(
    file_path: str | Path,
    target_column: str = "target",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load CSV dataset, separate features and target, and perform stratified train/test split."""
    df = pd.read_csv(file_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' missing from dataset")

    feature_cols = [c for c in df.columns if c != target_column]
    X = df[feature_cols].values
    y = df[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_cols
