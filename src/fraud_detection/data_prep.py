# src/fraud_detection/data_prep.py
from __future__ import annotations
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset(
    path: str, target_col: str, amount_col: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load a CSV and split into X (features), y (target), amounts (business amount)."""
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {path}")
    if amount_col not in df.columns:
        raise ValueError(f"Amount column '{amount_col}' not found in {path}")

    y = df[target_col].astype(int)
    amounts = df[amount_col]
    X = df.drop(columns=[target_col, amount_col])
    return X, y, amounts


def _split_cols(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num_cols, cat_cols


def build_preprocess_pipeline(
    *, num_cols: list[str], cat_cols: list[str]
) -> ColumnTransformer:
    """
    Preprocessing:
      - numeric -> StandardScaler(with_mean=False)  (sparse-safe)
      - categorical -> OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    """
    transformers: list[tuple[str, object, list[str]]] = []
    if num_cols:
        transformers.append(("num", StandardScaler(with_mean=False), num_cols))
    if cat_cols:
        # sklearn >=1.4: use sparse_output instead of deprecated/removed 'sparse'
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                cat_cols,
            )
        )
    if not transformers:
        # passthrough everything if no transformers are needed
        return ColumnTransformer([("passthrough", "passthrough", [])])
    return ColumnTransformer(transformers)
