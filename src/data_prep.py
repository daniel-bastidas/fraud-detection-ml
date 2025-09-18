from __future__ import annotations
from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
import numpy as np

DEFAULT_NUM_COLS: List[str] = ["amount"]
DEFAULT_CAT_COLS: List[str] = []

def build_preprocess_pipeline(
    num_cols: List[str] | None = None,
    cat_cols: List[str] | None = None,
) -> ColumnTransformer:
    if num_cols is None:
        num_cols = DEFAULT_NUM_COLS
    if cat_cols is None:
        cat_cols = DEFAULT_CAT_COLS

    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

    if not transformers:
        raise ValueError("No numeric or categorical columns provided")

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return pre

def load_dataset(path: str, target_col: str, amount_col: str):
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")
    df = pd.read_csv(path)
    if target_col not in df.columns or amount_col not in df.columns:
        raise ValueError("target_col or amount_col not found in CSV")
    y = df[target_col].astype(int)
    amounts = df[amount_col].astype(float)
    X = df.drop(columns=[target_col])
    return X, y, amounts