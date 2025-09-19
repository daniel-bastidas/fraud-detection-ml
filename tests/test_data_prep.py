# tests/test_data_prep.py
import pandas as pd
import numpy as np
from pathlib import Path

from fraud_detection.data_prep import load_dataset, build_preprocess_pipeline


def test_preprocess_pipeline_minimal(tmp_path: Path):
    # Minimal mixed-type frame with target + amount kept separate
    df = pd.DataFrame(
        {
            "fecha": ["2020-03-10 10:00:00", "2020-03-11 12:00:00"],
            "monto_raw": [100.0, 57.5],
            "score": [50, 25],
            "fraude": [0, 1],
            "g": ["AR", "BR"],
            "j": ["cat_x", "cat_y"],
            "p": ["Y", "N"],
            "a": [1, 0],
            "b": [0, 1],
        }
    )
    csv = tmp_path / "mini.csv"
    df.to_csv(csv, index=False)

    # Load using the current API: X (no target/amount), y, amounts
    X, y, amounts = load_dataset(str(csv), target_col="fraude", amount_col="monto_raw")
    assert "fraude" not in X.columns
    assert "monto_raw" not in X.columns
    assert y.tolist() == [0, 1]
    assert np.allclose(amounts.values, df["monto_raw"].values)

    # Build ColumnTransformer and fit/transform
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    pre = build_preprocess_pipeline(num_cols=num_cols, cat_cols=cat_cols)

    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] > 0  # produced features
