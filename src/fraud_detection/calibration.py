# src/fraud_detection/calibration.py
from __future__ import annotations
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


def calibrate_pipeline(
    pipe: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series, method: str = "isotonic"
) -> CalibratedClassifierCV:
    """
    Calibrate an already-fitted pipeline on validation data.
    We wrap the whole pipeline to avoid leakage.
    """
    if method not in {"isotonic", "sigmoid"}:
        raise ValueError("method must be 'isotonic' or 'sigmoid'")
    calib = CalibratedClassifierCV(estimator=pipe, cv="prefit", method=method)
    calib.fit(X_valid, y_valid)
    return calib
