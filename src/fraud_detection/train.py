# src/fraud_detection/train.py
from __future__ import annotations
import argparse
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .calibration import calibrate_pipeline
from .data_prep import build_preprocess_pipeline, load_dataset


# --- Local helpers (keep utils clean) -------------------------------------------------
def versioned_filename(dirpath: str | Path, stem: str, ext: str = ".pkl") -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    p = Path(dirpath)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{stem}_{ts}{ext}"


def compute_profit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amount: np.ndarray,
    thr: float,
    gain_legit: float = 0.25,
    cost_fraud: float = 1.0,
) -> float:
    """Approve if p_fraud < thr; profit = +0.25*legit_amount_approved - 1.0*fraud_amount_approved."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    amount = np.asarray(amount, dtype=float)
    approve = y_prob < thr
    legit, fraud = (y_true == 0), (y_true == 1)
    gain = gain_legit * amount[approve & legit].sum()
    loss = cost_fraud * amount[approve & fraud].sum()
    return float(gain - loss)


def find_best_threshold_for_profit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amount: np.ndarray,
    thr_grid: np.ndarray | None = None,
    gain_legit: float = 0.25,
    cost_fraud: float = 1.0,
) -> dict:
    if thr_grid is None:
        thr_grid = np.linspace(0.01, 0.99, 99)
    best_thr, best_profit = 0.5, -np.inf
    for t in thr_grid:
        p = compute_profit(y_true, y_prob, amount, float(t), gain_legit, cost_fraud)
        if p > best_profit:
            best_thr, best_profit = float(t), float(p)
    return {"thr": best_thr, "profit": best_profit}


# --- Training ----------------------------------------------------------------------------
def train_model(
    data_path: str,
    target_col: str,
    amount_col: str,
    model_out_dir: str = "artifacts/models",
    n_trials: int = 20,
    seed: int = 42,
    calibrate: str = "isotonic",  # "isotonic" | "sigmoid" | "none"
) -> str:
    # seeds
    np.random.seed(seed)

    # data
    X, y, amounts = load_dataset(data_path, target_col, amount_col)
    X_train, X_valid, y_train, y_valid, amt_train, amt_valid = train_test_split(
        X, y, amounts, test_size=0.2, random_state=seed, stratify=y
    )

    # preprocessing
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    cat_cols = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
    pre = build_preprocess_pipeline(num_cols=num_cols, cat_cols=cat_cols)

    # Optuna objective: maximize PR-AUC on validation
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": 4,
            "tree_method": "hist",
            "random_state": seed,
        }
        model = XGBClassifier(**params)
        pipe = Pipeline(steps=[("pre", pre), ("clf", model)])
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_valid)[:, 1]
        score = average_precision_score(y_valid, proba)  # PR-AUC
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        tree_method="hist",
        random_state=seed,
    )
    pipe: Pipeline | object = Pipeline(steps=[("pre", pre), ("clf", best_model)])
    pipe.fit(X_train, y_train)

    # Optional calibration on validation set
    if calibrate.lower() in {"isotonic", "sigmoid"}:
        try:
            pipe = calibrate_pipeline(pipe, X_valid, y_valid, method=calibrate.lower())
        except Exception as e:
            print(f"[WARN] Calibration skipped: {e}")

    # Validation metrics + profit-threshold on monto_raw (amount_col)
    proba_valid = pipe.predict_proba(X_valid)[:, 1]
    roc = roc_auc_score(y_valid, proba_valid)
    pr = average_precision_score(y_valid, proba_valid)
    res = find_best_threshold_for_profit(y_valid.values, proba_valid, amt_valid.values)
    thr, best_profit = res["thr"], res["profit"]

    y_pred_thr = (proba_valid >= thr).astype(int)
    prec = precision_score(y_valid, y_pred_thr)
    rec = recall_score(y_valid, y_pred_thr)
    f1 = f1_score(y_valid, y_pred_thr)
    profit_at_thr = compute_profit(y_valid.values, proba_valid, amt_valid.values, thr)

    print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Best Thr: {thr:.3f} | Profit: {best_profit:,.2f}")
    print(f"[THRESH {thr:.3f}] Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | Profit: {profit_at_thr:,.2f}")

    # Save bundle (pipeline + threshold)
    os.makedirs(model_out_dir, exist_ok=True)
    out_path = versioned_filename(model_out_dir, "model_best", ".pkl")
    joblib.dump({"pipeline": pipe, "threshold": thr}, out_path)
    print(f"Modelo guardado en: {out_path}")
    return str(out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenamiento de modelo de fraude (PR-AUC + umbral por ganancia)")
    p.add_argument("--data", required=True, help="Ruta al CSV con los datos")
    p.add_argument("--target", required=True, help="Nombre de la columna objetivo (0/1)")
    p.add_argument("--amount_col", required=True, help="Nombre de la columna de monto (ej. monto_raw)")
    p.add_argument("--trials", type=int, default=20, help="Optuna trials")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calibrate", choices=["none", "isotonic", "sigmoid"], default="isotonic")
    return p.parse_args()


def cli() -> None:
    args = parse_args()
    train_model(
        data_path=args.data,
        target_col=args.target,
        amount_col=args.amount_col,
        n_trials=args.trials,
        seed=args.seed,
        calibrate=args.calibrate,
    )


if __name__ == "__main__":
    cli()
