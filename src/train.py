from __future__ import annotations
import argparse
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import optuna

from .data_prep import build_preprocess_pipeline, load_dataset
from .utils import versioned_filename, find_best_threshold_for_profit, labels_from_threshold, compute_profit
from .calibration import calibrate_pipeline


def train_model(data_path: str, target_col: str, amount_col: str, model_out_dir: str = "artifacts/models") -> str:
    X, y, amounts = load_dataset(data_path, target_col, amount_col)

    X_train, X_valid, y_train, y_valid, amt_train, amt_valid = train_test_split(
        X, y, amounts, test_size=0.2, random_state=42, stratify=y
    )

    pre = build_preprocess_pipeline(
        num_cols=[c for c in X.columns if np.issubdtype(X[c].dtype, np.number)],
        cat_cols=[c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)],
    )

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
        }
        model = XGBClassifier(**params)
        pipe = Pipeline(steps=[("pre", pre), ("clf", model)])
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_valid)[:, 1]
        score = average_precision_score(y_valid, proba)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    best_params = study.best_params
    best_model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        tree_method="hist",
    )
    pipe = Pipeline(steps=[("pre", pre), ("clf", best_model)])
    pipe.fit(X_train, y_train)

    # Optional calibration
    try:
        pipe = calibrate_pipeline(pipe, X_valid, y_valid, method="isotonic")
    except Exception as e:
        print(f"[WARN] Calibration skipped: {e}")

    proba_valid = pipe.predict_proba(X_valid)[:, 1]
    roc = roc_auc_score(y_valid, proba_valid)
    pr = average_precision_score(y_valid, proba_valid)
    thr, best_profit = find_best_threshold_for_profit(y_valid, proba_valid, amt_valid)

    # Metrics at business-optimal threshold
    y_pred_thr = (proba_valid >= thr).astype(int)
    prec = precision_score(y_valid, y_pred_thr)
    rec = recall_score(y_valid, y_pred_thr)
    f1 = f1_score(y_valid, y_pred_thr)
    profit_at_thr = compute_profit(y_valid, y_pred_thr, amt_valid)

    print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | Best Thr: {thr:.3f} | Profit: {best_profit:,.2f}")
    print(f"[THRESHOLD {thr:.3f}] Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | Profit: {profit_at_thr:,.2f}")

    os.makedirs(model_out_dir, exist_ok=True)
    out_path = versioned_filename(os.path.join(model_out_dir, "model_best"))
    joblib.dump({"pipeline": pipe, "threshold": thr}, out_path)
    print(f"Modelo guardado en: {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenamiento de modelo de fraude")
    p.add_argument("--data", required=True, help="Ruta al CSV con los datos")
    p.add_argument("--target", required=True, help="Nombre de la columna objetivo (0/1)")
    p.add_argument("--amount_col", required=True, help="Nombre de la columna de monto")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args.data, args.target, args.amount_col)