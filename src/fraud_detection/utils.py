# src/utils.py
from __future__ import annotations
import json, os, random
from pathlib import Path
from typing import Iterable, Optional, Sequence
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# --- Reproducibilidad ---
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed); random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)

# --- IO de artefactos ---
def save_joblib(obj, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); joblib.dump(obj, p)

def load_joblib(path: str | Path):
    return joblib.load(path)

def save_json(data: dict, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

# --- Features ---
def align_features(df: pd.DataFrame, ref_cols: Sequence[str]) -> pd.DataFrame:
    """Reindex exacto a ref_cols; agrega faltantes con 0 y descarta extra."""
    return df.reindex(columns=list(ref_cols), fill_value=0)

# --- Metricas ---
def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob))

def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob))

def eval_binary(y_true: np.ndarray, y_prob: np.ndarray, name: str = "") -> dict:
    y_true = np.asarray(y_true).astype(int); y_prob = np.asarray(y_prob, dtype=float)
    return {"name": name, "roc_auc": roc_auc(y_true, y_prob), "pr_auc": pr_auc(y_true, y_prob)}

# --- Desbalance ---
def compute_scale_pos_weight(y: np.ndarray) -> float:
    y = np.asarray(y).astype(int); pos = (y == 1).sum(); neg = (y == 0).sum()
    return float(neg / max(pos, 1))

def compute_sample_weight(strategy: str | float, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int); w = np.ones_like(y, dtype=float)
    if strategy == "balanced":
        w_pos = compute_scale_pos_weight(y)
    elif isinstance(strategy, (int, float)):
        w_pos = float(strategy)
    else:
        raise ValueError("strategy debe ser 'balanced' o un float")
    w[y == 1] = w_pos
    return w

# --- Negocio: ganancia con regla de APROBACION por umbral ---
def profit_total(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amount: np.ndarray,
    thr: float,
    gain_legit: float = 0.25,
    cost_fraud: float = 1.0,
) -> float:
    """
    Apruebo si p_fraude < thr.
    Ganancia = +gain_legit * monto_legit_aprobado - cost_fraud * monto_fraude_aprobado
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    amount = np.asarray(amount, dtype=float)
    approve = y_prob < thr
    legit, fraud = (y_true == 0), (y_true == 1)
    gain = gain_legit * amount[approve & legit].sum()
    loss = cost_fraud * amount[approve & fraud].sum()
    return float(gain - loss)

def find_best_threshold_by_profit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amount: np.ndarray,
    thr_grid: Optional[Iterable[float]] = None,
    gain_legit: float = 0.25,
    cost_fraud: float = 1.0,
) -> dict:
    """Busca el thr que maximiza profit_total en una grilla simple."""
    if thr_grid is None:
        thr_grid = np.linspace(0.01, 0.99, 99)
    best_thr, best_profit = None, -np.inf
    for t in thr_grid:
        p = profit_total(y_true, y_prob, amount, t, gain_legit, cost_fraud)
        if p > best_profit:
            best_thr, best_profit = float(t), float(p)
    return {"thr": best_thr, "profit": best_profit}
