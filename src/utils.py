from __future__ import annotations
import datetime
from typing import Iterable, Sequence
import numpy as np


def versioned_filename(prefix: str, ext: str = "pkl") -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{ext}"


def compute_profit(
    y_true: Sequence[int],
    y_pred_label: Sequence[int],
    amounts: Sequence[float],
    gain_rate_legit: float = 0.25,
    loss_rate_fraud_missed: float = 1.0,
) -> float:
    if not (len(y_true) == len(y_pred_label) == len(amounts)):
        raise ValueError("Inconsistent lengths between y_true, y_pred_label and amounts")

    profit = 0.0
    for yt, yp, amt in zip(y_true, y_pred_label, amounts):
        if yt == 0 and yp == 0:
            profit += gain_rate_legit * float(amt)
        elif yt == 1 and yp == 0:
            profit -= loss_rate_fraud_missed * float(amt)
    return float(profit)


def labels_from_threshold(y_proba: Iterable[float], threshold: float) -> np.ndarray:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    probs = np.array(list(y_proba), dtype=float)
    return (probs >= threshold).astype(int)


def find_best_threshold_for_profit(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    amounts: Sequence[float],
    grid: Iterable[float] | None = None,
) -> tuple[float, float]:
    import numpy as _np
    if grid is None:
        grid = _np.linspace(0.01, 0.99, 99)

    best_thr, best_profit = 0.5, -_np.inf
    for thr in grid:
        y_pred = labels_from_threshold(y_proba, thr)
        p = compute_profit(y_true, y_pred, amounts)
        if p > best_profit:
            best_profit, best_thr = p, thr
    return float(best_thr), float(best_profit)