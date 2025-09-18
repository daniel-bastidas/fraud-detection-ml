import numpy as np
from src.utils import compute_profit, labels_from_threshold, find_best_threshold_for_profit

def test_profit_metric_basic():
    y_true = [0, 1, 0, 1]   # 0=legit, 1=fraud
    y_pred = [0, 0, 0, 1]   # 0=approve, 1=reject
    amounts = [100, 200, 100, 200]
    profit = compute_profit(y_true, y_pred, amounts)
    assert profit == -150.0  # +25 + (-200) + 25 + 0 = -150

def test_labels_from_threshold():
    probs = [0.1, 0.6, 0.9]
    labels = labels_from_threshold(probs, 0.5)
    assert labels.tolist() == [0, 1, 1]

def test_best_threshold():
    y_true = [0, 1, 0, 1]
    probs = [0.2, 0.4, 0.1, 0.9]
    amounts = [100, 200, 100, 200]
    thr, prof = find_best_threshold_for_profit(y_true, probs, amounts)
    assert 0.0 <= thr <= 1.0