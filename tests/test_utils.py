# tests/test_utils.py
import numpy as np

from fraud_detection.utils import (
    compute_scale_pos_weight,
    compute_sample_weight,
    profit_total,
    find_best_threshold_by_profit,
)


def test_weights_and_profit():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.7])
    m = np.array([100, 50, 80, 200, 120], dtype=float)

    spw = compute_scale_pos_weight(y)
    assert spw > 1.0

    w = compute_sample_weight("balanced", y)
    assert np.isclose(w[y == 1].mean(), spw)

    res = find_best_threshold_by_profit(y, p, m)
    assert 0.0 < res["thr"] < 1.0

    prof = profit_total(y, p, m, res["thr"])
    assert isinstance(prof, float)
