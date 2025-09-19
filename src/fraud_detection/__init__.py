"""
Paquete principal de fraud_detection.

Expone utilidades comunes para facilitar importaciones en notebooks y tests.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

# --- Versión del paquete (si está instalado editable con Poetry, funcionará) ---
try:
    __version__ = _pkg_version("fraud-detection-ml")
except PackageNotFoundError:  # durante desarrollo sin build/install
    __version__ = "0.0.0-dev"

# --- Re-export de utilidades frecuentes ---
from .utils import (  # noqa: F401
    align_features,
    compute_sample_weight,
    compute_scale_pos_weight,
    eval_binary,
    find_best_threshold_by_profit,
    load_joblib,
    load_json,
    pr_auc,
    profit_total,
    roc_auc,
    save_joblib,
    save_json,
    set_seed,
)

__all__ = [
    "__version__",
    "set_seed",
    "save_joblib",
    "load_joblib",
    "save_json",
    "load_json",
    "align_features",
    "pr_auc",
    "roc_auc",
    "eval_binary",
    "compute_scale_pos_weight",
    "compute_sample_weight",
    "profit_total",
    "find_best_threshold_by_profit",
]
