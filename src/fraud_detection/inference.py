# src/fraud_detection/inference.py
from __future__ import annotations
import argparse
from pathlib import Path

import joblib
import pandas as pd


def _load_bundle(path: str | Path) -> dict:
    obj = joblib.load(path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj
    raise ValueError(
        "Model file must be a bundle like {'pipeline': ..., 'threshold': ...}"
    )


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Inferencia por lote (usa umbral del bundle)"
    )
    parser.add_argument(
        "--model", required=True, help="Ruta al .pkl con {'pipeline','threshold'}"
    )
    parser.add_argument("--input", required=True, help="CSV con features")
    parser.add_argument("--output", default=None, help="CSV de salida (opcional)")
    parser.add_argument(
        "--threshold", type=float, default=None, help="Override del umbral (opcional)"
    )
    parser.add_argument(
        "--proba",
        action="store_true",
        help="Devolver probabilidades en lugar de labels",
    )
    args = parser.parse_args()

    bundle = _load_bundle(args.model)
    pipe = bundle["pipeline"]
    thr_bundle = float(bundle.get("threshold", 0.5))

    df = pd.read_csv(args.input)

    if args.proba:
        preds = pipe.predict_proba(df)[:, 1]
        out = pd.DataFrame({"proba_fraud": preds})
    else:
        thr = float(args.threshold) if args.threshold is not None else thr_bundle
        probs = pipe.predict_proba(df)[:, 1]
        labels = (probs >= thr).astype(int)
        out = pd.DataFrame({"pred_label": labels, "proba_fraud": probs})

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
    else:
        # print one value per line to stdout (simple)
        if "pred_label" in out.columns:
            for v in out["pred_label"].tolist():
                print(int(v))
        else:
            for p in out["proba_fraud"].tolist():
                print(float(p))


if __name__ == "__main__":
    cli()
