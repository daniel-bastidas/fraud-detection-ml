from __future__ import annotations
import argparse
import pandas as pd
import joblib


def run_inference(model_path: str, input_csv: str, output_csv: str) -> None:
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    thr = bundle.get("threshold", 0.5)

    df = pd.read_csv(input_csv)
    proba = pipe.predict_proba(df)[:, 1]
    pred = (proba >= thr).astype(int)

    out = df.copy()
    out["proba_fraud"] = proba
    out["pred_label"] = pred  # 1=fraude (rechazar), 0=aprobar
    out.to_csv(output_csv, index=False)
    print(f"Predicciones guardadas en: {output_csv}")


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Inferencia con modelo de fraude")
    p.add_argument("--model", required=True, help="Ruta al .pkl con pipeline y threshold")
    p.add_argument("--input", required=True, help="CSV de entrada con features")
    p.add_argument("--output", required=True, help="CSV de salida con predicciones")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args.model, args.input, args.output)