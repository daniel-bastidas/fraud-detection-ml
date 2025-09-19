# Fraud Detection ML Challenge

Repo para el business case de **prevencion de fraude**. Enfocado en **reproducibilidad**, **metricas orientadas a negocio (ganancia)**, y una estructura **production-like**.

## Estructura
```
fraud-detection-ml/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_inference.ipynb
├── src/
│   ├── data_prep.py
│   ├── train.py
│   ├── inference.py
│   ├── calibration.py
│   └── utils.py
├── artifacts/
│   ├── scalers/
│   └── models/
├── tests/
│   ├── test_data_prep.py
│   └── test_utils.py
└── .github/workflows/ci.yml
```

## Como ejecutar
```bash
# 1) Instalar dependencias
poetry install

# 2) Lint + tests
poetry run black src/ tests/
poetry run flake8
poetry run pytest -q

# 3) Entrenar
poetry run python -m src.train --data data/raw/dataset.csv --target is_fraud --amount_col amount

# 4) Inferencia
poetry run python -m src.inference --model artifacts/models/model_best.pkl --input data/processed/sample_inference.csv --output predictions.csv
```

## Notas
- Artefactos (scalers/modelos) se versionan con timestamp.
- Metrica de evaluacion incluye **ganancia esperada** y se reportan **Precision/Recall/F1** al **umbral optimo por ganancia**.
- **Calibracion isotonica** aplicada tras el ajuste para mejorar probas/threshold.
- Por tiempo, MLflow / IaC quedan como *future work* documentado en el informe.