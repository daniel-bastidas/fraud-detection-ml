# Fraud Detection ML

Repo para un *case* de **detección de fraude** con foco en:

- **Reproducibilidad** (Poetry + `src/` layout + seeds),
- **Métrica principal**: **PR-AUC**,
- **Decisión de negocio**: **umbral elegido por ganancia** usando `monto_raw`,
- **Pipeline** scikit-learn + XGBoost + calibración opcional.

## Estructura

```
fraud-detection-ml/
├── data/
│   ├── raw/                    # fuente (no usada directamente por los CLIs)
│   └── processed/              # train_processed.csv / val_processed.csv / test_processed.csv
├── notebooks/                  # exploración y prototipos
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_inference.ipynb
├── src/
│   └── fraud_detection/
│       ├── __init__.py
│       ├── data_prep.py        # load_dataset + ColumnTransformer
│       ├── train.py            # Optuna (PR-AUC), profit-threshold, bundle
│       ├── inference.py        # batch infer (proba/labels) desde bundle
│       ├── calibration.py      # CalibratedClassifierCV (isotonic/sigmoid)
│       └── utils.py            # IO, métricas, pesos por desbalance, profit()
├── artifacts/
│   ├── models/                 # model_best_YYYYMMDD-HHMMSS.pkl   (bundle)
│   └── predictions/            # (opcional) CSVs de inferencia
├── tests/
│   ├── test_data_prep.py
│   └── test_utils.py
├── .vscode/                    # calidad de vida (no obligatorio)
├── .flake8  .gitignore  poetry.lock  pyproject.toml
└── README.md
```

## Requisitos

- **Python 3.11**
- **Poetry**
- CSVs procesados en `data/processed/` con:
  - **target** (ej. `fraude` en {0,1})
  - **monto_raw** (monto para la función de ganancia)
  - resto: features numéricas/categóricas ya limpias

## Instalación

```bash
poetry install
poetry check
poetry run python -c "import fraud_detection as fd; print('OK:', getattr(fd, '__version__', 'dev'))"
```

## Cómo usar

### 1) Entrenamiento

- Objetivo de Optuna: **PR-AUC** sobre *hold-out* de validación.
- El umbral se selecciona por **ganancia** usando `monto_raw`.
- Se guarda un **bundle** `{'pipeline': ..., 'threshold': ...}`.

```bash
poetry run fraud-train   --data data/processed/train_processed.csv   --target fraude   --amount_col monto_raw   --trials 20   --seed 42   --calibrate isotonic     # opciones: none | isotonic | sigmoid
```

Salida típica:
```
ROC-AUC: 0.87 | PR-AUC: 0.43 | Best Thr: 0.190 | Profit: 158,259.58
Modelo guardado en: artifacts/models/model_best_YYYYMMDD-HHMMSS.pkl
```

### 2) Inferencia

Usa el **bundle** más reciente (el umbral viene adentro).

```bash
# probabilidades
poetry run fraud-infer   --model artifacts/models/model_best_YYYYMMDD-HHMMSS.pkl   --input data/processed/test_processed.csv   --proba

# etiquetas (con umbral del bundle) y guardar CSV
poetry run fraud-infer   --model artifacts/models/model_best_YYYYMMDD-HHMMSS.pkl   --input data/processed/test_processed.csv   --output artifacts/predictions/test_preds.csv

# override de umbral (opcional)
poetry run fraud-infer --model ... --input ... --threshold 0.25
```

### 3) Calidad (opcional)

```bash
poetry run pytest -q
poetry run black src tests
poetry run flake8 src tests
```

## Sanity check rápido (PowerShell)

```powershell
$DATA = 'data\processed	rain_processed.csv'
$TEST = 'data\processed	est_processed.csv'

poetry run fraud-train --data "$DATA" --target fraude --amount_col monto_raw --trials 20 --seed 42 --calibrate isotonic
$MODEL = (Get-ChildItem artifacts\models\model_best_*.pkl | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName)
poetry run fraud-infer --model "$MODEL" --input "$TEST" --proba | Select-Object -First 5
poetry run fraud-infer --model "$MODEL" --input "$TEST"      | Select-Object -First 10
```

## Decisiones de diseño

- **Métrica primaria**: **PR-AUC** (desbalance).  
  **ROC-AUC** se reporta como referencia.
- **Decisión de negocio**: umbral que **maximiza ganancia**  
  `profit = +0.25 * monto_legit_aprobado − 1.0 * monto_fraude_aprobado`  
  Aprobación si `p_fraude < thr`.
- **Preprocesamiento**: `ColumnTransformer`  
  - numéricas → `StandardScaler(with_mean=False)`  
  - categóricas → `OneHotEncoder(handle_unknown="ignore", sparse_output=True)`
- **Modelo**: `XGBClassifier(tree_method="hist")` dentro del **Pipeline**.
- **Calibración**: `CalibratedClassifierCV(..., cv="prefit")` (isotónica por defecto).
- **Bundle**: `joblib` con `pipeline` + `threshold` (evita manejar `threshold.json` aparte).
- **Reproducibilidad**: `src/` layout, Poetry, seeds fijos, tests rápidos.

## Trade-offs / Trabajo futuro

- **Encoders/mappings fijos**: en notebooks hubo *mappings* experimentales.  
  En producción usamos `OneHotEncoder(..., handle_unknown="ignore")`; **no** reutilizamos codificadores guardados. Para paridad exacta con notebooks:
  - Persistir y cargar encoders (`TargetEncoder`, `OrdinalEncoder`, etc.),
  - O fijar un contrato estricto de features.
- **Validación**: *hold-out* sencillo por tiempo → mejorar con **CV estratificada** y *early stopping*.
- **Seguimiento**: sin **MLflow/W&B** ni **Dockerfile** por alcance → agregar tracking, imágenes reproducibles y CI matrix.
- **Seguridad & PII**: dataset anonimizado; faltan controles de *drift* y *schema*.  
  → Añadir validaciones con `pandera`/`pydantic`.
- **Infra**: no hay orquestación ni *feature store*.  
  → Plantillas para batch/real-time y contratos upstream/downstream.
- **Calibración**: `cv="prefit"` muestra *FutureWarning* en sklearn≥1.6; válido hoy, migrar cuando sklearn estabilice.

## Licencia

Este proyecto usa **MIT License**. Puedes reutilizar el código con atribución.

Consulta el archivo `LICENSE` para el texto completo.
