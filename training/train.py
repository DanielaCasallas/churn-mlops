"""
Entrena un clasificador de riesgo de deserción (churn) de clientes.

Uso:
    python training/train.py

Genera en models/:
    - model.joblib        clasificador entrenado
    - preprocessor.joblib preprocesador (encoders + scaler)
    - metadata.json       features esperadas, tipos, fecha y métricas
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Configuración reproducible -------------------------------------------------
SEED = 42
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "telco_churn.csv"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

TARGET_COL = "Churn"
ID_COL = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")

    # TotalCharges llega como texto y trae filas con solo espacios en blanco
    # (clientes con tenure=0, recién ingresados). Se castea y esas filas se descartan:
    # no aportan señal real de churn todavía.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["TotalCharges"])
    dropped = before - len(df)
    if dropped:
        logger.info("Descartadas %d filas con TotalCharges vacío (tenure=0)", dropped)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Cargando datos desde %s", DATA_PATH)
    df = load_data(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    logger.info("Entrenando pipeline (preprocesador + LogisticRegression)")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # --- Métricas sobre datos NO vistos ---
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    logger.info("Métricas sobre test set: %s", metrics)

    # --- Serialización de artefactos ---
    # Se guarda el pipeline completo (preprocesador + modelo) como un solo artefacto,
    # y además el preprocesador por separado para poder inspeccionarlo si hace falta.
    joblib.dump(pipeline, MODELS_DIR / "model.joblib")
    joblib.dump(pipeline.named_steps["preprocessor"], MODELS_DIR / "preprocessor.joblib")

    metadata = {
        "model_type": "LogisticRegression",
        "trained_at": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "target": TARGET_COL,
        "metrics": metrics,
    }
    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Artefactos guardados en %s", MODELS_DIR)


if __name__ == "__main__":
    main()
