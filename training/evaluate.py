"""
Vuelve a evaluar el modelo ya entrenado y serializado en models/model.joblib,
usando el mismo split (misma seed) para verificar que las métricas reportadas
en metadata.json son reproducibles.

Uso:
    python training/evaluate.py
"""

import json
import logging

import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from train import DATA_PATH, FEATURE_COLUMNS, MODELS_DIR, SEED, TARGET_COL, load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    model_path = MODELS_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No existe {model_path}. Corre training/train.py primero.")

    pipeline = joblib.load(model_path)
    df = load_data(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COL]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    logger.info("Métricas recalculadas: %s", metrics)

    metadata_path = MODELS_DIR / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            saved = json.load(f)
        logger.info("Métricas guardadas en metadata.json: %s", saved.get("metrics"))


if __name__ == "__main__":
    main()
