"""
Compara 4 modelos candidatos (LogisticRegression, RandomForest, GradientBoosting,
XGBoost) sobre el mismo split y con validación cruzada de 5 folds, para justificar
con datos —no solo por preferencia a priori— la elección del modelo de producción.

Uso:
    python training/compare_models.py

Imprime una tabla comparativa y la guarda en models/model_comparison.json
"""

import json
import logging
import time
import warnings

import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from train import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURE_COLUMNS,
    MODELS_DIR,
    NUMERIC_FEATURES,
    SEED,
    load_data,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_candidates(scale_pos_weight: float) -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=SEED, class_weight="balanced", max_depth=8
        ),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=SEED, max_depth=3),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            random_state=SEED,
            max_depth=4,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        ),
    }


def main() -> None:
    df = load_data(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    candidates = build_candidates(scale_pos_weight)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results = []
    for name, clf in candidates.items():
        pipeline = Pipeline([("preprocessor", make_preprocessor()), ("model", clf)])

        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

        result = {
            "model": name,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "roc_auc_test": round(roc_auc_score(y_test, y_proba), 4),
            "roc_auc_cv_mean": round(cv_scores.mean(), 4),
            "roc_auc_cv_std": round(cv_scores.std(), 4),
            "train_time_seconds": round(train_time, 3),
        }
        results.append(result)
        logger.info("%s: %s", name, result)

    results.sort(key=lambda r: r["roc_auc_cv_mean"], reverse=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODELS_DIR / "model_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Comparación guardada en %s", output_path)
    logger.info("Ranking por ROC-AUC (validación cruzada, 5 folds): %s", [r["model"] for r in results])


if __name__ == "__main__":
    main()
