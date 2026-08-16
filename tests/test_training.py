import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from train import DATA_PATH, FEATURE_COLUMNS, MODELS_DIR, build_pipeline, load_data  # noqa: E402


def test_load_data_drops_blank_total_charges():
    df = load_data(DATA_PATH)
    assert df["TotalCharges"].isna().sum() == 0
    assert df["Churn"].isin([0, 1]).all()


def test_pipeline_trains_and_predicts_probabilities():
    df = load_data(DATA_PATH)
    X = df[FEATURE_COLUMNS].head(200)
    y = df["Churn"].head(200)

    pipeline = build_pipeline()
    pipeline.fit(X, y)
    probs = pipeline.predict_proba(X)[:, 1]

    assert (probs >= 0).all() and (probs <= 1).all()


def test_saved_metadata_has_expected_shape():
    metadata_path = MODELS_DIR / "metadata.json"
    assert metadata_path.exists(), "Corre training/train.py antes de correr los tests"

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    assert "metrics" in metadata
    assert "accuracy" in metadata["metrics"]
    assert 0.0 <= metadata["metrics"]["accuracy"] <= 1.0
    assert metadata["seed"] == 42
