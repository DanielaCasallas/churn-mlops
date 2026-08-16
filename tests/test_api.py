import pytest
from fastapi.testclient import TestClient

from app.main import app

VALID_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_schema(client):
    r = client.get("/model/schema")
    assert r.status_code == 200
    body = r.json()
    assert body["target"] == "Churn"
    assert "metrics" in body
    assert "accuracy" in body["metrics"]


def test_predict_valid_payload_returns_200(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_prediction"], bool)
    assert body["threshold_used"] == 0.5


def test_predict_missing_field_returns_422(client):
    payload = VALID_PAYLOAD.copy()
    del payload["tenure"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert any(e["field"] == "tenure" for e in body["errors"])


def test_predict_invalid_category_returns_422(client):
    payload = VALID_PAYLOAD.copy()
    payload["Contract"] = "Lifetime"  # no es una categoría válida
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert any(e["field"] == "Contract" for e in body["errors"])


def test_predict_negative_tenure_returns_422(client):
    payload = VALID_PAYLOAD.copy()
    payload["tenure"] = -5
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_wrong_type_returns_422(client):
    payload = VALID_PAYLOAD.copy()
    payload["MonthlyCharges"] = "no-es-un-numero"
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_custom_threshold_changes_decision(client):
    low_threshold = VALID_PAYLOAD | {"threshold": 0.01}
    high_threshold = VALID_PAYLOAD | {"threshold": 0.99}
    r_low = client.post("/predict", json=low_threshold)
    r_high = client.post("/predict", json=high_threshold)
    assert r_low.json()["churn_prediction"] is True
    assert r_high.json()["churn_prediction"] is False


def test_predict_batch_multiple_instances(client):
    payload = {"instances": [VALID_PAYLOAD, VALID_PAYLOAD]}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 2


def test_predict_batch_empty_list_returns_422(client):
    r = client.post("/predict/batch", json={"instances": []})
    assert r.status_code == 422
