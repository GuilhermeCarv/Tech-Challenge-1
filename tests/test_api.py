import json

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.api import PredictionService, create_app


def _client_with_model(tmp_path) -> TestClient:
    model_path = tmp_path / "champion_model.joblib"
    metadata_path = tmp_path / "champion_model_metadata.json"
    features = pd.DataFrame({"monthly_charges": [50.0, 80.0]})
    target = [0, 1]
    model = Pipeline([("classifier", DummyClassifier(strategy="prior"))])
    model.fit(features, target)
    joblib.dump(model, model_path)
    metadata_path.write_text(
        json.dumps({"feature_columns": ["monthly_charges"]}),
        encoding="utf-8",
    )
    return TestClient(create_app(PredictionService(model_path, metadata_path)))


def test_health_returns_online_status():
    response = TestClient(create_app()).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_predict_returns_churn_probability(tmp_path):
    response = _client_with_model(tmp_path).post(
        "/predict", json={"features": {"monthly_charges": 75.0}}
    )

    assert response.status_code == 200
    assert response.json()["churn_prediction"] in (0, 1)
    assert 0 <= response.json()["churn_probability"] <= 1
    assert response.json()["risk_level"] in {"low", "medium", "high"}
