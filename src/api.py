"""API FastAPI para inferência do modelo campeão de churn."""

from __future__ import annotations

import json
from pathlib import Path as FilePath
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

PROJECT_ROOT = FilePath(__file__).resolve().parents[2]

class PredictionRequest(BaseModel):
    """Features numéricas do cliente, usando os nomes do artefato do modelo."""

    features: dict[str, float] = Field(
        ...,
        min_length=1,
        description="Features do cliente no mesmo formato usado no treinamento.",
    )

 
class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str


class HealthResponse(BaseModel):
    status: str
    model_ready: bool


class PredictionService:
    """Carrega o pipeline campeão e aplica seu contrato de features."""

    def __init__(
        self,
        model_path: str | FilePath = PROJECT_ROOT / "models" / "champion_model.joblib",
        metadata_path: str | Path = PROJECT_ROOT / "models" / "champion_model_metadata.json",
    ) -> None:
        self.model_path = FilePath(model_path)
        self.metadata_path = FilePath(metadata_path)
        self.model: Any | None = None
        self.feature_columns: list[str] = []

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo campeão não encontrado: {self.model_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadados do modelo campeão não encontrados: {self.metadata_path}"
            )

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        feature_columns = metadata.get("feature_columns")
        if not isinstance(feature_columns, list) or not all(
            isinstance(column, str) for column in feature_columns
        ):
            raise ValueError("Metadados inválidos: feature_columns deve ser uma lista.")

        self.model = joblib.load(self.model_path)
        self.feature_columns = feature_columns

    def predict(self, features: dict[str, float]) -> PredictionResponse:
        if not self.is_ready:
            self.load()

        received_columns = set(features)
        expected_columns = set(self.feature_columns)
        missing_columns = sorted(expected_columns - received_columns)
        unexpected_columns = sorted(received_columns - expected_columns)
        if missing_columns or unexpected_columns:
            details: list[str] = []
            if missing_columns:
                details.append(f"features ausentes: {', '.join(missing_columns)}")
            if unexpected_columns:
                details.append(f"features desconhecidas: {', '.join(unexpected_columns)}")
            raise ValueError("; ".join(details))

        data = pd.DataFrame([features], columns=self.feature_columns)
        probability = self._positive_probability(data)
        prediction = int(probability >= 0.5)
        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=probability,
            risk_level=self._risk_level(probability),
        )

    def _positive_probability(self, data: pd.DataFrame) -> float:
        if self.model is None or not hasattr(self.model, "predict_proba"):
            raise ValueError("O modelo campeão não suporta probabilidades de predição.")
        probabilities = self.model.predict_proba(data)
        classifier = getattr(self.model, "named_steps", {}).get("classifier")
        classes = getattr(classifier, "classes_", None)
        if classes is None or 1 not in classes:
            raise ValueError("O modelo campeão não possui a classe positiva de churn.")
        positive_index = list(classes).index(1)
        return float(probabilities[0][positive_index])

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability < 0.30:
            return "low"
        if probability < 0.60:
            return "medium"
        return "high"


def create_app(service: PredictionService | None = None) -> FastAPI:
    """Cria a aplicação, permitindo injetar um serviço isolado nos testes."""
    prediction_service = service or PredictionService()
    app = FastAPI(
        title="Telco Churn API",
        version="1.0.0",
        description="API de inferência para previsão de churn de clientes.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="online", model_ready=prediction_service.is_ready)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        try:
            return prediction_service.predict(request.features)
        except (FileNotFoundError, OSError) as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error

    return app


app = create_app()
