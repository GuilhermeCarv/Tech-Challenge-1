"""Treinamento de Random Forest para churn binário."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .baseline_training import BaselineTrainer
from .metrics import calcular_metricas_classificacao
from .splitting import get_or_create_split_indices

LOGGER = logging.getLogger(__name__)


class RandomForestTrainer:
    """Treina um ensemble de árvores no holdout compartilhado do projeto."""

    def __init__(
        self,
        input_path: str | Path = "data/processed/telco_feature_engineered.csv",
        target: str = "Churn Value",
        test_size: float = 0.2,
        validation_size: float = 0.2,
        random_seed: int = 42,
        n_estimators: int = 300,
    ) -> None:
        self.input_path = Path(input_path)
        self.target = target
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_seed = random_seed
        self.n_estimators = n_estimators

    def criar_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        class_weight="balanced",
                        random_state=self.random_seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def executar(
        self,
        output_model: str | Path = "models/random_forest.joblib",
        output_metrics: str | Path = "results/random_forest_metrics.json",
    ) -> dict[str, Any]:
        baseline = BaselineTrainer(
            input_path=self.input_path,
            target=self.target,
            test_size=self.test_size,
            random_seed=self.random_seed,
        )
        features, target = baseline.preparar_features(baseline.carregar_dataset())
        split = get_or_create_split_indices(
            target.to_numpy(),
            test_size=self.test_size,
            validation_size=self.validation_size,
            random_seed=self.random_seed,
        )
        X_train, y_train = features.iloc[split.train], target.iloc[split.train]
        X_test, y_test = features.iloc[split.test], target.iloc[split.test]

        model = self.criar_pipeline()
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = model.predict(X_test)
        metrics = calcular_metricas_classificacao(
            y_test.to_numpy(), predictions, probabilities
        )

        model_path = Path(output_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        metrics_path = Path(output_metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "model": "RandomForestClassifier",
                    "parameters": model.named_steps["classifier"].get_params(),
                    "metrics": metrics,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        LOGGER.info("Random Forest salvo em %s; métricas: %s", model_path, metrics)
        return {"model": model, "metrics": metrics, "model_path": model_path}


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina Random Forest para churn")
    parser.add_argument("--input", type=Path, default=Path("data/processed/telco_feature_engineered.csv"))
    parser.add_argument("--output-model", type=Path, default=Path("models/random_forest.joblib"))
    parser.add_argument("--output-metrics", type=Path, default=Path("results/random_forest_metrics.json"))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parsear_argumentos()
    RandomForestTrainer(
        input_path=args.input,
        n_estimators=args.n_estimators,
        random_seed=args.random_seed,
    ).executar(args.output_model, args.output_metrics)


if __name__ == "__main__":
    main()
