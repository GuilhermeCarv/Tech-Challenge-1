"""Comparação, seleção e persistência reproduzível de modelos de churn."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from .baseline_training import BaselineTrainer
from .mlp_training import MLPTrainer
from .tree_training import RandomForestTrainer
from .splitting import get_or_create_split_indices

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelComparator:
    """Compara candidatos por CV e salva o melhor modelo treinado para inferência."""

    def __init__(
        self,
        input_path: str | Path = PROJECT_ROOT / "data" / "processed" / "telco_feature_engineered.csv",
        target: str = "Churn Value",
        test_size: float = 0.2,
        validation_size: float = 0.2,
        cv_folds: int = 5,
        random_seed: int = 42,
        models_dir: str | Path = PROJECT_ROOT / "models",
        results_dir: str | Path = PROJECT_ROOT / "results",
        minimum_recall: float | None = None,
        mlp_parameters: dict[str, Any] | None = None,
    ) -> None:
        if cv_folds < 2:
            raise ValueError("cv_folds deve ser pelo menos 2.")
        if minimum_recall is not None and not 0 <= minimum_recall <= 1:
            raise ValueError("minimum_recall deve estar entre 0 e 1.")

        self.input_path = Path(input_path)
        self.target = target
        self.test_size = test_size
        self.validation_size = validation_size
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.minimum_recall = minimum_recall
        self.mlp_parameters = mlp_parameters or {}

    def criar_candidatos(self) -> dict[str, Any]:
        baseline = BaselineTrainer(
            input_path=self.input_path,
            target=self.target,
            test_size=self.test_size,
            random_seed=self.random_seed,
        )
        mlp = MLPTrainer(
            input_path=self.input_path,
            target=self.target,
            test_size=self.test_size,
            validation_size=self.validation_size,
            random_seed=self.random_seed,
        ).criar_pipeline()
        if self.mlp_parameters:
            mlp.set_params(**self._normalizar_parametros_mlp())

        return {
            "logistic_regression": baseline.criar_pipeline_logistica(self.random_seed),
            "random_forest": RandomForestTrainer(
                input_path=self.input_path,
                target=self.target,
                test_size=self.test_size,
                validation_size=self.validation_size,
                random_seed=self.random_seed,
            ).criar_pipeline(),
            "mlp_classifier": mlp,
        }

    def executar(self) -> dict[str, Any]:
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

        X_cross_validation = features.iloc[split.train]
        y_cross_validation = target.iloc[split.train]
        development_indices = sorted((*split.train.tolist(), *split.validation.tolist()))
        X_development = features.iloc[development_indices]
        y_development = target.iloc[development_indices]
        X_test = features.iloc[split.test]
        y_test = target.iloc[split.test]
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed
        )
        scoring = {"f1": "f1", "recall": "recall", "roc_auc": "roc_auc"}

        rows: list[dict[str, float | str]] = []
        candidates = self.criar_candidatos()
        for name, model in candidates.items():
            scores = cross_validate(
                model,
                X_cross_validation,
                y_cross_validation,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
            )
            rows.append(
                {
                    "model": name,
                    "cv_f1_mean": float(scores["test_f1"].mean()),
                    "cv_f1_std": float(scores["test_f1"].std()),
                    "cv_recall_mean": float(scores["test_recall"].mean()),
                    "cv_recall_std": float(scores["test_recall"].std()),
                    "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
                    "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
                }
            )

        comparison = pd.DataFrame(rows).sort_values(
            ["cv_f1_mean", "cv_roc_auc_mean"], ascending=False
        ).reset_index(drop=True)
        winner_name = self.selecionar_campeao(comparison)
        winner = candidates[winner_name]
        winner.fit(X_development, y_development)

        test_metrics = baseline.avaliar_modelo(winner, X_test, y_test)

        self.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.models_dir / "champion_model.joblib"
        metadata_path = self.models_dir / "champion_model_metadata.json"
        joblib.dump(winner, model_path)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = self.results_dir / "model_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        metadata = {
            "champion_name": winner_name,
            "selection_metric": "cv_f1_mean",
            "minimum_recall": self.minimum_recall,
            "cv_folds": self.cv_folds,
            "random_seed": self.random_seed,
            "feature_columns": features.columns.tolist(),
            "test_metrics": test_metrics,
            "comparison_path": str(comparison_path),
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        LOGGER.info(
            "Modelo campeão: %s (F1 CV: %.4f). Artefato salvo em %s",
            winner_name,
            comparison.loc[comparison["model"] == winner_name, "cv_f1_mean"].iloc[0],
            model_path,
        )
        return {
            "comparison": comparison,
            "champion_name": winner_name,
            "test_metrics": test_metrics,
            "model_path": model_path,
            "metadata_path": metadata_path,
        }

    def _normalizar_parametros_mlp(self) -> dict[str, Any]:
        valid_parameters = {
            "alpha",
            "batch_size",
            "hidden_layer_sizes",
            "learning_rate_init",
            "max_iter",
        }
        parameters: dict[str, Any] = {}
        for name, value in self.mlp_parameters.items():
            if name not in valid_parameters:
                raise ValueError(f"Parâmetro de MLP não suportado: {name}")
            if name == "hidden_layer_sizes" and isinstance(value, str):
                value = tuple(int(layer) for layer in value.split(","))
            parameters[f"classifier__{name}"] = value
        return parameters

    def selecionar_campeao(self, comparison: pd.DataFrame) -> str:
        eligible = comparison
        if self.minimum_recall is not None:
            eligible = comparison[
                comparison["cv_recall_mean"] >= self.minimum_recall
            ]
        if eligible.empty:
            raise ValueError(
                "Nenhum candidato atingiu o mínimo de recall definido para seleção."
            )
        return str(eligible.iloc[0]["model"])
