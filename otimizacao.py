"""Otimização de hiperparâmetros do MLPClassifier com Optuna."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import optuna
from sklearn.metrics import f1_score

from .baseline_training import BaselineTrainer
from .mlp_training import MLPTrainer
from .splitting import get_or_create_split_indices

LOGGER = logging.getLogger(__name__)


class OtimizadorMLP:
    """Otimiza o MLP usando somente treino e validação do split compartilhado."""

    def __init__(
        self,
        input_path: str | Path = "data/processed/telco_feature_engineered.csv",
        n_trials: int = 30,
        timeout: float | None = None,
        random_seed: int = 42,
        output_dir: str | Path = "results/optimization",
    ) -> None:
        self.input_path = Path(input_path)
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_seed = random_seed
        self.output_dir = Path(output_dir)

        baseline = BaselineTrainer(
            input_path=self.input_path, random_seed=self.random_seed
        )
        self.features, self.target = baseline.preparar_features(
            baseline.carregar_dataset()
        )
        self.split = get_or_create_split_indices(
            self.target.to_numpy(), random_seed=self.random_seed
        )

    def objective(self, trial: optuna.trial.Trial) -> float:
        hidden_layers_text = trial.suggest_categorical(
            "hidden_layer_sizes",
            ["64", "128,64", "256,128", "128,64,32"],
        )
        hidden_layers = tuple(int(layer) for layer in hidden_layers_text.split(","))
        trainer = MLPTrainer(
            input_path=self.input_path,
            hidden_layer_sizes=hidden_layers,
            max_iter=trial.suggest_int("max_iter", 200, 600, step=100),
            random_seed=self.random_seed,
        )
        model = trainer.criar_pipeline()
        model.set_params(
            classifier__alpha=trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            classifier__learning_rate_init=trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True
            ),
            classifier__batch_size=trial.suggest_categorical(
                "batch_size", [32, 64, 128]
            ),
        )
        model.fit(
            self.features.iloc[self.split.train],
            self.target.iloc[self.split.train],
        )
        predictions = model.predict(self.features.iloc[self.split.validation])
        return float(f1_score(self.target.iloc[self.split.validation], predictions))

    def rodar(self) -> tuple[optuna.study.Study, dict[str, object]]:
        study = optuna.create_study(
            direction="maximize",
            study_name="mlp_classifier_optimization",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
        )
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        best_trial = study.best_trial
        result = {
            "params": best_trial.params,
            "f1_validation": best_trial.value,
            "number": best_trial.number,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "optuna_mlp_classifier_results.json"
        output_path.write_text(
            json.dumps(
                {
                    "best_trial": result,
                    "study_trials": [
                        {
                            "number": trial.number,
                            "value": trial.value,
                            "params": trial.params,
                            "state": trial.state.name,
                        }
                        for trial in study.trials
                    ],
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        LOGGER.info("Melhor configuração: %s", result)
        return study, result


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Otimiza o MLPClassifier com Optuna")
    parser.add_argument("--input", type=Path, default=Path("data/processed/telco_feature_engineered.csv"))
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/optimization"))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parsear_argumentos()
    OtimizadorMLP(
        input_path=args.input,
        n_trials=args.n_trials,
        timeout=args.timeout,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
    ).rodar()


if __name__ == "__main__":
    main()
