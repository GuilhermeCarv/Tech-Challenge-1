"""Otimização de hiperparâmetros do MLP com Optuna."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from mlp_training import MLP, MLPTrainer, ChurnDataset

LOGGER = logging.getLogger(__name__)


class OtimizadorMLP:
    def __init__(
        self,
        input_path: str | Path = 'data/processed/telco_feature_engineered.csv',
        task: str = 'binary',
        n_trials: int = 30,
        timeout: float | None = None,
        random_seed: int = 42,
        output_dir: str | Path = 'results/optimization',
    ) -> None:
        self.input_path = Path(input_path)
        self.task = task
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_seed = random_seed
        self.output_dir = Path(output_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def carregar_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.input_path)
        LOGGER.info('Dataset carregado para otimização com shape %s', df.shape)
        return df

    def _espaco_busca(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        hidden_sizes = trial.suggest_categorical(
            'hidden_sizes',
            [
                '64',
                '128,64',
                '256,128,64',
                '128,64,32',
                '256,128',
            ],
        )
        return {
            'hidden_sizes': [int(item.strip()) for item in hidden_sizes.split(',') if item.strip()],
            'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.05),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'epochs': trial.suggest_int('epochs', 20, 80),
            'patience': trial.suggest_int('patience', 5, 20),
        }

    def objective(self, trial: optuna.trial.Trial) -> float:
        params = self._espaco_busca(trial)
        df = self.carregar_dataset()
        trainer = MLPTrainer(
            input_path=self.input_path,
            task=self.task,
            batch_size=params['batch_size'],
            hidden_sizes=params['hidden_sizes'],
            dropout=params['dropout'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            epochs=params['epochs'],
            patience=params['patience'],
            random_seed=self.random_seed,
            no_cuda=False,
        )

        X, y, _, _ = trainer.preparar_base(df, self.task)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y if self.task == 'binary' else None,
            random_state=self.random_seed,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.2,
            stratify=y_train_val if self.task == 'binary' else None,
            random_state=self.random_seed,
        )

        train_loader, val_loader = trainer.criar_dataloaders(X_train, y_train, X_val, y_val, params['batch_size'])
        model = MLP(
            input_dim=X.shape[1],
            hidden_dims=params['hidden_sizes'],
            output_dim=1,
            dropout=params['dropout'],
        ).to(self.device)

        if self.task == 'binary':
            contagem = np.bincount(y_train.astype(int))
            pos_weight = float(contagem[0] / contagem[1]) if len(contagem) > 1 and contagem[1] > 0 else 1.0
            criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=self.device))
        else:
            criterio = nn.MSELoss()

        otimizador = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

        melhor_valor = float('inf') if self.task == 'score' else -float('inf')
        melhor_estado: dict[str, torch.Tensor] | None = None
        contador_sem_melhora = 0

        for epoca in range(1, params['epochs'] + 1):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                otimizador.zero_grad()
                logits = model(X_batch)
                perda = criterio(logits, y_batch)
                perda.backward()
                otimizador.step()

            metricas_val, _, _ = trainer.avaliar_modelo(model, val_loader, self.task, self.device)

            if self.task == 'binary':
                valor_atual = metricas_val['recall']
                melhor_melhora = valor_atual > melhor_valor
            else:
                valor_atual = metricas_val['rmse']
                melhor_melhora = valor_atual < melhor_valor

            if melhor_melhora:
                melhor_valor = valor_atual
                melhor_estado = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                contador_sem_melhora = 0
            else:
                contador_sem_melhora += 1

            if contador_sem_melhora >= params['patience']:
                break

        if melhor_estado is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in melhor_estado.items()})

        test_loader = DataLoader(ChurnDataset(X_test, y_test), batch_size=params['batch_size'], shuffle=False)
        metricas_teste, _, _ = trainer.avaliar_modelo(model, test_loader, self.task, self.device)

        if self.task == 'binary':
            score = metricas_teste['recall']
            return score
        return -metricas_teste['rmse']

    def rodar(self) -> tuple[optuna.study.Study, dict[str, Any]]:
        direction = 'maximize' if self.task == 'binary' else 'minimize'
        study = optuna.create_study(direction=direction, study_name=f'{self.task}_mlp_optimization', sampler=optuna.samplers.TPESampler(seed=self.random_seed))
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)

        melhor = {
            'params': study.best_trial.params,
            'value': study.best_trial.value,
            'number': study.best_trial.number,
            'state': study.best_trial.state.name,
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / f'optuna_{self.task}_results.json').open('w', encoding='utf-8') as arquivo:
            json.dump({
                'best_trial': melhor,
                'study_trials': [
                    {
                        'number': t.number,
                        'value': t.value,
                        'params': t.params,
                    }
                    for t in study.trials
                ],
            }, arquivo, indent=2, ensure_ascii=False)

        LOGGER.info('Melhor configuração para %s: %s', self.task, melhor)
        return study, melhor


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Otimização de hiperparâmetros do MLP com Optuna')
    parser.add_argument('--input', type=Path, default=Path('data/processed/telco_feature_engineered.csv'), help='Dataset processado')
    parser.add_argument('--task', type=str, default='binary', choices=['binary', 'score'], help='Tarefa a otimizar')
    parser.add_argument('--n-trials', type=int, default=30, help='Número de trials do Optuna')
    parser.add_argument('--timeout', type=float, default=None, help='Timeout em segundos')
    parser.add_argument('--random-seed', type=int, default=42, help='Semente aleatória')
    parser.add_argument('--output-dir', type=Path, default=Path('results/optimization'), help='Diretório para salvar os resultados')
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parsear_argumentos()

    otimizador = OtimizadorMLP(
        input_path=args.input,
        task=args.task,
        n_trials=args.n_trials,
        timeout=args.timeout,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
    )
    otimizador.rodar()


if __name__ == '__main__':
    main()
