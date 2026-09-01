"""Treino baseline de churn usando DummyClassifier e Regressão Logística.

Refatorado para classe BaselineTrainer, com API amigável a notebooks.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import calcular_metricas_classificacao
from .splitting import get_or_create_split_indices

LOGGER = logging.getLogger(__name__)


class BaselineTrainer:
    def __init__(
        self,
        input_path: str | Path = 'data/processed/telco_feature_engineered.csv',
        target: str = 'Churn Value',
        test_size: float = 0.2,
        random_seed: int = 42,
    ) -> None:
        self.input_path = Path(input_path)
        self.target = target
        self.test_size = test_size
        self.random_seed = random_seed

    def carregar_dataset(self, caminho_entrada: Path | str | None = None) -> pd.DataFrame:
        caminho = Path(caminho_entrada) if caminho_entrada is not None else self.input_path
        LOGGER.info('Carregando dataset de %s', caminho)
        df = pd.read_csv(caminho)
        LOGGER.info('Dataset carregado com shape %s', df.shape)
        return df

    def preparar_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        alvo = self.target
        colunas_para_descartar = [
            'Churn Value',
            'Churn Score',
            'City',
            'Gender',
            'Senior Citizen',
            'Partner',
            'Dependents',
            'Phone Service',
            'Multiple Lines',
            'Internet Service',
            'Online Security',
            'Online Backup',
            'Device Protection',
            'Tech Support',
            'Streaming TV',
            'Streaming Movies',
            'Contract',
            'Paperless Billing',
            'Payment Method',
        ]
        colunas_para_descartar = [col for col in colunas_para_descartar if col in df.columns]
        if alvo in df.columns:
            colunas_para_descartar.append(alvo)

        y = df[alvo].astype(int)
        df = df.drop(columns=colunas_para_descartar, errors='ignore')
        if 'tenure_bucket' in df.columns:
            df = pd.get_dummies(df, columns=['tenure_bucket'], drop_first=True)

        X = df.select_dtypes(include=[np.number]).copy()

        if X.isna().any().any():
            LOGGER.info('Existem valores ausentes nas features; valores serão imputados')

        LOGGER.info('Conjunto de features preparado com shape %s', X.shape)
        return X, y

    def criar_pipeline_logistica(self, seed: int) -> Pipeline:
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000, random_state=seed)),
        ])

    def treinar_baselines(self, X_train, X_test, y_train, y_test, seed: int) -> dict:
        resultados: dict = {}

        dummy = DummyClassifier(strategy='most_frequent', random_state=seed)
        dummy.fit(X_train, y_train)
        resultados['dummy_classifier'] = self.avaliar_modelo(dummy, X_test, y_test)

        logistica = self.criar_pipeline_logistica(seed)
        logistica.fit(X_train, y_train)
        resultados['logistic_regression'] = self.avaliar_modelo(logistica, X_test, y_test)

        return resultados

    def avaliar_modelo(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_prob = None

        return calcular_metricas_classificacao(y_test.to_numpy(), y_pred, y_prob)

    def salvar_metricas(self, resultados: dict, caminho_saida: Path) -> None:
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        with caminho_saida.open('w', encoding='utf-8') as arquivo:
            json.dump(resultados, arquivo, indent=2, ensure_ascii=False)
        LOGGER.info('Métricas salvas em %s', caminho_saida)

    def executar(self, caminho_entrada: Path | str | None = None, caminho_saida: Path | str | None = None) -> dict[str, Any]:
        df = self.carregar_dataset(caminho_entrada)
        X, y = self.preparar_features(df)

        split = get_or_create_split_indices(
            y.to_numpy(),
            test_size=self.test_size,
            random_seed=self.random_seed,
        )
        X_train, y_train = X.iloc[split.train], y.iloc[split.train]
        X_test, y_test = X.iloc[split.test], y.iloc[split.test]

        LOGGER.info('Treinando baselines com %d amostras para treino e %d amostras para teste', X_train.shape[0], X_test.shape[0])
        resultados = self.treinar_baselines(X_train, X_test, y_train, y_test, self.random_seed)

        payload = {
            'params': {
                'input': str(caminho_entrada or self.input_path),
                'target': self.target,
                'test_size': self.test_size,
                'random_seed': self.random_seed,
            },
            'results': resultados,
        }

        output_path = Path(caminho_saida) if caminho_saida is not None else Path('results/baseline_metrics.json')
        self.salvar_metricas(payload, output_path)

        LOGGER.info('Resultados do DummyClassifier: %s', resultados['dummy_classifier'])
        LOGGER.info('Resultados da Regressão Logística: %s', resultados['logistic_regression'])

        return payload


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Treino baseline com DummyClassifier e Regressão Logística para churn')
    parser.add_argument('--input', type=Path, default=Path('data/processed/telco_feature_engineered.csv'), help='Caminho para o dataset processado')
    parser.add_argument('--output-metrics', type=Path, default=Path('results/baseline_metrics.json'), help='Caminho para salvar as métricas do baseline')
    parser.add_argument('--target', type=str, default='Churn Value', help='Nome da coluna target binária')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proporção do conjunto de teste')
    parser.add_argument('--random-seed', type=int, default=42, help='Semente para aleatoriedade')
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parsear_argumentos()

    trainer = BaselineTrainer(input_path=args.input, target=args.target, test_size=args.test_size, random_seed=args.random_seed)
    trainer.executar(caminho_entrada=args.input, caminho_saida=args.output_metrics)


if __name__ == '__main__':
    main()
