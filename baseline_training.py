"""Treino baseline de churn usando DummyClassifier e Regressão Logística."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger(__name__)


def carregar_dataset(caminho_entrada: Path) -> pd.DataFrame:
    LOGGER.info('Carregando dataset de %s', caminho_entrada)
    df = pd.read_csv(caminho_entrada)
    LOGGER.info('Dataset carregado com shape %s', df.shape)
    return df


def preparar_features(df: pd.DataFrame, alvo: str) -> tuple[pd.DataFrame, pd.Series]:
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


def criar_pipeline_logistica(seed: int) -> Pipeline:
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000, random_state=seed)),
    ])


def avaliar_modelo(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }

    if hasattr(model, 'predict_proba'):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        except Exception:
            metrics['roc_auc'] = float('nan')
    else:
        metrics['roc_auc'] = float('nan')

    return metrics


def treinar_baselines(X_train, X_test, y_train, y_test, seed: int) -> dict:
    resultados: dict = {}

    dummy = DummyClassifier(strategy='most_frequent', random_state=seed)
    dummy.fit(X_train, y_train)
    resultados['dummy_classifier'] = avaliar_modelo(dummy, X_test, y_test)

    logistica = criar_pipeline_logistica(seed)
    logistica.fit(X_train, y_train)
    resultados['logistic_regression'] = avaliar_modelo(logistica, X_test, y_test)

    return resultados


def salvar_metricas(resultados: dict, caminho_saida: Path) -> None:
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    with caminho_saida.open('w', encoding='utf-8') as arquivo:
        json.dump(resultados, arquivo, indent=2, ensure_ascii=False)
    LOGGER.info('Métricas salvas em %s', caminho_saida)


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Treino baseline com DummyClassifier e Regressão Logística para churn')
    parser.add_argument('--input', type=Path, default=Path('data/processed/telco_feature_engineered.csv'), help='Caminho para o dataset processado')
    parser.add_argument('--output-metrics', type=Path, default=Path('results/baseline_metrics.json'), help='Caminho para salvar as métricas do baseline')
    parser.add_argument('--target', type=str, default='churn_target', help='Nome da coluna target binária')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proporção do conjunto de teste')
    parser.add_argument('--random-seed', type=int, default=42, help='Semente para aleatoriedade')
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parsear_argumentos()

    df = carregar_dataset(args.input)
    X, y = preparar_features(df, args.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_seed,
    )

    LOGGER.info('Treinando baselines com %d amostras para treino e %d amostras para teste', X_train.shape[0], X_test.shape[0])
    resultados = treinar_baselines(X_train, X_test, y_train, y_test, args.random_seed)

    payload = {
        'params': {
            'input': str(args.input),
            'target': args.target,
            'test_size': args.test_size,
            'random_seed': args.random_seed,
        },
        'results': resultados,
    }
    salvar_metricas(payload, args.output_metrics)

    LOGGER.info('Resultados do DummyClassifier: %s', resultados['dummy_classifier'])
    LOGGER.info('Resultados da Regressão Logística: %s', resultados['logistic_regression'])


if __name__ == '__main__':
    main()
