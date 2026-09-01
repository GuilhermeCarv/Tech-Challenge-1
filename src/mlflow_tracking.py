"""Registro de experimentos de churn no MLflow."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow

LOGGER = logging.getLogger(__name__)


def configurar_experimento(nome_experimento: str) -> str:
    mlflow.set_experiment(nome_experimento)
    experimento = mlflow.get_experiment_by_name(nome_experimento)
    if experimento is None:
        raise RuntimeError(f'Falha ao criar ou recuperar experimento {nome_experimento}')
    LOGGER.info('Experimento MLflow configurado: %s (id=%s)', nome_experimento, experimento.experiment_id)
    return experimento.experiment_id


def carregar_metricas(caminho_metricas: Path) -> dict:
    with caminho_metricas.open('r', encoding='utf-8') as arquivo:
        return json.load(arquivo)


def registrar_metricas(experimento: str, nome_execucao: str, metricas: dict, caminho_metricas: Path) -> None:
    configurar_experimento(experimento)
    with mlflow.start_run(run_name=nome_execucao) as execucao:
        params = metricas.get('params', {})
        mlflow.log_params(params)

        resultados = metricas.get('results', {})
        for nome_modelo, medidas in resultados.items():
            for nome_medida, valor in medidas.items():
                chave = f'{nome_modelo}_{nome_medida}'
                mlflow.log_metric(chave, float(valor))

        mlflow.log_artifact(str(caminho_metricas), artifact_path='baseline_metrics')
        LOGGER.info('Run MLflow registrada: %s', execucao.info.run_id)


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Registrar métricas de baseline no MLflow')
    parser.add_argument('--experiment-name', type=str, default='telco_churn_baseline', help='Nome do experimento MLflow')
    parser.add_argument('--run-name', type=str, default='baseline_run', help='Nome da execução MLflow')
    parser.add_argument('--metrics-path', type=Path, default=Path('results/baseline_metrics.json'), help='Caminho para o arquivo JSON com métricas')
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parsear_argumentos()
    metricas = carregar_metricas(args.metrics_path)
    registrar_metricas(args.experiment_name, args.run_name, metricas, args.metrics_path)


if __name__ == '__main__':
    main()
