"""Treinamento de MLP PyTorch para churn binário e churn score."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from metrics import calcular_metricas_classificacao, calcular_metricas_regressao
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


LOGGER = logging.getLogger(__name__)


class ChurnDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(dim=-1)


class MLPTrainer:
    def __init__(
        self,
        input_path: str | Path = 'data/processed/telco_feature_engineered.csv',
        task: str = 'binary',
        experiment_name: str = 'telco_churn_mlp',
        run_name: str = 'mlp_run',
        output_dir: str | Path = 'results/mlp',
        test_size: float = 0.2,
        val_size: float = 0.2,
        batch_size: int = 64,
        hidden_sizes: str | list[int] = '128,64',
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 50,
        patience: int = 10,
        random_seed: int = 42,
        no_cuda: bool = False,
    ) -> None:
        self.input_path = Path(input_path)
        self.task = task
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.hidden_sizes = self._parse_hidden_sizes(hidden_sizes)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.random_seed = random_seed
        self.no_cuda = no_cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() and not self.no_cuda else 'cpu')
        self.model: nn.Module | None = None
        self.scaler: StandardScaler | None = None
        self.feature_columns: list[str] = []
        self.dataset: pd.DataFrame | None = None

    @staticmethod
    def _parse_hidden_sizes(valor: str | list[int]) -> list[int]:
        if isinstance(valor, list):
            return [int(item) for item in valor]
        return [int(item.strip()) for item in str(valor).split(',') if item.strip()]

    def carregar_dataset(self, caminho_entrada: Path | str | None = None) -> pd.DataFrame:
        caminho = Path(caminho_entrada) if caminho_entrada is not None else self.input_path
        LOGGER.info('Carregando dataset de %s', caminho)
        df = pd.read_csv(caminho)
        LOGGER.info('Dataset carregado com shape %s', df.shape)
        self.dataset = df
        return df

    def remover_colunas_redundantes(self, df: pd.DataFrame, tarefa: str | None = None) -> pd.DataFrame:
        tarefa = tarefa or self.task
        colunas_a_descartar = [
            'City',
            'Gender',
            'CustomerID',
            'Count',
            'Country',
            'State',
            'Zip Code',
            'Lat Long',
            'Latitude',
            'Longitude',
            'Churn Label',
            'Churn Reason',
            'Churn Value' if tarefa == 'score' else 'Churn Score',
        ]
        return df.drop(columns=[col for col in colunas_a_descartar if col in df.columns], errors='ignore')

    def preparar_base(self, df: pd.DataFrame | None = None, tarefa: str | None = None) -> tuple[np.ndarray, np.ndarray, list[str], StandardScaler]:
        tarefa = tarefa or self.task
        if tarefa == 'binary':
            alvo = 'Churn Value'
        elif tarefa == 'score':
            alvo = 'Churn Score'
        else:
            raise ValueError('Tarefa inválida, use binary ou score')

        dataset = self.carregar_dataset() if df is None else df
        if alvo not in dataset.columns:
            raise KeyError(f'Alvo não encontrado: {alvo}')

        y = dataset[alvo].astype(float).to_numpy(dtype=np.float32)
        df_limpo = self.remover_colunas_redundantes(dataset, tarefa)
        df_limpo = df_limpo.drop(columns=[alvo], errors='ignore')

        if 'tenure_bucket' in df_limpo.columns:
            df_limpo = pd.get_dummies(df_limpo, columns=['tenure_bucket'], drop_first=True)

        X = df_limpo.select_dtypes(include=[np.number]).copy()
        feature_columns = X.columns.tolist()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler
        self.feature_columns = feature_columns

        if np.isnan(X_scaled).any():
            LOGGER.warning('A base contém valores NaN após escala; verifique a preparação dos dados')

        return X_scaled, y, feature_columns, scaler

    def criar_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int | None = None,
    ) -> tuple[DataLoader, DataLoader]:
        batch_size = batch_size or self.batch_size
        train_dataset = ChurnDataset(X_train, y_train)
        val_dataset = ChurnDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        return train_loader, val_loader

    def criar_modelo(self, input_dim: int) -> nn.Module:
        return MLP(input_dim=input_dim, hidden_dims=self.hidden_sizes, output_dim=1, dropout=self.dropout).to(self.device)

    def criar_criterio(self, tarefa: str | None = None, peso_positiva: float | None = None) -> nn.Module:
        tarefa = tarefa or self.task
        if tarefa == 'binary':
            if peso_positiva is not None:
                return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(peso_positiva, dtype=torch.float32, device=self.device))
            return nn.BCEWithLogitsLoss()
        return nn.MSELoss()

    def avaliar_modelo(
        self,
        model: nn.Module,
        loader: DataLoader,
        tarefa: str | None = None,
        device: torch.device | None = None,
    ) -> tuple[dict[str, float], list[float], list[float]]:
        tarefa = tarefa or self.task
        device = device or self.device
        model.eval()
        y_true: list[float] = []
        y_pred: list[float] = []
        y_prob: list[float] = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch)
                y_true.extend(y_batch.cpu().numpy().tolist())

                if tarefa == 'binary':
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs >= 0.5).astype(int)
                    y_pred.extend(preds.tolist())
                    y_prob.extend(probs.tolist())
                else:
                    preds = logits.cpu().numpy()
                    y_pred.extend(preds.tolist())

        y_true_arr = np.asarray(y_true, dtype=np.float32)
        y_pred_arr = np.asarray(y_pred, dtype=np.float32)

        if tarefa == 'binary':
            metricas = calcular_metricas_classificacao(y_true_arr, y_pred_arr, np.asarray(y_prob, dtype=np.float32))
        else:
            metricas = calcular_metricas_regressao(y_true_arr, y_pred_arr)

        return metricas, y_true, y_pred

    def treinar_epoca(
        self,
        model: nn.Module,
        loader: DataLoader,
        otimizador: optim.Optimizer,
        criterio: nn.Module,
        device: torch.device | None = None,
    ) -> float:
        device = device or self.device
        model.train()
        soma_perda = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            otimizador.zero_grad()
            logits = model(X_batch)
            perda = criterio(logits, y_batch)
            perda.backward()
            otimizador.step()
            soma_perda += perda.item() * X_batch.size(0)
        return soma_perda / len(loader.dataset)

    def salvar_artifacts(
        self,
        diretorio_saida: Path,
        model: nn.Module,
        scaler: StandardScaler,
        feature_columns: list[str],
        config: dict[str, Any],
    ) -> None:
        diretorio_saida.mkdir(parents=True, exist_ok=True)
        modelo_path = diretorio_saida / 'mlp_model.pt'
        scaler_path = diretorio_saida / 'scaler.joblib'
        features_path = diretorio_saida / 'feature_columns.json'
        config_path = diretorio_saida / 'config.json'

        torch.save(model.state_dict(), modelo_path)
        joblib.dump(scaler, scaler_path)

        with features_path.open('w', encoding='utf-8') as arquivo:
            json.dump(feature_columns, arquivo, indent=2, ensure_ascii=False)

        with config_path.open('w', encoding='utf-8') as arquivo:
            json.dump(config, arquivo, indent=2, ensure_ascii=False)

        LOGGER.info('Artefatos salvos em %s', diretorio_saida)

    def registrar_mlflow(
        self,
        experiment_name: str,
        run_name: str,
        params: dict[str, Any],
        metricas: dict[str, float],
        artifacts_dir: Path,
    ) -> None:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as execucao:
            mlflow.log_params(params)
            for nome, valor in metricas.items():
                mlflow.log_metric(nome, float(valor))

            for arquivo in artifacts_dir.iterdir():
                if arquivo.is_file():
                    mlflow.log_artifact(str(arquivo), artifact_path='artifacts')

            LOGGER.info('Registro MLflow criado: %s', execucao.info.run_id)

    def treinar_tarefa(self, tarefa: str | None = None, df: pd.DataFrame | None = None) -> dict[str, Any]:
        tarefa = tarefa or self.task
        self.task = tarefa
        dataset = self.carregar_dataset() if df is None else df
        X, y, feature_columns, scaler = self.preparar_base(dataset, tarefa)

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=y if tarefa == 'binary' else None,
            random_state=self.random_seed,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=self.val_size,
            stratify=y_train_val if tarefa == 'binary' else None,
            random_state=self.random_seed,
        )

        train_loader, val_loader = self.criar_dataloaders(X_train, y_train, X_val, y_val, self.batch_size)
        model = self.criar_modelo(X.shape[1])
        self.model = model

        if tarefa == 'binary':
            contagem = np.bincount(y_train.astype(int))
            pos_weight = float(contagem[0] / contagem[1]) if len(contagem) > 1 and contagem[1] > 0 else 1.0
            criterio = self.criar_criterio(tarefa, peso_positiva=pos_weight)
        else:
            criterio = self.criar_criterio(tarefa)

        otimizador = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        melhor_valor = float('inf') if tarefa == 'score' else 0.0
        contador_sem_melhora = 0
        melhor_estado: dict[str, torch.Tensor] | None = None

        for epoca in range(1, self.epochs + 1):
            perda_treino = self.treinar_epoca(model, train_loader, otimizador, criterio, self.device)
            metricas_val, _, _ = self.avaliar_modelo(model, val_loader, tarefa, self.device)

            if tarefa == 'binary':
                valor_monitorado = metricas_val['recall']
                melhor_melhora = valor_monitorado > melhor_valor
            else:
                valor_monitorado = metricas_val['rmse']
                melhor_melhora = valor_monitorado < melhor_valor

            if melhor_melhora:
                melhor_valor = valor_monitorado
                contador_sem_melhora = 0
                melhor_estado = model.state_dict()
            else:
                contador_sem_melhora += 1

            LOGGER.info('Época %d/%d - perda treino %.4f - val %s', epoca, self.epochs, perda_treino)
            LOGGER.info('Métricas validação: %s', metricas_val)

            if contador_sem_melhora >= self.patience:
                LOGGER.info('Early stopping após %d épocas sem melhora', self.patience)
                break

        if melhor_estado is not None:
            model.load_state_dict(melhor_estado)

        test_loader = DataLoader(ChurnDataset(X_test, y_test), batch_size=self.batch_size, shuffle=False)
        metricas_teste, _, _ = self.avaliar_modelo(model, test_loader, tarefa, self.device)

        diretorio_saida = self.output_dir / tarefa
        self.salvar_artifacts(
            diretorio_saida,
            model,
            scaler,
            feature_columns,
            {
                'task': tarefa,
                'hidden_sizes': self.hidden_sizes,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'batch_size': self.batch_size,
                'epochs': epoca,
                'seed': self.random_seed,
                'feature_columns': feature_columns,
            },
        )

        run_name = self.run_name if self.run_name.endswith(tarefa) else f'{self.run_name}_{tarefa}'
        self.registrar_mlflow(
            experiment_name=self.experiment_name,
            run_name=run_name,
            params={
                'task': tarefa,
                'hidden_sizes': self.hidden_sizes,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'batch_size': self.batch_size,
                'epochs': epoca,
                'test_size': self.test_size,
                'val_size': self.val_size,
                'random_seed': self.random_seed,
                'output_dir': str(diretorio_saida),
            },
            metricas={f'test_{k}': v for k, v in metricas_teste.items()},
            artifacts_dir=diretorio_saida,
        )

        LOGGER.info('Treinamento finalizado para tarefa %s. Métricas de teste: %s', tarefa, metricas_teste)
        return {
            'task': tarefa,
            'model': model,
            'metrics': metricas_teste,
            'feature_columns': feature_columns,
            'scaler': scaler,
            'artifacts_dir': diretorio_saida,
            'dataset': dataset,
        }

    def treinar_multitarefa(self, tarefas: list[str] | tuple[str, ...] | None = None) -> dict[str, dict[str, Any]]:
        tarefas = list(tarefas or ['binary', 'score'])
        resultados: dict[str, dict[str, Any]] = {}
        for tarefa in tarefas:
            resultados[tarefa] = self.treinar_tarefa(tarefa=tarefa)
        return resultados


def carregar_dataset(caminho_entrada: Path) -> pd.DataFrame:
    return MLPTrainer(input_path=caminho_entrada).carregar_dataset(caminho_entrada)


def parsear_hidden_sizes(valor: str) -> list[int]:
    return MLPTrainer._parse_hidden_sizes(valor)


def remover_colunas_redundantes(df: pd.DataFrame, tarefa: str) -> pd.DataFrame:
    return MLPTrainer(task=tarefa).remover_colunas_redundantes(df, tarefa)


def preparar_base(df: pd.DataFrame, tarefa: str) -> tuple[np.ndarray, np.ndarray, list[str], StandardScaler]:
    return MLPTrainer(task=tarefa).preparar_base(df, tarefa)


def criar_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    trainer = MLPTrainer(batch_size=batch_size)
    return trainer.criar_dataloaders(X_train, y_train, X_val, y_val, batch_size)


def calcular_metricas_classificacao(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    resultados = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        resultados['roc_auc'] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        resultados['roc_auc'] = float('nan')
    return resultados


def calcular_metricas_regressao(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metricas: dict[str, float] = {
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float('nan'),
    }
    return metricas


def avaliar_modelo(
    model: nn.Module,
    loader: DataLoader,
    tarefa: str,
    device: torch.device,
) -> tuple[dict[str, float], list[float], list[float]]:
    trainer = MLPTrainer(task=tarefa)
    return trainer.avaliar_modelo(model, loader, tarefa, device)


def treinar_epoca(
    model: nn.Module,
    loader: DataLoader,
    otimizador: optim.Optimizer,
    criterio: nn.Module,
    device: torch.device,
) -> float:
    trainer = MLPTrainer()
    return trainer.treinar_epoca(model, loader, otimizador, criterio, device)


def criar_criterio(tarefa: str, peso_positiva: float | None = None) -> nn.Module:
    trainer = MLPTrainer(task=tarefa)
    return trainer.criar_criterio(tarefa, peso_positiva)


def salvar_artifacts(
    diretorio_saida: Path,
    model: nn.Module,
    scaler: StandardScaler,
    feature_columns: list[str],
    config: dict[str, Any],
) -> None:
    MLPTrainer().salvar_artifacts(diretorio_saida, model, scaler, feature_columns, config)


def registrar_mlflow(
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metricas: dict[str, float],
    artifacts_dir: Path,
) -> None:
    MLPTrainer().registrar_mlflow(experiment_name, run_name, params, metricas, artifacts_dir)


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Treinar MLP PyTorch para churn')
    parser.add_argument('--input', type=Path, default=Path('data/processed/telco_feature_engineered.csv'), help='Caminho para o dataset processado')
    parser.add_argument('--task', type=str, default='binary', choices=['binary', 'score'], help='Tarefa de modelagem')
    parser.add_argument('--experiment-name', type=str, default='telco_churn_mlp', help='Nome do experimento MLflow')
    parser.add_argument('--run-name', type=str, default='mlp_run', help='Nome da execução MLflow')
    parser.add_argument('--output-dir', type=Path, default=Path('results/mlp'), help='Diretório para salvar artefatos do modelo')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proporção de teste')
    parser.add_argument('--val-size', type=float, default=0.2, help='Proporção de validação em relação ao treino')
    parser.add_argument('--batch-size', type=int, default=64, help='Tamanho do batch')
    parser.add_argument('--hidden-sizes', type=str, default='128,64', help='Tamanhos das camadas escondidas separados por vírgula')
    parser.add_argument('--dropout', type=float, default=0.2, help='Taxa de dropout')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Taxa de aprendizado')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Decaimento de peso')
    parser.add_argument('--epochs', type=int, default=50, help='Número máximo de épocas')
    parser.add_argument('--patience', type=int, default=10, help='Pacência para early stopping')
    parser.add_argument('--random-seed', type=int, default=42, help='Semente aleatória')
    parser.add_argument('--no-cuda', action='store_true', help='Desabilitar GPU mesmo que esteja disponível')
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parsear_argumentos()
    trainer = MLPTrainer(
        input_path=args.input,
        task=args.task,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        random_seed=args.random_seed,
        no_cuda=args.no_cuda,
    )
    trainer.treinar_tarefa(args.task)


if __name__ == '__main__':
    main()
