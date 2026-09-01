"""Ensemble que combina o MLP binário e o MLP de score.

A ideia é carregar os artefatos gerados por `mlp_training` (modelo, scaler,
feature_columns) e oferecer um ponto de entrada simples para o notebook:

- carregar ambos os modelos
- gerar predições de probabilidade (binário) e predições de score (regressão)
- combinar em uma decisão final baseada em regra/ponderação

Este arquivo fornece a classe `EnsembleMLP` com métodos:
- load_artifacts(path_to_task_dir)
- predict(df)
- predict_proba_and_score(df)
- decision_logic(prob, score)  -> decision (0/1) e metadata

Observação: imports de torch e joblib são usados para usar os modelos salvo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from mlp_training import MLP


class EnsembleMLP:
    def __init__(self, binary_dir: Path | str = 'results/mlp/binary', score_dir: Path | str = 'results/mlp/score', device: str | None = None) -> None:
        self.binary_dir = Path(binary_dir)
        self.score_dir = Path(score_dir)
        self.device = torch.device(device) if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # placeholders
        self.binary_model: nn.Module | None = None
        self.score_model: nn.Module | None = None
        self.binary_scaler = None
        self.score_scaler = None
        self.binary_features: list[str] | None = None
        self.score_features: list[str] | None = None

    def _load_task_artifacts(self, task_dir: Path) -> tuple[nn.Module, Any, list[str]]:
        """Carrega modelo.pt, scaler.joblib, feature_columns.json do diretório da tarefa."""
        model_path = task_dir / 'mlp_model.pt'
        scaler_path = task_dir / 'scaler.joblib'
        features_path = task_dir / 'feature_columns.json'

        if not model_path.exists():
            raise FileNotFoundError(f'Modelo não encontrado em {model_path}')
        if not scaler_path.exists():
            raise FileNotFoundError(f'Scaler não encontrado em {scaler_path}')
        if not features_path.exists():
            raise FileNotFoundError(f'Feature columns não encontrado em {features_path}')

        with features_path.open('r', encoding='utf-8') as f:
            feature_columns = json.load(f)

        scaler = joblib.load(scaler_path)

        # model architecture precisa ser reconstruída com input_dim = len(feature_columns)
        input_dim = len(feature_columns)
        # usar arquitetura padrão (o config.json pode conter detalhes; tentamos lê-lo)
        config_path = task_dir / 'config.json'
        hidden_sizes = [128, 64]
        dropout = 0.2
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text(encoding='utf-8'))
                hidden_sizes = cfg.get('hidden_sizes', hidden_sizes)
                dropout = cfg.get('dropout', dropout)
            except Exception:
                pass

        model = MLP(input_dim=input_dim, hidden_dims=hidden_sizes, output_dim=1, dropout=dropout)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model, scaler, feature_columns

    def load(self) -> None:
        """Carrega ambos os modelos e artefatos."""
        self.binary_model, self.binary_scaler, self.binary_features = self._load_task_artifacts(self.binary_dir)
        self.score_model, self.score_scaler, self.score_features = self._load_task_artifacts(self.score_dir)

    def _prepare_input(self, df: pd.DataFrame, feature_columns: list[str], scaler) -> np.ndarray:
        X = df.reindex(columns=feature_columns).copy()
        X_num = X.select_dtypes(include=[np.number])
        # preencher NA com 0 para evitar erro; recomenda-se imputação anterior
        X_num = X_num.fillna(0.0)
        X_scaled = scaler.transform(X_num)
        return X_scaled

    def predict_proba_and_score(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Retorna lista de resultados com keys: prob, score, decision, metadata"""
        if self.binary_model is None or self.score_model is None:
            raise RuntimeError('Modelos não carregados. Chame load() antes.')

        X_bin = self._prepare_input(df, self.binary_features, self.binary_scaler)
        X_score = self._prepare_input(df, self.score_features, self.score_scaler)

        with torch.no_grad():
            xb = torch.from_numpy(X_bin.astype('float32')).to(self.device)
            logits = self.binary_model(xb)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

            xs = torch.from_numpy(X_score.astype('float32')).to(self.device)
            preds_score = self.score_model(xs).cpu().numpy().reshape(-1)

        results = []
        for p, s in zip(probs.tolist(), preds_score.tolist()):
            decision, meta = self.decision_logic(p, s)
            results.append({'prob': float(p), 'score': float(s), 'decision': int(decision), 'meta': meta})
        return results

    def decision_logic(self, prob: float, score: float, prob_threshold: float = 0.5, score_threshold: float | None = None) -> tuple[int, dict[str, Any]]:
        """Combina probabilidade e score em uma decisão final.

        Estratégia padrão (ajustável):
        - Se prob >= prob_threshold -> churn=1
        - Senão, se score_threshold definido e score >= score_threshold -> churn=1
        - Senão churn=0

        Também devolve metadados com justificativa e valores.
        """
        if prob >= prob_threshold:
            return 1, {'reason': 'prob_high', 'prob': prob, 'score': score}
        if score_threshold is not None and score >= score_threshold:
            return 1, {'reason': 'score_high', 'prob': prob, 'score': score}
        # regra simples: combinar weighted score could be added
        return 0, {'reason': 'below_thresholds', 'prob': prob, 'score': score}

    def predict(self, df: pd.DataFrame, prob_threshold: float = 0.5, score_threshold: float | None = None) -> pd.DataFrame:
        rows = self.predict_proba_and_score(df)
        out = pd.DataFrame(rows)
        # se quiser retornar também features originais
        return out
