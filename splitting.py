"""Particionamento reproduzível e compartilhado entre experimentos."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitIndices:
    """Índices posicionais usados nas partições de treino, validação e teste."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def get_or_create_split_indices(
    y: np.ndarray,
    artifact_path: str | Path = "data/processed/split_indices.npz",
    *,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_seed: int = 42,
) -> SplitIndices:
    """Carrega ou cria as mesmas partições estratificadas para todos os modelos.

    ``validation_size`` é a fração do conjunto restante após separar o teste.
    O artefato é invalidado de forma explícita se a quantidade de registros ou
    os parâmetros do particionamento forem alterados.
    """
    target = np.asarray(y)
    if target.ndim != 1:
        raise ValueError("O alvo para particionamento deve ter uma dimensão.")
    if len(target) == 0:
        raise ValueError("Não é possível particionar um dataset vazio.")

    artifact = Path(artifact_path)
    metadata_path = artifact.with_suffix(".json")
    expected_metadata = {
        "n_samples": len(target),
        "target_sha256": hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest(),
        "test_size": test_size,
        "validation_size": validation_size,
        "random_seed": random_seed,
    }

    if artifact.exists() or metadata_path.exists():
        if not artifact.exists() or not metadata_path.exists():
            raise RuntimeError(
                "Artefatos de split incompletos. Remova ambos os arquivos para recriá-los."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata != expected_metadata:
            raise ValueError(
                "O split persistido não corresponde ao dataset ou à configuração atual. "
                "Remova os artefatos de split para criar uma nova partição."
            )
        with np.load(artifact) as saved:
            split = SplitIndices(
                train=saved["train"],
                validation=saved["validation"],
                test=saved["test"],
            )
        _validate_split(split, len(target))
        return split

    indices = np.arange(len(target))
    train_validation, test = train_test_split(
        indices,
        test_size=test_size,
        stratify=target,
        random_state=random_seed,
    )
    train, validation = train_test_split(
        train_validation,
        test_size=validation_size,
        stratify=target[train_validation],
        random_state=random_seed,
    )
    split = SplitIndices(train=train, validation=validation, test=test)
    _validate_split(split, len(target))

    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact, train=train, validation=validation, test=test)
    metadata_path.write_text(
        json.dumps(expected_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return split


def _validate_split(split: SplitIndices, n_samples: int) -> None:
    combined = np.concatenate((split.train, split.validation, split.test))
    if len(combined) != n_samples or len(np.unique(combined)) != n_samples:
        raise ValueError("Os índices do split precisam ser disjuntos e cobrir todo o dataset.")
    if combined.min() < 0 or combined.max() >= n_samples:
        raise ValueError("Os índices do split estão fora dos limites do dataset.")
