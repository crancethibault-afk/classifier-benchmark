"""Découpages externes de la validation croisée.

Les plis sont générés une seule fois puis enregistrés dans ``splits.json`` :
tous les modèles relisent exactement les mêmes indices.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold


def make_outer_splits(
    y: np.ndarray, n_splits: int, n_repeats: int, random_state: int
) -> list[dict]:
    """Génère les plis externes stratifiés (indices positionnels)."""
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    placeholder_X = np.zeros((len(y), 1))
    return [
        {
            "repeat": i // n_splits,
            "fold": i % n_splits,
            "train": train.tolist(),
            "test": test.tolist(),
        }
        for i, (train, test) in enumerate(cv.split(placeholder_X, y))
    ]


def save_splits(splits: list[dict], metadata: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"metadata": metadata, "splits": splits}, f)


def load_splits(path: Path) -> tuple[list[dict], dict]:
    with open(path) as f:
        content = json.load(f)
    return content["splits"], content["metadata"]


def splits_id(splits: list[dict]) -> str:
    """Empreinte courte des plis, pour relier chaque résultat à ses découpages."""
    payload = json.dumps(splits, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def get_or_create_splits(
    y: np.ndarray, path: Path, n_splits: int, n_repeats: int, random_state: int
) -> list[dict]:
    """Relit les plis enregistrés, ou les (re)génère si le protocole a changé."""
    metadata = {
        "n_samples": int(len(y)),
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "random_state": random_state,
    }
    if path.exists():
        splits, saved_metadata = load_splits(path)
        if saved_metadata == metadata:
            return splits
        print(f"Protocole modifié ({saved_metadata} -> {metadata}) : plis régénérés.")

    splits = make_outer_splits(y, n_splits, n_repeats, random_state)
    save_splits(splits, metadata, path)
    return splits
