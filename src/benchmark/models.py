"""Modèles comparés et leurs grilles d'hyperparamètres.

C'est le seul endroit où les hyperparamètres sont définis. Chaque entrée de
``MODELS`` associe un pipeline (prétraitement + classifieur) à la grille
explorée par la boucle interne. Les clés de grille sont préfixées par le nom
de l'étape du pipeline (``clf__C``...).

La normalisation fait partie du pipeline : elle est réapprise sur chaque
partie entraînement et ne voit jamais les données de test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from benchmark.config import RANDOM_STATE


@dataclass(frozen=True)
class ModelSpec:
    label: str
    build: Callable[[], Pipeline]
    param_grid: dict


def _logreg() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])


def _knn() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier()),
    ])


MODELS: dict[str, ModelSpec] = {
    # Référence : régression logistique régularisée L2.
    "logreg": ModelSpec(
        label="Régression logistique",
        build=_logreg,
        param_grid={
            "clf__C": np.logspace(-3, 3, 7).tolist(),
        },
    ),
    "knn": ModelSpec(
        label="kNN",
        build=_knn,
        param_grid={
            "clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31, 41, 61, 81],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],  # distance de Manhattan / euclidienne
        },
    ),
}

BASELINE = "logreg"
