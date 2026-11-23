"""Validation croisée imbriquée.

- Boucle externe : plis lus dans ``splits.json``, utilisés uniquement pour
  mesurer les performances.
- Boucle interne : ``GridSearchCV`` exécuté sur la seule partie entraînement
  du pli externe, pour choisir les hyperparamètres. Le pli de test externe
  n'intervient jamais dans ce choix.
"""
from __future__ import annotations

import json
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from benchmark.models import ModelSpec

# Métriques fondées sur les prédictions (seuil par défaut du modèle).
_LABEL_METRICS = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "precision": lambda y, p: precision_score(y, p, zero_division=0),
    "recall": recall_score,
    "f1": f1_score,
}
# Métriques fondées sur un score continu (indépendantes du seuil).
_SCORE_METRICS = {
    "roc_auc": roc_auc_score,
}


def _positive_scores(estimator, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def compute_metrics(estimator, X: pd.DataFrame, y: pd.Series, metrics) -> dict:
    unknown = set(metrics) - set(_LABEL_METRICS) - set(_SCORE_METRICS)
    if unknown:
        raise ValueError(f"Métriques non implémentées : {sorted(unknown)}")

    y_pred = estimator.predict(X)
    y_score = _positive_scores(estimator, X)
    results = {}
    for name in metrics:
        if name in _LABEL_METRICS:
            results[name] = _LABEL_METRICS[name](y, y_pred)
        else:
            results[name] = _SCORE_METRICS[name](y, y_score)
    return results


def tune_and_score(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    split: dict,
    *,
    metrics,
    selection_metric: str,
    n_inner_splits: int,
    random_state: int,
    n_jobs: int,
) -> tuple[GridSearchCV, dict]:
    """Règle le modèle sur la partie entraînement d'un pli, l'évalue sur le test."""
    X_train, y_train = X.iloc[split["train"]], y.iloc[split["train"]]
    X_test, y_test = X.iloc[split["test"]], y.iloc[split["test"]]

    # Même graine pour la boucle interne : à pli externe égal, tous les modèles
    # sont réglés sur les mêmes sous-découpages.
    inner_cv = StratifiedKFold(
        n_splits=n_inner_splits, shuffle=True, random_state=random_state
    )
    search = GridSearchCV(
        spec.build(),
        spec.param_grid,
        scoring=selection_metric,
        cv=inner_cv,
        n_jobs=n_jobs,
        refit=True,  # réentraîne le meilleur modèle sur tout X_train
    )
    start = perf_counter()
    search.fit(X_train, y_train)
    tuning_time = perf_counter() - start

    scores = compute_metrics(search.best_estimator_, X_test, y_test, metrics)
    scores["tuning_time_s"] = tuning_time
    return search, scores


def evaluate_model(
    key: str,
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[dict],
    **kwargs,
) -> tuple[list[dict], list[dict]]:
    """Évalue un modèle sur tous les plis externes.

    Renvoie les scores par pli et les hyperparamètres retenus par pli.
    """
    score_rows, param_rows = [], []
    for split in splits:
        search, scores = tune_and_score(spec, X, y, split, **kwargs)
        ids = {"model": key, "repeat": split["repeat"], "fold": split["fold"]}
        score_rows.append({**ids, **scores})
        param_rows.append({
            **ids,
            "inner_best_score": search.best_score_,
            "best_params": json.dumps(search.best_params_, default=str),
        })
    return score_rows, param_rows
