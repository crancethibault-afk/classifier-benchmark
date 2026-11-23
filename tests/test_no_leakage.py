"""Le pli de test externe ne doit jamais servir au réglage ni au prétraitement."""
import numpy as np
import pytest

from benchmark.data import load_data
from benchmark.evaluation import tune_and_score
from benchmark.models import MODELS
from benchmark.splits import make_outer_splits

KWARGS = dict(
    metrics=("accuracy", "roc_auc"),
    selection_metric="roc_auc",
    n_inner_splits=3,
    random_state=42,
    n_jobs=1,
)


@pytest.fixture(scope="module")
def data():
    X, y = load_data()
    split = make_outer_splits(y.to_numpy(), n_splits=5, n_repeats=1, random_state=42)[0]
    return X, y, split


@pytest.mark.parametrize("key", list(MODELS))
def test_tuning_ignores_test_rows(data, key):
    """Corrompre les lignes de test ne doit rien changer au réglage."""
    X, y, split = data
    X_poisoned = X.copy()
    X_poisoned.iloc[split["test"]] = 1e6
    y_poisoned = y.copy()
    y_poisoned.iloc[split["test"]] = 1 - y_poisoned.iloc[split["test"]]

    clean, _ = tune_and_score(MODELS[key], X, y, split, **KWARGS)
    poisoned, _ = tune_and_score(MODELS[key], X_poisoned, y_poisoned, split, **KWARGS)
    assert clean.best_params_ == poisoned.best_params_
    assert clean.best_score_ == poisoned.best_score_


@pytest.mark.parametrize("key", list(MODELS))
def test_scaler_fitted_on_training_rows_only(data, key):
    X, y, split = data
    search, _ = tune_and_score(MODELS[key], X, y, split, **KWARGS)
    scaler = search.best_estimator_.named_steps.get("scaler")
    if scaler is None:
        pytest.skip("pas de normalisation dans ce pipeline")
    np.testing.assert_allclose(scaler.mean_, X.iloc[split["train"]].mean().to_numpy())
    assert not np.allclose(scaler.mean_, X.mean().to_numpy())
