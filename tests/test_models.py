import pytest
from sklearn.model_selection import ParameterGrid

from benchmark.data import load_data
from benchmark.evaluation import compute_metrics
from benchmark.models import BASELINE, MODELS


@pytest.fixture(scope="module")
def sample():
    X, y = load_data()
    return X.iloc[:200], y.iloc[:200]


def test_target_is_recoded_with_malignant_positive():
    _, y = load_data()
    assert int(y.sum()) == 212  # 212 malins, 357 bénins
    assert len(y) == 569


def test_baseline_is_registered():
    assert BASELINE in MODELS


@pytest.mark.parametrize("key", list(MODELS))
def test_every_grid_point_is_valid_and_fits(sample, key):
    X, y = sample
    spec = MODELS[key]
    for params in ParameterGrid(spec.param_grid):
        spec.build().set_params(**params)  # lève une erreur si une clé est invalide
    model = spec.build().fit(X, y)
    scores = compute_metrics(model, X, y, ("accuracy", "recall", "roc_auc"))
    assert all(0.0 <= v <= 1.0 for v in scores.values())
