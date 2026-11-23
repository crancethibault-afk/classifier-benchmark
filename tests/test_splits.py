import numpy as np
import pytest

from benchmark.data import load_data
from benchmark.splits import get_or_create_splits, make_outer_splits, splits_id


@pytest.fixture(scope="module")
def y():
    return load_data()[1].to_numpy()


def test_splits_are_deterministic(y):
    a = make_outer_splits(y, n_splits=5, n_repeats=2, random_state=42)
    b = make_outer_splits(y, n_splits=5, n_repeats=2, random_state=42)
    assert splits_id(a) == splits_id(b)


def test_train_and_test_are_disjoint_and_complete(y):
    for split in make_outer_splits(y, n_splits=5, n_repeats=2, random_state=42):
        train, test = set(split["train"]), set(split["test"])
        assert not train & test
        assert train | test == set(range(len(y)))


def test_each_sample_tested_once_per_repeat(y):
    splits = make_outer_splits(y, n_splits=5, n_repeats=2, random_state=42)
    for repeat in (0, 1):
        tested = np.concatenate([s["test"] for s in splits if s["repeat"] == repeat])
        assert sorted(tested) == list(range(len(y)))


def test_folds_are_stratified(y):
    n_splits = 5
    for split in make_outer_splits(y, n_splits=n_splits, n_repeats=1, random_state=42):
        n_pos = y[split["test"]].sum()
        assert abs(n_pos - y.sum() / n_splits) <= 1


def test_saved_splits_are_reused(y, tmp_path):
    path = tmp_path / "splits.json"
    first = get_or_create_splits(y, path, n_splits=5, n_repeats=1, random_state=42)
    # Même protocole : plis relus à l'identique. Protocole modifié : régénérés.
    again = get_or_create_splits(y, path, n_splits=5, n_repeats=1, random_state=42)
    assert splits_id(first) == splits_id(again)
    changed = get_or_create_splits(y, path, n_splits=5, n_repeats=1, random_state=0)
    assert splits_id(changed) != splits_id(first)
