"""Point d'entrée : données -> plis -> validation croisée imbriquée -> rapport.

Exemples :
    python scripts/run_benchmark.py                 # tous les modèles
    python scripts/run_benchmark.py --models knn    # un seul modèle

Les résultats d'un modèle relancé remplacent ses anciens résultats ; ceux des
autres modèles sont conservés tant que les plis n'ont pas changé.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# numpy 2.0 + Accelerate (macOS arm64) émet des RuntimeWarning parasites dans
# matmul, même sur des données finies ; les résultats sont corrects. On ne
# masque que ces messages-là. La variable d'environnement couvre aussi les
# processus parallèles lancés par GridSearchCV.
_MATMUL_WARNINGS = ("divide by zero", "overflow", "invalid value")
for _msg in _MATMUL_WARNINGS:
    warnings.filterwarnings("ignore", f"{_msg} encountered in matmul", RuntimeWarning)
os.environ["PYTHONWARNINGS"] = ",".join(
    filter(None, [os.environ.get("PYTHONWARNINGS")]
           + [f"ignore:{m} encountered in matmul:RuntimeWarning" for m in _MATMUL_WARNINGS])
)

import pandas as pd  # noqa: E402

from benchmark import config  # noqa: E402
from benchmark.data import load_data  # noqa: E402
from benchmark.evaluation import evaluate_model  # noqa: E402
from benchmark.models import MODELS  # noqa: E402
from benchmark.reporting import plot_fold_scores, summarize, write_summary  # noqa: E402
from benchmark.splits import get_or_create_splits, splits_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models", nargs="+", choices=list(MODELS), default=list(MODELS),
        help="modèles à évaluer (par défaut : tous)",
    )
    parser.add_argument("--n-jobs", type=int, default=config.N_JOBS)
    return parser.parse_args()


def merge_with_previous(new: pd.DataFrame, path: Path, current_splits: str) -> pd.DataFrame:
    """Remplace les lignes des modèles relancés, garde les autres si mêmes plis."""
    if not path.exists():
        return new
    previous = pd.read_csv(path)
    keep = (previous["splits_id"] == current_splits) & ~previous["model"].isin(new["model"])
    return pd.concat([previous[keep], new], ignore_index=True)


def main() -> None:
    args = parse_args()
    X, y = load_data()
    splits = get_or_create_splits(
        y.to_numpy(), config.SPLITS_PATH,
        n_splits=config.N_OUTER_SPLITS, n_repeats=config.N_OUTER_REPEATS,
        random_state=config.RANDOM_STATE,
    )
    sid = splits_id(splits)
    print(f"{len(X)} échantillons, {int(y.sum())} malins | "
          f"{len(splits)} plis externes (id {sid}) | sélection : {config.SELECTION_METRIC}")

    all_scores, all_params = [], []
    total_start = perf_counter()
    for key in args.models:
        start = perf_counter()
        scores, params = evaluate_model(
            key, MODELS[key], X, y, splits,
            metrics=config.METRICS,
            selection_metric=config.SELECTION_METRIC,
            n_inner_splits=config.N_INNER_SPLITS,
            random_state=config.RANDOM_STATE,
            n_jobs=args.n_jobs,
        )
        all_scores += scores
        all_params += params
        print(f"  {MODELS[key].label:<25} {perf_counter() - start:6.1f} s")
    print(f"Durée totale : {perf_counter() - total_start:.1f} s")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores_df = pd.DataFrame(all_scores).assign(splits_id=sid)
    params_df = pd.DataFrame(all_params).assign(splits_id=sid)
    scores_df = merge_with_previous(scores_df, config.FOLD_SCORES_PATH, sid)
    params_df = merge_with_previous(params_df, config.BEST_PARAMS_PATH, sid)
    scores_df.to_csv(config.FOLD_SCORES_PATH, index=False)
    params_df.to_csv(config.BEST_PARAMS_PATH, index=False)

    summary = summarize(scores_df, config.METRICS)
    summary_path = write_summary(summary, config.METRICS, len(splits), config.REPORTS_DIR)
    figure_path = plot_fold_scores(
        scores_df, config.METRICS, config.FIGURES_DIR / "fold_scores.png"
    )

    print()
    print(summary_path.read_text())
    print(f"Figure : {figure_path.relative_to(config.PROJECT_ROOT)}")
    print(f"Détails par pli : {config.FOLD_SCORES_PATH.relative_to(config.PROJECT_ROOT)}, "
          f"{config.BEST_PARAMS_PATH.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
