"""Paramètres du protocole d'évaluation.

C'est le seul endroit à modifier pour changer le protocole (graine, nombre de
plis, métriques, critère de sélection). Les hyperparamètres des modèles sont
définis à part, dans ``models.py``.
"""
from pathlib import Path

# --- Reproductibilité -------------------------------------------------------
RANDOM_STATE = 42

# --- Validation croisée -----------------------------------------------------
# Boucle externe : sert uniquement à l'évaluation.
N_OUTER_SPLITS = 5
N_OUTER_REPEATS = 1  # passer à 5 une fois le temps d'exécution mesuré
# Boucle interne : sert uniquement au réglage des hyperparamètres,
# exécutée sur la partie entraînement de chaque pli externe.
N_INNER_SPLITS = 5

# --- Métriques --------------------------------------------------------------
# Classe positive = tumeur maligne (voir data.py).
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
)
# Métrique optimisée par la recherche d'hyperparamètres (boucle interne).
SELECTION_METRIC = "roc_auc"

# --- Exécution --------------------------------------------------------------
N_JOBS = -1  # parallélisme de GridSearchCV

# --- Chemins ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"   # sorties brutes, ignorées par git
REPORTS_DIR = PROJECT_ROOT / "reports"   # synthèse et figures, versionnées
FIGURES_DIR = REPORTS_DIR / "figures"

SPLITS_PATH = RESULTS_DIR / "splits.json"
FOLD_SCORES_PATH = RESULTS_DIR / "fold_scores.csv"
BEST_PARAMS_PATH = RESULTS_DIR / "best_params.csv"
