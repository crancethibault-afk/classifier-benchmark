"""Synthèse des résultats : tableau et figures, écrits dans ``reports/``."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # rendu sans affichage
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark.models import MODELS  # noqa: E402

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Accuracy équilibrée",
    "precision": "Précision (malin)",
    "recall": "Rappel (malin)",
    "f1": "F1 (malin)",
    "roc_auc": "ROC AUC",
}

# Couleurs : une seule teinte pour les points, l'identité des modèles est
# portée par l'axe des abscisses.
_POINT = "#2a78d6"
_MEAN = "#0d366b"
_PAIR_LINE = "#c3c2b7"
_GRID = "#e6e5e0"
_TEXT_MUTED = "#6b6a63"


def _ordered_models(scores: pd.DataFrame) -> list[str]:
    """Ordre de MODELS (référence en premier), restreint aux modèles présents."""
    present = set(scores["model"])
    return [key for key in MODELS if key in present]


def summarize(scores: pd.DataFrame, metrics) -> pd.DataFrame:
    """Moyenne et écart-type de chaque métrique sur les plis externes."""
    models = _ordered_models(scores)
    agg = scores.groupby("model")[list(metrics) + ["tuning_time_s"]].agg(["mean", "std"])
    return agg.loc[models]


def write_summary(summary: pd.DataFrame, metrics, n_folds: int, reports_dir: Path) -> Path:
    """Écrit ``summary.csv`` (valeurs brutes) et ``summary.md`` (lisible)."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    flat = summary.copy()
    flat.columns = [f"{metric}_{stat}" for metric, stat in flat.columns]
    flat.to_csv(reports_dir / "summary.csv", float_format="%.4f")

    header = ["Modèle"] + [METRIC_LABELS.get(m, m) for m in metrics] + ["Réglage (s/pli)"]
    lines = [
        f"Moyenne ± écart-type sur {n_folds} plis externes.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
    ]
    for key, row in summary.iterrows():
        cells = [MODELS[key].label]
        cells += [f"{row[(m, 'mean')]:.3f} ± {row[(m, 'std')]:.3f}" for m in metrics]
        cells.append(f"{row[('tuning_time_s', 'mean')]:.2f}")
        lines.append("| " + " | ".join(cells) + " |")

    path = reports_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def plot_fold_scores(scores: pd.DataFrame, metrics, path: Path) -> Path:
    """Un panneau par métrique : score de chaque pli, reliés entre modèles
    (les plis sont identiques), et moyenne en trait horizontal."""
    models = _ordered_models(scores)
    labels = [MODELS[key].label for key in models]
    positions = range(len(models))

    n_cols = 3
    n_rows = -(-len(metrics) // n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 3.4 * n_rows), squeeze=False
    )

    wide = scores.pivot_table(index=["repeat", "fold"], columns="model", values=list(metrics))

    for ax, metric in zip(axes.flat, metrics):
        table = wide[metric][models]
        for _, fold_scores in table.iterrows():
            ax.plot(positions, fold_scores.values, color=_PAIR_LINE, linewidth=1, zorder=1)
        for x, key in zip(positions, models):
            ax.scatter(
                [x] * len(table), table[key], s=36, color=_POINT,
                edgecolor="white", linewidth=1.5, zorder=2,
            )
            mean = table[key].mean()
            ax.hlines(mean, x - 0.22, x + 0.22, color=_MEAN, linewidth=2.5, zorder=3)
            ax.annotate(
                f"{mean:.3f}", (x + 0.26, mean), va="center", fontsize=8, color=_MEAN
            )

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11, loc="left")
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlim(-0.5, len(models) - 0.5)
        ax.tick_params(axis="y", labelsize=8, colors=_TEXT_MUTED)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(_GRID)

    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(
        "Scores par pli externe (points) et moyenne (trait) — plis reliés entre modèles",
        fontsize=10, color=_TEXT_MUTED, x=0.01, ha="left",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path
