"""Chargement du jeu de données Breast Cancer Wisconsin."""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Renvoie ``(X, y)`` avec ``y = 1`` pour une tumeur maligne.

    Dans scikit-learn, la cible d'origine code 0 = malin et 1 = bénin. On la
    recode pour que la classe positive soit la classe maligne : le rappel
    mesure alors la proportion de cancers détectés.
    """
    dataset = load_breast_cancer(as_frame=True)
    if list(dataset.target_names) != ["malignant", "benign"]:
        raise RuntimeError(f"Codage de la cible inattendu : {dataset.target_names}")

    X = dataset.data
    y = (dataset.target == 0).astype(int).rename("malignant")
    return X, y
