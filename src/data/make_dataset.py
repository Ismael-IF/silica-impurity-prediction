"""Étape 1 — séparation train / test.

Lit le fichier brut, isole la cible et produit quatre jeux versionnés par
DVC. Le découpage est fait avant toute transformation : c'est ce qui garantit
qu'aucune information du test ne fuite dans l'apprentissage.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import get_paths, load_params, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    params = load_params()
    paths = get_paths(params)
    paths.mkdirs()

    data_cfg = params["data"]
    target = data_cfg["target"]

    frame = pd.read_csv(paths.raw)
    logger.info("Fichier brut : %d lignes, %d colonnes", *frame.shape)

    # Les colonnes non prédictives (horodatage) sont retirées si elles
    # existent : le pipeline reste utilisable sur un export qui ne les a pas.
    to_drop = [c for c in data_cfg.get("drop_columns", []) if c in frame.columns]
    if to_drop:
        frame = frame.drop(columns=to_drop)
        logger.info("Colonnes retirées : %s", ", ".join(to_drop))

    if target not in frame.columns:
        raise KeyError(f"Colonne cible absente du fichier brut : {target!r}")
    if frame.isna().any().any():
        raise ValueError(
            "Valeurs manquantes détectées ; le pipeline attend un export complet."
        )

    features = frame.drop(columns=[target])
    labels = frame[target]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=data_cfg["test_size"],
        random_state=params["seed"],
    )

    for name, obj in {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }.items():
        obj.to_csv(paths.processed / f"{name}.csv", index=False)

    logger.info(
        "Découpage écrit dans %s — train : %d lignes, test : %d lignes",
        paths.processed,
        len(X_train),
        len(X_test),
    )


if __name__ == "__main__":
    setup_logging()
    main()
