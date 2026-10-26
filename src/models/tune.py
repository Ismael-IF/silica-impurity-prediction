"""Étape 3 — recherche des hyperparamètres par validation croisée.

La grille est décrite dans `params.yaml`, jamais dans le code. Le résultat
est écrit sur disque en JSON (lisible, comparable d'un commit à l'autre) et
en pickle (format demandé par l'énoncé d'origine), puis consommé par
l'étape d'entraînement : plus de valeurs recopiées à la main entre les deux.
"""

from __future__ import annotations

import json
import logging
import pickle

import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold

from src.config import get_paths, load_params, setup_logging
from src.registry import build_estimator

logger = logging.getLogger(__name__)


def main() -> None:
    params = load_params()
    paths = get_paths(params)
    paths.mkdirs()

    cfg = params["tuning"]
    seed = params["seed"]

    X_train = pd.read_csv(paths.processed / "X_train_scaled.csv")
    # `squeeze` ramène la cible à une série 1-D : scikit-learn attend un
    # vecteur, pas une matrice à une colonne (évite un avertissement et un
    # comportement différent selon les versions).
    y_train = pd.read_csv(paths.processed / "y_train.csv").squeeze("columns")

    estimator = build_estimator(cfg["estimator"], seed=seed)
    grid = {k: v for k, v in cfg["param_grid"].items()}

    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        cv=KFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed),
        scoring=cfg["scoring"],
        n_jobs=params["n_jobs"],
        verbose=cfg.get("verbose", 0),
    )
    search.fit(X_train, y_train)

    best_params = search.best_params_
    logger.info(
        "Meilleur score %s en validation croisée : %.4f",
        cfg["scoring"],
        search.best_score_,
    )
    logger.info("Hyperparamètres retenus : %s", best_params)

    (paths.models / "best_params.json").write_text(
        json.dumps(best_params, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (paths.models / "best_params.pkl").open("wb") as handle:
        pickle.dump(best_params, handle)

    # Trace du résultat de la recherche : suivie comme métrique par DVC, donc
    # comparable entre deux expériences avec `dvc metrics diff`.
    (paths.metrics / "tuning.json").write_text(
        json.dumps(
            {
                "estimator": cfg["estimator"],
                "scoring": cfg["scoring"],
                "cv_folds": cfg["cv_folds"],
                "best_cv_score": float(search.best_score_),
                "n_candidates": len(search.cv_results_["params"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    setup_logging()
    main()
