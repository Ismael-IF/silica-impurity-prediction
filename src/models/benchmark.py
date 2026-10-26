"""Étape optionnelle — comparaison des familles de modèles.

Cette étape est la pièce justificative du choix d'estimateur : elle mesure,
en validation croisée sur le train uniquement, ce que chaque famille apporte
par rapport à la référence « on prédit la moyenne ». Le jeu de test n'est pas
touché ici — il reste vierge pour la seule évaluation finale.
"""

from __future__ import annotations

import json
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, cross_val_score

from src import viz
from src.config import get_paths, load_params, setup_logging
from src.registry import LABELS, build_estimator

logger = logging.getLogger(__name__)


def main() -> None:
    params = load_params()
    paths = get_paths(params)
    paths.mkdirs()

    cfg = params["benchmark"]
    seed = params["seed"]

    X_train = pd.read_csv(paths.processed / "X_train_scaled.csv")
    y_train = pd.read_csv(paths.processed / "y_train.csv").squeeze("columns")
    cv = KFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)

    results = []
    for name in cfg["candidates"]:
        estimator = build_estimator(name, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            r2 = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="r2")
            rmse = -cross_val_score(
                estimator,
                X_train,
                y_train,
                cv=cv,
                scoring="neg_root_mean_squared_error",
            )
        results.append(
            {
                "modele": name,
                "libelle": LABELS.get(name, name),
                "r2_moyen": round(float(np.mean(r2)), 4),
                "r2_ecart_type": round(float(np.std(r2)), 4),
                "rmse_moyen": round(float(np.mean(rmse)), 4),
            }
        )
        logger.info(
            "%-24s R2 %.4f (+/- %.4f)  RMSE %.4f",
            name,
            np.mean(r2),
            np.std(r2),
            np.mean(rmse),
        )

    retained = params["tuning"]["estimator"]
    payload = {
        "protocole": f"validation croisee {cfg['cv_folds']} plis sur le jeu d'entrainement",
        "modele_retenu": retained,
        "resultats": sorted(results, key=lambda row: row["r2_moyen"], reverse=True),
    }
    (paths.metrics / "model_comparison.json").write_text(
        json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    viz.apply_style()
    viz.plot_model_comparison(
        [row["libelle"] for row in results],
        [row["r2_moyen"] for row in results],
        highlight=LABELS.get(retained, retained),
        path=paths.figures / "comparaison_modeles.png",
        subtitle=(
            "Validation croisée 5 plis sur le train — la référence « moyenne » "
            "matérialise le niveau zéro"
        ),
    )


if __name__ == "__main__":
    setup_logging()
    main()
