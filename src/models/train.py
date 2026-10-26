"""Étape 4 — entraînement du modèle final.

Les hyperparamètres viennent du fichier produit par l'étape de tuning. Si ce
fichier est absent (exécution partielle, environnement de test), on retombe
sur les valeurs de repli de `params.yaml` plutôt que d'échouer : le pipeline
reste exécutable bout en bout dans la CI sans payer la recherche sur grille.
"""

from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

from src.config import get_paths, load_params, setup_logging
from src.registry import build_estimator

logger = logging.getLogger(__name__)


def resolve_hyperparams(params: dict, best_params_path) -> dict:
    """Renvoie les hyperparamètres à utiliser et journalise leur provenance."""
    if best_params_path.exists():
        hyperparams = json.loads(best_params_path.read_text(encoding="utf-8"))
        logger.info("Hyperparamètres issus du tuning : %s", hyperparams)
        return hyperparams
    hyperparams = params["train"]["fallback_params"]
    logger.warning(
        "%s absent — repli sur les valeurs de params.yaml : %s",
        best_params_path.name,
        hyperparams,
    )
    return hyperparams


def main() -> None:
    params = load_params()
    paths = get_paths(params)
    paths.mkdirs()

    X_train = pd.read_csv(paths.processed / "X_train_scaled.csv")
    y_train = pd.read_csv(paths.processed / "y_train.csv").squeeze("columns")

    hyperparams = resolve_hyperparams(params, paths.models / "best_params.json")
    model = build_estimator(
        params["tuning"]["estimator"], seed=params["seed"], **hyperparams
    )
    model.fit(X_train, y_train)

    model_path = paths.models / "model.joblib"
    joblib.dump(model, model_path)
    logger.info(
        "Modèle %s entraîné sur %d observations et enregistré dans %s",
        params["tuning"]["estimator"],
        len(X_train),
        model_path,
    )


if __name__ == "__main__":
    setup_logging()
    main()
