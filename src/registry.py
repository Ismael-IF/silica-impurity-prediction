"""Registre des modèles candidats.

Centraliser les estimateurs ici permet de les désigner par leur nom dans
`params.yaml` : changer de modèle devient un changement de configuration,
versionné par DVC, et non une modification de code.
"""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

#: Libellés lisibles pour les rapports et les graphiques.
LABELS: dict[str, str] = {
    "dummy_mean": "Référence (moyenne)",
    "ridge": "Régression Ridge",
    "knn": "k plus proches voisins",
    "svr": "SVR (noyau RBF)",
    "random_forest_exam": "Random Forest (version examen)",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "gradient_boosting": "Gradient Boosting",
    "hist_gradient_boosting": "Hist Gradient Boosting",
}


def build_estimator(name: str, seed: int = 42, **overrides: Any):
    """Instancie un estimateur du registre.

    Args:
        name: clé du registre, par exemple ``extra_trees``.
        seed: graine aléatoire, propagée aux modèles qui en acceptent une.
        **overrides: hyperparamètres appliqués par-dessus les valeurs par défaut
            (typiquement ceux issus de la recherche sur grille).

    Raises:
        KeyError: si le nom ne figure pas dans le registre.
    """
    factories = {
        "dummy_mean": lambda: DummyRegressor(strategy="mean"),
        "ridge": lambda: Ridge(alpha=1.0, random_state=None),
        "knn": lambda: KNeighborsRegressor(n_neighbors=10),
        "svr": lambda: SVR(C=10.0, gamma="scale"),
        # Reproduit à l'identique le modèle rendu pour l'examen : sert de
        # point de comparaison dans le benchmark.
        "random_forest_exam": lambda: RandomForestRegressor(
            random_state=seed,
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            min_samples_split=2,
        ),
        "random_forest": lambda: RandomForestRegressor(
            random_state=seed, n_estimators=500
        ),
        "extra_trees": lambda: ExtraTreesRegressor(random_state=seed, n_estimators=500),
        "gradient_boosting": lambda: GradientBoostingRegressor(random_state=seed),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(
            random_state=seed
        ),
    }
    if name not in factories:
        raise KeyError(f"Modèle inconnu : {name!r}. Disponibles : {sorted(factories)}")
    estimator = factories[name]()
    if overrides:
        estimator.set_params(**overrides)
    return estimator
