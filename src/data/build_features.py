"""Étape 2 — normalisation des variables explicatives.

Le scaler est ajusté sur le train uniquement, puis **sérialisé**. C'est le
point qui manquait à la version examen : sans scaler persisté, un modèle
déployé ne peut pas transformer une observation entrante de la même façon
qu'à l'entraînement, et les prédictions dérivent silencieusement.
"""

from __future__ import annotations

import logging

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import get_paths, load_params, setup_logging

logger = logging.getLogger(__name__)

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler}


def main() -> None:
    params = load_params()
    paths = get_paths(params)
    paths.mkdirs()

    scaler_name = params["features"]["scaler"]
    if scaler_name not in SCALERS:
        raise KeyError(
            f"Scaler inconnu : {scaler_name!r}. Disponibles : {sorted(SCALERS)}"
        )

    X_train = pd.read_csv(paths.processed / "X_train.csv")
    X_test = pd.read_csv(paths.processed / "X_test.csv")

    scaler = SCALERS[scaler_name]()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    X_train_scaled.to_csv(paths.processed / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(paths.processed / "X_test_scaled.csv", index=False)

    scaler_path = paths.models / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    logger.info(
        "Normalisation %s appliquée ; scaler conservé dans %s", scaler_name, scaler_path
    )


if __name__ == "__main__":
    setup_logging()
    main()
