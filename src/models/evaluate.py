"""Étape 5 — évaluation et traduction en indicateurs d'exploitation.

Deux fichiers de métriques sont produits volontairement :

* `metrics/metrics.json` — les métriques statistiques (MSE, RMSE, MAE, R²),
  celles qui servent à arbitrer entre deux versions du modèle ;
* `metrics/business_metrics.json` — ce qu'un responsable de production
  regarde : part des heures prédites dans la tolérance, gain d'erreur face
  au réflexe « on prend la moyenne », et capacité à lever une alerte avant
  que le lot ne sorte hors spécification.

Un modèle peut gagner 0,02 de R² et ne rien changer en salle de contrôle.
C'est le second fichier qui tranche.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from joblib import load
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src import viz
from src.config import get_paths, load_params, setup_logging

logger = logging.getLogger(__name__)


def statistical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Métriques de régression classiques."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def operational_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_prediction: float,
    tolerance_pp: float,
    spec_limit_pp: float,
) -> dict:
    """Traduit les erreurs en indicateurs utilisables par l'exploitation.

    Args:
        y_true: teneurs en silice mesurées au laboratoire.
        y_pred: teneurs prédites.
        baseline_prediction: valeur du réflexe de référence, c'est-à-dire la
            moyenne observée sur le train. C'est le vrai concurrent du modèle :
            sans modèle, c'est ce que l'opérateur suppose.
        tolerance_pp: écart acceptable, en points de pourcentage.
        spec_limit_pp: seuil contractuel au-delà duquel l'heure est hors spec.
    """
    errors = y_true - y_pred
    absolute_errors = np.abs(errors)

    rmse = float(np.sqrt(np.mean(errors**2)))
    baseline_rmse = float(np.sqrt(np.mean((y_true - baseline_prediction) ** 2)))

    actual_alert = y_true > spec_limit_pp
    predicted_alert = y_pred > spec_limit_pp
    true_positive = int(np.sum(actual_alert & predicted_alert))
    false_positive = int(np.sum(~actual_alert & predicted_alert))
    false_negative = int(np.sum(actual_alert & ~predicted_alert))
    true_negative = int(np.sum(~actual_alert & ~predicted_alert))

    def ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    return {
        "n_heures_test": int(len(y_true)),
        "tolerance_pp": tolerance_pp,
        "part_heures_dans_tolerance_pct": round(
            100 * float(np.mean(absolute_errors <= tolerance_pp)), 1
        ),
        "erreur_absolue_mediane_pp": round(float(np.median(absolute_errors)), 4),
        "erreur_absolue_p90_pp": round(float(np.percentile(absolute_errors, 90)), 4),
        "biais_moyen_pp": round(float(np.mean(errors)), 4),
        "reference_moyenne_train_pp": round(baseline_prediction, 4),
        "rmse_reference_moyenne_pp": round(baseline_rmse, 4),
        "rmse_modele_pp": round(rmse, 4),
        "reduction_erreur_vs_reference_pct": round(
            100 * (baseline_rmse - rmse) / baseline_rmse, 1
        ),
        "alerte_hors_spec": {
            "seuil_pp": spec_limit_pp,
            "heures_hors_spec_reelles": int(np.sum(actual_alert)),
            "detectees": true_positive,
            "manquees": false_negative,
            "fausses_alertes": false_positive,
            "conformes_bien_classees": true_negative,
            "rappel": ratio(true_positive, true_positive + false_negative),
            "precision": ratio(true_positive, true_positive + false_positive),
        },
    }


def alert_sweep(
    y_true: np.ndarray, y_pred: np.ndarray, spec_limit: float, thresholds: np.ndarray
) -> list[dict]:
    """Rappel et précision de l'alerte pour chaque seuil de déclenchement."""
    actual_alert = y_true > spec_limit
    n_actual = int(np.sum(actual_alert))
    sweep = []
    for threshold in thresholds:
        predicted_alert = y_pred > threshold
        true_positive = int(np.sum(actual_alert & predicted_alert))
        false_positive = int(np.sum(~actual_alert & predicted_alert))
        n_alerts = true_positive + false_positive
        sweep.append(
            {
                "seuil_pp": round(float(threshold), 3),
                "alertes_levees": n_alerts,
                "detectees": true_positive,
                "manquees": n_actual - true_positive,
                "fausses_alertes": false_positive,
                "rappel": round(true_positive / n_actual, 4) if n_actual else 0.0,
                "precision": round(true_positive / n_alerts, 4) if n_alerts else 0.0,
            }
        )
    return sweep


def calibrate_alert_threshold(sweep: list[dict], recall_target: float) -> dict:
    """Choisit le seuil de déclenchement de l'alerte.

    Le réflexe naturel consiste à alerter dès que la prédiction dépasse la
    limite contractuelle. C'est un mauvais réglage : un modèle de régression
    tire ses prédictions vers la moyenne, donc il sous-estime les pics et
    laisse passer la majorité des heures réellement hors spécification.

    On prend donc le seuil **le plus haut** qui atteigne encore l'objectif de
    détection : parmi les réglages qui tiennent la promesse faite à
    l'exploitation, c'est celui qui lève le moins de fausses alertes. Si aucun
    seuil n'atteint l'objectif, on retient le plus détectant.
    """
    eligible = [row for row in sweep if row["rappel"] >= recall_target]
    if eligible:
        chosen = max(eligible, key=lambda row: row["seuil_pp"])
        return {"objectif_rappel_atteint": True, **chosen}
    chosen = max(sweep, key=lambda row: (row["rappel"], row["precision"]))
    return {"objectif_rappel_atteint": False, **chosen}


def main() -> None:
    params = load_params()
    paths = get_paths(params)
    paths.mkdirs()

    eval_cfg = params["evaluation"]
    tolerance = float(eval_cfg["tolerance_pp"])
    spec_limit = float(eval_cfg["spec_limit_pp"])
    recall_target = float(eval_cfg["alert_recall_target"])
    grid_cfg = eval_cfg["alert_threshold_grid"]
    thresholds = np.arange(
        float(grid_cfg["start"]), float(grid_cfg["stop"]), float(grid_cfg["step"])
    )

    X_test = pd.read_csv(paths.processed / "X_test_scaled.csv")
    y_test = pd.read_csv(paths.processed / "y_test.csv").squeeze("columns").to_numpy()
    y_train = pd.read_csv(paths.processed / "y_train.csv").squeeze("columns").to_numpy()

    model = load(paths.models / "model.joblib")
    predictions = model.predict(X_test)

    metrics = statistical_metrics(y_test, predictions)
    business = operational_metrics(
        y_test,
        predictions,
        baseline_prediction=float(np.mean(y_train)),
        tolerance_pp=tolerance,
        spec_limit_pp=spec_limit,
    )

    sweep = alert_sweep(y_test, predictions, spec_limit, thresholds)
    calibrated = calibrate_alert_threshold(sweep, recall_target)
    business["alerte_calibree"] = {
        "objectif_rappel": recall_target,
        **calibrated,
        "lecture": (
            f"Alerter dès que la prédiction dépasse {calibrated['seuil_pp']} % "
            f"(et non la limite contractuelle de {spec_limit} %) détecte "
            f"{calibrated['detectees']} des "
            f"{calibrated['detectees'] + calibrated['manquees']} heures hors "
            f"spécification, au prix de {calibrated['fausses_alertes']} fausses "
            "alertes."
        ),
    }

    (paths.metrics / "metrics.json").write_text(
        json.dumps(metrics, indent=4), encoding="utf-8"
    )
    (paths.metrics / "business_metrics.json").write_text(
        json.dumps(business, indent=4, ensure_ascii=False), encoding="utf-8"
    )
    (paths.metrics / "alert_threshold_sweep.json").write_text(
        json.dumps(sweep, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Les prédictions sont conservées : elles permettent de rejouer une
    # analyse d'erreur sans réentraîner, et de comparer deux versions ligne
    # à ligne.
    pd.DataFrame(
        {
            "silice_mesuree": y_test,
            "silice_predite": predictions,
            "erreur": y_test - predictions,
        }
    ).to_csv(paths.processed / "predictions.csv", index=False)

    # --- figures du rapport -------------------------------------------------
    viz.apply_style()
    viz.plot_pred_vs_actual(
        y_test,
        predictions,
        tolerance,
        paths.figures / "predictions_vs_mesures.png",
        subtitle=(
            f"{business['n_heures_test']} heures du jeu de test • "
            f"{business['part_heures_dans_tolerance_pct']} % des heures dans la "
            f"tolérance de ±{tolerance} point"
        ),
    )
    viz.plot_residuals(
        y_test - predictions,
        paths.figures / "distribution_erreurs.png",
        subtitle=(
            f"RMSE {metrics['rmse']:.3f} point contre "
            f"{business['rmse_reference_moyenne_pp']:.3f} pour la référence « moyenne »"
        ),
    )

    alert = business["alerte_hors_spec"]
    viz.plot_alert_matrix(
        [
            [alert["conformes_bien_classees"], alert["fausses_alertes"]],
            [alert["manquees"], alert["detectees"]],
        ],
        spec_limit,
        paths.figures / "alerte_hors_spec.png",
        subtitle=(
            f"Rappel {alert['rappel']} • précision {alert['precision']} — "
            "la lecture utile pour l'exploitation"
        ),
    )

    viz.plot_alert_tradeoff(
        [row["seuil_pp"] for row in sweep],
        [row["rappel"] for row in sweep],
        [row["precision"] for row in sweep],
        chosen=calibrated["seuil_pp"],
        spec_limit=spec_limit,
        path=paths.figures / "seuil_alerte.png",
        subtitle=(
            f"Objectif de détection fixé à {recall_target:.0%} des heures hors "
            "spécification — le seuil est un choix d'exploitation, pas un réglage statistique"
        ),
    )

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=params["seed"],
        scoring="r2",
        n_jobs=params["n_jobs"],
    )
    viz.plot_importances(
        list(X_test.columns),
        importance.importances_mean,
        importance.importances_std,
        paths.figures / "importance_variables.png",
        subtitle="Importance par permutation sur le jeu de test, 20 répétitions",
    )

    logger.info("Métriques statistiques : %s", metrics)
    logger.info(
        "Heures dans la tolérance : %s %% — réduction d'erreur vs référence : %s %%",
        business["part_heures_dans_tolerance_pct"],
        business["reduction_erreur_vs_reference_pct"],
    )
    logger.info("Alerte calibrée : %s", business["alerte_calibree"]["lecture"])


if __name__ == "__main__":
    setup_logging()
    main()
