"""Tests unitaires : configuration, registre de modèles, métriques."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import get_paths, load_params, project_root
from src.models.evaluate import (
    alert_sweep,
    calibrate_alert_threshold,
    operational_metrics,
    statistical_metrics,
)
from src.registry import LABELS, build_estimator


class TestConfiguration:
    def test_params_contient_toutes_les_sections_attendues(self):
        params = load_params()
        for section in ("seed", "data", "features", "tuning", "train", "evaluation"):
            assert section in params, f"section absente de params.yaml : {section}"

    def test_les_chemins_sont_sous_la_racine_du_projet(self):
        paths = get_paths()
        root = project_root()
        for path in (paths.raw, paths.processed, paths.models, paths.metrics):
            assert root in path.parents or path == root

    def test_la_racine_suit_la_variable_denvironnement(self, tmp_path, monkeypatch):
        from src.config import ROOT_ENV_VAR

        monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
        assert project_root() == tmp_path.resolve()


class TestRegistre:
    def test_tous_les_candidats_du_benchmark_sont_constructibles(self):
        params = load_params()
        for name in params["benchmark"]["candidates"]:
            assert build_estimator(name, seed=42) is not None

    def test_chaque_candidat_a_un_libelle_lisible(self):
        params = load_params()
        for name in params["benchmark"]["candidates"]:
            assert name in LABELS

    def test_un_nom_inconnu_echoue_explicitement(self):
        with pytest.raises(KeyError, match="Modèle inconnu"):
            build_estimator("reseau_de_neurones_magique")

    def test_les_surcharges_sont_appliquees(self):
        estimator = build_estimator("extra_trees", seed=1, n_estimators=77)
        assert estimator.get_params()["n_estimators"] == 77

    def test_le_modele_de_lexamen_garde_ses_hyperparametres(self):
        """Le point de comparaison doit rester figé, sinon le benchmark ment."""
        params = build_estimator("random_forest_exam", seed=42).get_params()
        assert params["n_estimators"] == 200
        assert params["max_depth"] == 10
        assert params["min_samples_leaf"] == 2


class TestMetriquesStatistiques:
    def test_prediction_parfaite(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = statistical_metrics(y, y)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)

    def test_valeurs_connues(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 2.5, 3.5])
        metrics = statistical_metrics(y_true, y_pred)
        assert metrics["mae"] == pytest.approx(0.5)
        assert metrics["mse"] == pytest.approx(0.25)
        assert metrics["rmse"] == pytest.approx(0.5)


class TestMetriquesOperationnelles:
    def test_comptage_des_alertes(self):
        y_true = np.array([1.0, 3.5, 3.5, 1.0])
        y_pred = np.array([1.0, 3.5, 1.0, 3.5])
        result = operational_metrics(
            y_true, y_pred, baseline_prediction=2.0, tolerance_pp=0.5, spec_limit_pp=3.0
        )
        alert = result["alerte_hors_spec"]
        assert alert["heures_hors_spec_reelles"] == 2
        assert alert["detectees"] == 1
        assert alert["manquees"] == 1
        assert alert["fausses_alertes"] == 1
        assert alert["rappel"] == pytest.approx(0.5)
        assert alert["precision"] == pytest.approx(0.5)

    def test_la_part_dans_la_tolerance_est_un_pourcentage(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 5.0, 6.0])
        result = operational_metrics(
            y_true, y_pred, baseline_prediction=2.5, tolerance_pp=0.5, spec_limit_pp=3.0
        )
        assert result["part_heures_dans_tolerance_pct"] == pytest.approx(50.0)

    def test_sans_aucune_alerte_predite_la_precision_est_indefinie(self):
        """Diviser par zéro renverrait NaN dans le JSON : on renvoie None."""
        y_true = np.array([1.0, 1.2])
        y_pred = np.array([1.0, 1.1])
        alert = operational_metrics(
            y_true, y_pred, baseline_prediction=1.0, tolerance_pp=0.5, spec_limit_pp=3.0
        )["alerte_hors_spec"]
        assert alert["precision"] is None


class TestCalibrationDuSeuil:
    def _sweep(self):
        # Cible : 4 heures hors spec sur 10.
        y_true = np.array([1.0, 1.5, 2.0, 2.2, 2.5, 2.8, 3.2, 3.5, 4.0, 4.5])
        y_pred = np.array([1.2, 1.4, 1.9, 2.1, 2.3, 2.4, 2.6, 2.7, 2.9, 3.1])
        return alert_sweep(y_true, y_pred, 3.0, np.arange(2.0, 3.2, 0.2))

    def test_le_rappel_decroit_quand_le_seuil_monte(self):
        rappels = [row["rappel"] for row in self._sweep()]
        assert rappels == sorted(rappels, reverse=True)

    def test_on_retient_le_seuil_le_plus_haut_atteignant_lobjectif(self):
        sweep = self._sweep()
        chosen = calibrate_alert_threshold(sweep, recall_target=0.75)
        eligible = [row["seuil_pp"] for row in sweep if row["rappel"] >= 0.75]
        assert chosen["objectif_rappel_atteint"] is True
        assert chosen["seuil_pp"] == max(eligible)

    def test_objectif_inatteignable_signale_et_maximise_la_detection(self):
        sweep = self._sweep()
        chosen = calibrate_alert_threshold(sweep, recall_target=1.01)
        assert chosen["objectif_rappel_atteint"] is False
        assert chosen["rappel"] == max(row["rappel"] for row in sweep)
