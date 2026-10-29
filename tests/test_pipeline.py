"""Test d'intégration : le pipeline complet sur un jeu de données jouet.

Ces tests protègent les propriétés qu'on ne veut jamais voir régresser :
le test ne fuite pas dans l'entraînement, le scaler est bien celui du
train, et les hyperparamètres trouvés par la recherche sur grille sont
réellement ceux utilisés à l'entraînement.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import get_paths, load_params
from src.data import build_features, make_dataset
from src.models import evaluate, train, tune


@pytest.fixture
def pipeline_execute(fake_project):
    """Rejoue les cinq étapes dans l'ordre et renvoie les chemins produits."""
    make_dataset.main()
    build_features.main()
    tune.main()
    train.main()
    evaluate.main()
    return get_paths(load_params())


class TestDecoupage:
    def test_les_quatre_jeux_sont_produits(self, fake_project):
        make_dataset.main()
        paths = get_paths(load_params())
        for name in ("X_train", "X_test", "y_train", "y_test"):
            assert (paths.processed / f"{name}.csv").exists()

    def test_la_colonne_horodatage_est_retiree(self, fake_project):
        make_dataset.main()
        paths = get_paths(load_params())
        assert "date" not in pd.read_csv(paths.processed / "X_train.csv").columns

    def test_la_cible_nest_pas_dans_les_variables_explicatives(self, fake_project):
        """Fuite la plus banale et la plus destructrice."""
        make_dataset.main()
        paths = get_paths(load_params())
        assert (
            "silica_concentrate"
            not in pd.read_csv(paths.processed / "X_train.csv").columns
        )

    def test_aucune_observation_nest_perdue_ni_dupliquee(self, fake_project):
        make_dataset.main()
        paths = get_paths(load_params())
        n_train = len(pd.read_csv(paths.processed / "X_train.csv"))
        n_test = len(pd.read_csv(paths.processed / "X_test.csv"))
        n_raw = len(pd.read_csv(paths.raw))
        assert n_train + n_test == n_raw

    def test_le_decoupage_est_reproductible(self, fake_project):
        make_dataset.main()
        paths = get_paths(load_params())
        first = pd.read_csv(paths.processed / "X_test.csv")
        make_dataset.main()
        second = pd.read_csv(paths.processed / "X_test.csv")
        pd.testing.assert_frame_equal(first, second)

    def test_une_cible_absente_echoue_clairement(self, fake_project):
        raw_path = fake_project / "data" / "raw" / "raw.csv"
        frame = pd.read_csv(raw_path).drop(columns=["silica_concentrate"])
        frame.to_csv(raw_path, index=False)
        with pytest.raises(KeyError, match="cible"):
            make_dataset.main()


class TestNormalisation:
    def test_le_scaler_est_ajuste_sur_le_train_seul(self, fake_project):
        """Le train normalisé est centré-réduit ; le test, lui, ne l'est pas
        exactement — c'est précisément la preuve qu'aucun paramètre du scaler
        n'a été calculé sur le test."""
        make_dataset.main()
        build_features.main()
        paths = get_paths(load_params())

        train_scaled = pd.read_csv(paths.processed / "X_train_scaled.csv")
        assert np.allclose(train_scaled.mean(), 0, atol=1e-9)
        assert np.allclose(train_scaled.std(ddof=0), 1, atol=1e-9)

        test_scaled = pd.read_csv(paths.processed / "X_test_scaled.csv")
        assert not np.allclose(test_scaled.mean(), 0, atol=1e-6)

    def test_le_scaler_est_conserve_pour_le_service(self, fake_project):
        """Sans scaler persisté, aucune mise en production n'est possible."""
        make_dataset.main()
        build_features.main()
        paths = get_paths(load_params())
        scaler = joblib.load(paths.models / "scaler.joblib")
        raw_train = pd.read_csv(paths.processed / "X_train.csv")
        replayed = scaler.transform(raw_train)
        stored = pd.read_csv(paths.processed / "X_train_scaled.csv").to_numpy()
        assert np.allclose(replayed, stored)

    def test_les_colonnes_gardent_leur_nom_et_leur_ordre(self, fake_project):
        make_dataset.main()
        build_features.main()
        paths = get_paths(load_params())
        before = pd.read_csv(paths.processed / "X_train.csv").columns.tolist()
        after = pd.read_csv(paths.processed / "X_train_scaled.csv").columns.tolist()
        assert before == after

    def test_un_scaler_inconnu_echoue_explicitement(self, fake_project):
        params_path = fake_project / "params.yaml"
        params_path.write_text(
            params_path.read_text(encoding="utf-8").replace(
                "scaler: standard", "scaler: quantique"
            ),
            encoding="utf-8",
        )
        make_dataset.main()
        with pytest.raises(KeyError, match="Scaler inconnu"):
            build_features.main()


class TestEnchainementTuningEntrainement:
    def test_les_hyperparametres_trouves_sont_ceux_utilises(self, fake_project):
        """Le défaut corrigé par rapport à la première version : la recherche sur
        grille écrivait un fichier que l'entraînement ignorait."""
        make_dataset.main()
        build_features.main()
        tune.main()
        paths = get_paths(load_params())
        best = json.loads((paths.models / "best_params.json").read_text(encoding="utf-8"))

        train.main()
        model = joblib.load(paths.models / "model.joblib")
        for key, value in best.items():
            assert model.get_params()[key] == value

    def test_sans_tuning_lentrainement_retombe_sur_les_valeurs_de_repli(
        self, fake_project
    ):
        make_dataset.main()
        build_features.main()
        train.main()
        paths = get_paths(load_params())
        model = joblib.load(paths.models / "model.joblib")
        fallback = load_params()["train"]["fallback_params"]
        assert model.get_params()["n_estimators"] == fallback["n_estimators"]


class TestEvaluation:
    def test_les_deux_fichiers_de_metriques_sont_produits(self, pipeline_execute):
        paths = pipeline_execute
        assert (paths.metrics / "metrics.json").exists()
        assert (paths.metrics / "business_metrics.json").exists()

    def test_les_metriques_sont_du_json_serialisable(self, pipeline_execute):
        """Un NaN ou un type numpy casserait `dvc metrics show`."""
        for name in ("metrics.json", "business_metrics.json"):
            payload = json.loads(
                (pipeline_execute.metrics / name).read_text(encoding="utf-8")
            )
            json.dumps(payload)

    def test_le_modele_bat_la_reference_sur_un_signal_connu(self, pipeline_execute):
        """La cible du jeu jouet dépend réellement de deux variables : si le
        pipeline n'arrive pas à faire mieux que la moyenne, une étape casse
        la correspondance entre X et y."""
        metrics = json.loads(
            (pipeline_execute.metrics / "metrics.json").read_text(encoding="utf-8")
        )
        assert metrics["r2"] > 0.3

    def test_toutes_les_figures_du_rapport_sont_generees(self, pipeline_execute):
        attendues = {
            "predictions_vs_mesures.png",
            "distribution_erreurs.png",
            "alerte_hors_spec.png",
            "seuil_alerte.png",
            "importance_variables.png",
        }
        produites = {p.name for p in pipeline_execute.figures.glob("*.png")}
        assert attendues <= produites

    def test_les_predictions_sont_conservees_ligne_a_ligne(self, pipeline_execute):
        predictions = pd.read_csv(pipeline_execute.processed / "predictions.csv")
        y_test = pd.read_csv(pipeline_execute.processed / "y_test.csv")
        assert len(predictions) == len(y_test)
        assert np.allclose(
            predictions["erreur"],
            predictions["silice_mesuree"] - predictions["silice_predite"],
        )
