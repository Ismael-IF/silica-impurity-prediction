"""Fixtures partagées."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    """Projet jouet complet : données brutes synthétiques et params.yaml réduit.

    Le pipeline réel est rejoué dessus en quelques secondes, ce qui permet
    de tester l'enchaînement des étapes sans dépendre du jeu de données de
    production ni écraser ses artefacts.
    """
    from src.config import ROOT_ENV_VAR

    rng = np.random.default_rng(7)
    n = 240
    features = pd.DataFrame(
        {
            "ave_flot_air_flow": rng.normal(296, 6, n),
            "ave_flot_level": rng.normal(458, 67, n),
            "iron_feed": rng.normal(57, 6, n),
            "starch_flow": rng.normal(3107, 831, n),
            "amina_flow": rng.normal(502, 66, n),
            "ore_pulp_flow": rng.normal(400, 1, n),
            "ore_pulp_pH": rng.normal(9.9, 0.3, n),
            "ore_pulp_density": rng.normal(1.68, 0.05, n),
        }
    )
    # Cible construite à partir de deux variables : le pipeline doit la
    # retrouver, sinon c'est qu'une étape casse le lien entre X et y.
    features["date"] = pd.date_range("2017-03-01", periods=n, freq="h")
    target = (
        0.01 * features["amina_flow"]
        - 0.02 * features["ave_flot_air_flow"]
        + rng.normal(0, 0.3, n)
        + 3.0
    )
    raw = features.copy()
    raw["silica_concentrate"] = target

    (tmp_path / "data" / "raw").mkdir(parents=True)
    raw.to_csv(tmp_path / "data" / "raw" / "raw.csv", index=False)

    params = {
        "seed": 42,
        "n_jobs": 1,
        "data": {
            "raw_path": "data/raw/raw.csv",
            "processed_dir": "data/processed",
            "target": "silica_concentrate",
            "drop_columns": ["date"],
            "test_size": 0.25,
        },
        "features": {"scaler": "standard"},
        "tuning": {
            "cv_folds": 2,
            "scoring": "r2",
            "verbose": 0,
            "estimator": "extra_trees",
            "param_grid": {"n_estimators": [20], "max_depth": [None]},
        },
        "train": {"fallback_params": {"n_estimators": 20}},
        "evaluation": {
            "tolerance_pp": 0.5,
            "spec_limit_pp": 3.0,
            "alert_recall_target": 0.7,
            "alert_threshold_grid": {"start": 1.5, "stop": 3.3, "step": 0.25},
        },
        "benchmark": {"cv_folds": 2, "candidates": ["dummy_mean", "extra_trees"]},
    }
    (tmp_path / "params.yaml").write_text(
        yaml.safe_dump(params, allow_unicode=True), encoding="utf-8"
    )

    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path))
    return tmp_path
