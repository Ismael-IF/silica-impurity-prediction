"""Chargement de la configuration et résolution des chemins.

Toutes les étapes du pipeline passent par ce module : aucun chemin absolu,
aucune constante dupliquée. La racine du projet est déduite de
l'emplacement du fichier, donc le code tourne à l'identique en local, dans
un conteneur ou dans la CI. La variable d'environnement
``SILICA_PROJECT_ROOT`` permet de la déplacer — c'est ce qui rend le
pipeline testable sur un jeu de données jouet, sans toucher aux artefacts
réels.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT_ENV_VAR = "SILICA_PROJECT_ROOT"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    """Racine du projet, surchargeable par variable d'environnement."""
    override = os.environ.get(ROOT_ENV_VAR)
    return Path(override).resolve() if override else DEFAULT_PROJECT_ROOT


@dataclass(frozen=True)
class Paths:
    """Emplacements standards, tous relatifs à la racine du projet."""

    root: Path
    raw: Path
    processed: Path
    models: Path
    metrics: Path
    figures: Path

    def mkdirs(self) -> None:
        for directory in (self.processed, self.models, self.metrics, self.figures):
            directory.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure les journaux des étapes du pipeline.

    La console Windows est en page de code 1252 par défaut : sans ce
    forçage en UTF-8, les accents des messages ressortent en caractères
    parasites dans les journaux d'exécution et dans la CI.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=level, format="%(levelname)-7s %(message)s")


def load_params(path: Path | str | None = None) -> dict[str, Any]:
    """Lit `params.yaml` et renvoie son contenu."""
    params_path = Path(path) if path else project_root() / "params.yaml"
    with params_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_paths(params: dict[str, Any] | None = None) -> Paths:
    """Construit les chemins du projet à partir de la configuration."""
    params = params or load_params()
    data_cfg = params["data"]
    root = project_root()
    return Paths(
        root=root,
        raw=root / data_cfg["raw_path"],
        processed=root / data_cfg["processed_dir"],
        models=root / "models",
        metrics=root / "metrics",
        figures=root / "reports" / "figures",
    )
