"""Rendu des graphiques du rapport.

Une seule feuille de style pour toutes les figures : fond clair explicite
(lisible aussi bien sur un README en thème sombre que clair), grille
discrète, texte en encre neutre et couleur réservée à la donnée.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Palette validée pour la vision des couleurs : écart suffisant entre les
# teintes employées côte à côte, y compris en déficience de perception.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_FAINT = "#8c8b86"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
BLUE_LIGHT = "#9ec5f4"
ORANGE = "#eb6834"

BASE_STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 14,
    "axes.labelsize": 9.5,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "font.size": 10,
    "figure.dpi": 150,
}


def apply_style() -> None:
    """Active la feuille de style du projet."""
    plt.rcParams.update(BASE_STYLE)


def _save(fig, path: Path, subtitle: str | None) -> None:
    if subtitle:
        fig.text(0.012, 0.012, subtitle, color=INK_FAINT, fontsize=8.5, va="bottom")
        fig.subplots_adjust(bottom=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _declutter(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_model_comparison(labels, scores, highlight, path, subtitle=None) -> None:
    """Barres horizontales : un modèle par barre, le retenu en évidence.

    Le R² est comparable entre modèles ; la barre horizontale laisse la place
    aux noms complets et le tri rend la lecture immédiate.
    """
    order = np.argsort(scores)
    labels = [labels[i] for i in order]
    scores = [float(scores[i]) for i in order]
    colors = [ORANGE if lab == highlight else BLUE for lab in labels]

    fig, ax = plt.subplots(figsize=(8.4, 0.52 * len(labels) + 2.0))
    bars = ax.barh(labels, scores, color=colors, height=0.62)
    span = max(scores) - min(min(scores), 0.0)
    for bar, score in zip(bars, scores, strict=True):
        offset = max(span, 1e-6) * 0.02
        inside = score > span * 0.25
        ax.text(
            score + (-offset if inside else offset),
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            ha="right" if inside else "left",
            color=SURFACE if inside else INK_SOFT,
            fontsize=9,
            fontweight="bold" if inside else "normal",
        )
    ax.set_title("Qualité prédictive par modèle")
    ax.set_xlabel("R² moyen en validation croisée (5 plis) — plus haut, mieux")
    ax.axvline(0, color=INK_FAINT, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(min(scores) * 1.3, -0.02), max(scores) * 1.2)
    _declutter(ax)
    fig.tight_layout()
    _save(fig, path, subtitle)


def plot_pred_vs_actual(y_true, y_pred, tolerance, path, subtitle=None) -> None:
    """Nuage prédiction / mesure : où le modèle suit, où il écrase les extrêmes."""
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    lo = float(min(np.min(y_true), np.min(y_pred))) - 0.15
    hi = float(max(np.max(y_true), np.max(y_pred))) + 0.15

    ax.fill_between(
        [lo, hi],
        [lo - tolerance, hi - tolerance],
        [lo + tolerance, hi + tolerance],
        color=BLUE_LIGHT,
        alpha=0.45,
        linewidth=0,
        label=f"tolérance ±{tolerance} pt",
    )
    ax.plot(
        [lo, hi], [lo, hi], color=INK_FAINT, linewidth=1.2, label="prédiction parfaite"
    )
    ax.scatter(
        y_true,
        y_pred,
        s=26,
        color=BLUE,
        alpha=0.65,
        edgecolor=SURFACE,
        linewidth=0.6,
        label="heures de production (test)",
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title("Prédiction face à la mesure laboratoire")
    ax.set_xlabel("Silice mesurée au laboratoire (% massique)")
    ax.set_ylabel("Silice prédite (% massique)")
    ax.legend(loc="upper left")
    _declutter(ax)
    fig.tight_layout()
    _save(fig, path, subtitle)


def plot_residuals(residuals, path, subtitle=None) -> None:
    """Distribution des erreurs signées : biais et dispersion en une image."""
    residuals = np.asarray(residuals, dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.hist(residuals, bins=34, color=BLUE, alpha=0.9, edgecolor=SURFACE, linewidth=0.8)
    ax.axvline(0, color=INK_FAINT, linewidth=1.1)
    ax.axvline(
        float(residuals.mean()),
        color=ORANGE,
        linewidth=1.6,
        linestyle="--",
        label=f"biais moyen : {residuals.mean():+.3f} pt",
    )
    ax.set_title("Distribution de l'erreur de prédiction")
    ax.set_xlabel("Erreur = mesure − prédiction (points de % de silice)")
    ax.set_ylabel("Nombre d'heures")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    _declutter(ax)
    fig.tight_layout()
    _save(fig, path, subtitle)


def plot_importances(names, means, stds, path, subtitle=None) -> None:
    """Importance par permutation : perte de R² quand la variable est brouillée."""
    order = np.argsort(means)
    names = [names[i] for i in order]
    means = np.asarray(means, dtype=float)[order]
    stds = np.asarray(stds, dtype=float)[order]

    fig, ax = plt.subplots(figsize=(7.8, 0.5 * len(names) + 2.0))
    ax.barh(
        names,
        means,
        xerr=stds,
        color=BLUE,
        height=0.6,
        error_kw={"ecolor": INK_FAINT, "elinewidth": 1, "capsize": 3},
    )
    ax.set_title("Variables qui portent la prédiction")
    ax.set_xlabel("Baisse de R² lorsque la variable est rendue aléatoire")
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=INK_FAINT, linewidth=0.8)
    _declutter(ax)
    fig.tight_layout()
    _save(fig, path, subtitle)


def plot_alert_tradeoff(
    thresholds, recalls, precisions, chosen, spec_limit, path, subtitle=None
) -> None:
    """Arbitrage détection / fausses alertes selon le seuil de déclenchement.

    Deux courbes sur une seule échelle (toutes deux des proportions), donc
    pas de second axe : la lecture reste honnête. Le seuil retenu et la
    limite contractuelle sont annotés directement.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    ax.plot(
        thresholds,
        recalls,
        color=BLUE,
        linewidth=2.0,
        label="rappel — part des heures hors spec détectées",
    )
    ax.plot(
        thresholds,
        precisions,
        color=ORANGE,
        linewidth=2.0,
        label="précision — part des alertes justifiées",
    )

    ax.axvline(spec_limit, color=INK_FAINT, linewidth=1.0, linestyle=":")
    ax.annotate(
        f"limite contractuelle\n{spec_limit} %",
        xy=(spec_limit, 0.93),
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8.5,
        color=INK_FAINT,
    )

    index = int(np.argmin(np.abs(thresholds - chosen)))
    ax.scatter(
        [thresholds[index]],
        [recalls[index]],
        s=90,
        color=BLUE,
        edgecolor=SURFACE,
        linewidth=2,
        zorder=5,
    )
    ax.annotate(
        f"seuil retenu {chosen:.2f} %\nrappel {recalls[index]:.0%}, précision {precisions[index]:.0%}",
        xy=(thresholds[index], recalls[index]),
        xytext=(10, 14),
        textcoords="offset points",
        fontsize=9,
        color=INK,
        fontweight="bold",
    )

    ax.set_title("Seuil d'alerte : ce que chaque réglage coûte et rapporte")
    ax.set_xlabel("Seuil de déclenchement appliqué à la prédiction (% de silice)")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(float(thresholds.min()), float(thresholds.max()))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left")
    _declutter(ax)
    fig.tight_layout()
    _save(fig, path, subtitle)


def plot_alert_matrix(matrix, spec_limit, path, subtitle=None) -> None:
    """Matrice de décision au seuil contractuel, en heures de production.

    Rampe séquentielle à teinte unique : la couleur ne porte que la magnitude.
    """
    matrix = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max() * 1.05)

    ticks = [f"conforme\n(≤ {spec_limit} %)", f"hors spec\n(> {spec_limit} %)"]
    ax.set_xticks([0, 1], ticks)
    ax.set_yticks([0, 1], ticks)
    ax.set_xlabel("Prédiction du modèle")
    ax.set_ylabel("Réalité laboratoire")
    ax.set_title(f"Détection du hors-spécification (seuil {spec_limit} % de silice)")
    ax.grid(visible=False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    threshold = matrix.max() / 2
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(
            col,
            row,
            f"{int(value)} h",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=SURFACE if value > threshold else INK,
        )
    fig.tight_layout()
    _save(fig, path, subtitle)
