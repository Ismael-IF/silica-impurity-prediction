# Fiche modèle

## Identification

| | |
|---|---|
| **Nom** | Prédicteur d'impureté en silice — usine de flottation |
| **Version** | 1.0.0 |
| **Type** | Régression supervisée |
| **Algorithme** | Extra Trees (`sklearn.ensemble.ExtraTreesRegressor`) |
| **Sortie** | Teneur en silice du concentré, en % massique |
| **Auteur** | Ismael Foudjo |
| **Licence** | MIT |

### Hyperparamètres retenus

Issus d'une recherche sur grille (16 combinaisons, validation croisée 5 plis
sur le jeu d'entraînement uniquement) :

```json
{
  "max_depth": 20,
  "max_features": 0.6,
  "min_samples_leaf": 1,
  "n_estimators": 600
}
```

Ils ne sont pas recopiés dans le code : `tune` les écrit, `train` les relit.

## Usage prévu

### Ce pour quoi le modèle est conçu

Fournir à un opérateur de salle de contrôle une **estimation de la teneur en
silice de l'heure en cours**, avant que l'analyse laboratoire ne soit
disponible, afin qu'il puisse corriger le dosage des réactifs pendant que
l'heure est encore en cours.

La sortie opérationnelle attendue n'est pas le nombre mais **l'alerte** :
« cette heure risque de sortir hors spécification ».

### Utilisateurs visés

Opérateurs de flottation et ingénieurs procédé, qui connaissent le
fonctionnement de la ligne et sauront pondérer l'alerte avec ce qu'ils
observent par ailleurs.

### Usages explicitement hors périmètre

- **Pilotage automatique en boucle fermée.** Avec un R² de 0,26, envoyer les
  prédictions directement à un automate serait irresponsable.
- **Base contractuelle.** L'analyse laboratoire reste la mesure de référence
  vis-à-vis du client. Le modèle ne la remplace pas.
- **Interprétation causale.** L'importance des variables indique où le modèle
  puise son signal, pas ce qui cause la silice. Modifier un réglage *parce que
  le modèle y est sensible* est un contresens méthodologique.
- **Transposition à une autre usine.** Le modèle est calé sur une ligne, un
  minerai et une plage de réglages donnés. Sur un autre site, il faut
  réentraîner.

## Données

| | |
|---|---|
| **Source** | Relevés horaires d'une usine de flottation de minerai de fer, dérivés du jeu public *Quality Prediction in a Mining Process* |
| **Volume** | 1 817 heures de production |
| **Variables** | 8 mesures de capteurs |
| **Cible** | `silica_concentrate` — % massique de silice dans le concentré |
| **Valeurs manquantes** | aucune |
| **Doublons** | aucun |

### Distribution de la cible

| | Valeur |
|---|---|
| Moyenne | 2,16 % |
| Écart-type | 1,05 |
| Minimum | 0,85 % |
| Médiane | 1,82 % |
| Maximum | 5,33 % |

Distribution asymétrique à droite : les heures très chargées en silice sont
rares, ce qui est précisément ce qui rend leur détection difficile.

### Corrélation linéaire des variables avec la cible

| Variable | r |
|---|---:|
| `ave_flot_air_flow` | −0,235 |
| `ave_flot_level` | −0,228 |
| `amina_flow` | +0,221 |
| `ore_pulp_pH` | −0,096 |
| `ore_pulp_density` | +0,071 |
| `ore_pulp_flow` | −0,066 |
| `starch_flow` | −0,043 |
| `iron_feed` | −0,040 |

**Aucune corrélation linéaire ne dépasse 0,24.** C'est l'information la plus
importante de cette fiche : elle borne ce qu'on peut raisonnablement attendre,
et explique pourquoi un modèle non linéaire bien réglé plafonne autour de 0,26
de R². Le signal est réel mais faible.

### Découpage

80 % entraînement (1 453 heures) / 20 % test (364 heures), tirage aléatoire à
graine fixe. Le découpage précède toute transformation ; le scaler est ajusté
sur l'entraînement seul.

### Reconstitution du fichier brut

`data/raw/raw.csv` a été recomposé à partir du cache DVC de la première
version du pipeline, en
réassemblant les quatre jeux découpés. La colonne d'horodatage, que le pipeline
supprime de toute façon, n'y figure pas — ce qui interdit toute analyse
temporelle (voir *Limites*).

## Performance

Mesurée sur les 364 heures du jeu de test, jamais vues à l'entraînement.

### Métriques statistiques

| Métrique | Référence « moyenne » | Modèle |
|---|---:|---:|
| RMSE (points de %) | 1,023 | **0,878** |
| MAE | — | 0,678 |
| R² | 0,000 | **0,264** |
| Biais moyen | — | +0,001 |

Le biais quasi nul indique que le modèle ne sur- ni ne sous-estime
systématiquement — son erreur est dispersée, pas décalée.

### Métriques opérationnelles

| Indicateur | Valeur |
|---|---:|
| Heures prédites à ±0,5 point | 47 % |
| Erreur absolue médiane | 0,54 point |
| Erreur absolue au 90ᵉ centile | 1,44 point |
| Réduction d'erreur vs référence | 14,2 % |

### Détection du hors-spécification

76 des 364 heures de test dépassent réellement 3 % de silice.

| Seuil d'alerte | Détectées | Manquées | Fausses alertes | Rappel | Précision |
|---|---:|---:|---:|---:|---:|
| 3,00 % (limite contractuelle) | 21 | 55 | 9 | 28 % | 70 % |
| **2,15 % (calibré)** | **54** | **22** | **105** | **71 %** | **34 %** |

Le réglage calibré est celui retenu. Le raisonnement économique est détaillé
dans [`01-cas-metier.md`](01-cas-metier.md).

### Comparaison aux alternatives

Validation croisée 5 plis sur l'entraînement — voir
[`04-resultats.md`](04-resultats.md) pour le tableau complet. Extra Trees
devance Random Forest (0,247 contre 0,211) et, à performance voisine, présente
un écart-type inter-plis plus faible (0,047 contre 0,055) : c'est le modèle le
plus stable, donc le plus sûr en production.

## Limites

### Le plafond vient des données

Les huit variables sont des **moyennes horaires**. Toute la dynamique interne
de l'heure — montées, oscillations, transitoires — est perdue avant même
d'arriver au modèle. Le gain principal viendrait des mesures brutes à
20 secondes et de variables décalées dans le temps, absentes de ce jeu.

### Pas d'ordre temporel

Sans horodatage exploitable, impossible de faire une validation croisée
temporelle. Le protocole retenu suppose les heures indépendantes — **hypothèse
discutable sur un procédé continu**, où deux heures consécutives se
ressemblent. Les performances annoncées sont donc probablement un peu
optimistes par rapport à un déploiement réel.

### Les extrêmes sont écrasés

Comportement structurel d'un modèle entraîné à minimiser l'erreur quadratique :
il tire vers la moyenne. Les heures les plus chargées en silice sont
systématiquement sous-estimées — ce que la calibration du seuil compense, mais
ne corrige pas.

### Pas de surveillance de dérive

Un changement de gisement, de minerai ou de fournisseur de réactif déplacerait
la distribution des capteurs. Rien dans le projet actuel ne le détecterait. Un
suivi des distributions d'entrée est un prérequis à tout usage réel.

### Volume modeste

1 817 observations, dont 364 en test et seulement 76 heures hors spécification.
Les métriques de détection reposent donc sur un petit effectif : leur intervalle
de confiance est large.

## Reproductibilité

| | |
|---|---|
| Graine aléatoire | 42, propagée à toutes les étapes |
| Pipeline | `dvc repro` — graphe de dépendances complet |
| État figé | `dvc.lock` (empreintes de tous les artefacts) |
| Environnement | `requirements.txt`, versions bornées |
| Vérification | CI rejoue `dvc repro --force` à chaque push |

## Considérations d'usage

- **Le modèle peut se tromper confortablement.** 47 % des heures seulement sont
  prédites à ±0,5 point. Présenter la sortie sans indication d'incertitude
  induirait les opérateurs en erreur.
- **Une fausse alerte n'est pas gratuite.** Au réglage retenu, environ deux
  alertes sur trois sont injustifiées. Si les opérateurs apprennent à les
  ignorer, le système perd toute valeur — le réglage doit être revu avec eux,
  pas imposé.
- **Aucune donnée personnelle** n'est traitée.
