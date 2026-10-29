# Architecture et choix d'ingénierie

## Le graphe de dépendances

Six étapes déclarées dans [`dvc.yaml`](../dvc.yaml). Chacune annonce ses
entrées, ses sorties et les sections de `params.yaml` dont elle dépend. DVC en
déduit le graphe et **ne réexécute que ce qui est réellement affecté**.

```
                          params.yaml
                               │
   data/raw/raw.csv ──────► [split] ──► X_train  X_test  y_train  y_test
                               │
                               ▼
                         [normalize] ──► *_scaled.csv  +  scaler.joblib
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
         [tune]           [benchmark]        (jeu de test
            │                  │              intouché)
            │                  └──► model_comparison.json
            │                       comparaison_modeles.png
            ▼
     best_params.json
     best_params.pkl
     tuning.json
            │
            ▼
        [train] ──────────────────► model.joblib
            │
            ▼
      [evaluate] ──► metrics.json
                     business_metrics.json
                     alert_threshold_sweep.json
                     predictions.csv
                     5 figures
```

### Ce que cette granularité achète

| Changement | Étapes rejouées |
|---|---|
| `evaluation.spec_limit_pp` | `evaluate` seule — quelques secondes |
| `tuning.param_grid` | `tune`, `train`, `evaluate` |
| `data.test_size` | tout le pipeline |
| Une docstring dans `src/viz.py` | `evaluate` et `benchmark` (figures) |

Découper finement n'est pas de la coquetterie : ajuster le seuil d'alerte est
l'opération qu'on refait le plus souvent, et elle ne doit pas coûter un
réentraînement.

## La refonte du code initial

La première version fonctionnait, mais comme une suite de scripts, pas comme un
livrable. Le tableau ci-dessous liste ce qui a été repris et pourquoi.

| Avant | Problème | Après |
|---|---|---|
| `load("/home/ubuntu/projet/models/trained_model.joblib")` | Le code ne tourne que sur la VM de l'auteur | Chemins résolus depuis la racine du projet ([`src/config.py`](../src/config.py)), surchargeables par `SILICA_PROJECT_ROOT` |
| `MODELS_DIR = "/models"` | Pointe vers la racine du disque ; `os.makedirs` échoue ou crée un dossier parasite | Chemins typés via la dataclass `Paths` |
| Hyperparamètres recopiés à la main dans `train_model.py` | `GridSearch.py` écrivait un `.pkl` que personne ne lisait ; toute nouvelle recherche exigeait une édition manuelle | `tune` écrit `best_params.json`, `train` le relit — avec un test qui vérifie la correspondance |
| `GridSearch.py` hors pipeline | La recherche n'était pas reproductible ni versionnée | Étape DVC à part entière, avec son dépôt de métriques |
| Scaler jeté après usage | Modèle inutilisable en production : rien ne permet de transformer une observation entrante | `models/scaler.joblib` produit et testé (rejouer la transformation doit redonner les données d'entraînement) |
| `from sklearn.metrics import accuracy_score` | Métrique de classification importée dans un problème de régression | Supprimé |
| Un seul modèle, sans point de comparaison | Impossible de savoir si 0,23 de R² est bon ou mauvais | Étape `benchmark` sur 9 familles, dont une référence « moyenne » qui matérialise le niveau zéro |
| Constantes dispersées dans 5 fichiers | Changer la taille du test = éditer plusieurs scripts | [`params.yaml`](../params.yaml), source de vérité unique, suivie par DVC |
| Aucun test | Une fuite de données passerait inaperçue | 33 tests, dont 3 dédiés aux fuites |
| Aucune métrique métier | Le R² ne dit rien à un responsable de production | `business_metrics.json` et la calibration du seuil d'alerte |

La configuration initiale n'a pas été effacée : elle reste dans le registre sous
`random_forest_baseline`, avec ses hyperparamètres figés par un test. Elle sert
de point de comparaison honnête dans le benchmark.

## Décisions d'ingénierie

### `python -m src.<module>` plutôt que `python src/<...>.py`

Invoquer un module permet aux imports `from src.config import ...` de
fonctionner sans manipuler `sys.path` ni installer le paquet. Les commandes de
`dvc.yaml` sont donc identiques en local, en CI et dans un conteneur.

### Un registre de modèles

[`src/registry.py`](../src/registry.py) associe un nom à un estimateur. Changer
de modèle devient un changement de configuration — versionné par DVC, visible
dans un diff, et non une modification de code. C'est aussi ce qui permet au
benchmark d'itérer sur une liste de noms venue de `params.yaml`.

### Racine de projet surchargeable

`SILICA_PROJECT_ROOT` déplace la racine. C'est ce qui rend le pipeline testable
sur un jeu de données jouet, dans un `tmp_path`, sans écraser les artefacts
réels — et c'est également utile pour monter le projet ailleurs dans un
conteneur.

### Deux fichiers de métriques

`metrics.json` sert à arbitrer entre deux versions du modèle ;
`business_metrics.json` sert à décider si le modèle vaut la peine d'être
déployé. Les mélanger reviendrait à laisser croire qu'un gain de 0,02 de R² est
en soi une bonne nouvelle.

### Métriques et figures versionnées dans Git

Déclarées `cache: false` dans `dvc.yaml` : elles partent dans Git, pas dans le
stockage distant. Conséquence directe — `dvc metrics diff` montre l'évolution
entre deux commits, et une pull request affiche l'impact d'un changement sur
les performances. Les artefacts lourds (jeux normalisés, modèle), eux, restent
dans le cache DVC.

### Un repli sur `params.yaml` à l'entraînement

Si `best_params.json` est absent, `train` utilise `train.fallback_params` au
lieu d'échouer. Cela permet d'exécuter le pipeline en CI ou en test sans payer
le coût d'une recherche sur grille complète, tout en journalisant clairement la
provenance des hyperparamètres utilisés.

### `n_jobs` explicite

`n_jobs: -1` consomme tous les cœurs et, sur une machine de bureau, fait
tomber les workers joblib par saturation mémoire quand on entraîne des forêts
de 600 arbres. La valeur est désormais un réglage de premier niveau dans
`params.yaml`, fixé à 4 : plus sûr, et surtout plus reproductible d'une machine
à l'autre.

### Journaux en UTF-8

`setup_logging()` reconfigure explicitement la sortie standard. Sans cela, la
console Windows (page de code 1252) transforme les accents des messages en
caractères parasites, y compris dans les journaux de CI.

## Organisation du dépôt

```
├── params.yaml              source de vérité unique
├── dvc.yaml                 définition du pipeline
├── dvc.lock                 état figé : versions exactes des artefacts
├── data/raw/raw.csv         données brutes (334 Ko, dans Git)
├── src/
│   ├── config.py            chemins, chargement des params, journaux
│   ├── registry.py          registre des modèles
│   ├── viz.py               feuille de style unique des figures
│   ├── data/                split, normalisation
│   └── models/              tune, train, evaluate, benchmark
├── tests/                   33 tests (unitaires + intégration)
├── metrics/                 métriques versionnées
├── reports/figures/         figures versionnées
└── .github/workflows/ci.yml style, tests, et reproduction du pipeline
```

## Intégration continue

Deux travaux dans [`ci.yml`](../.github/workflows/ci.yml) :

1. **Style et tests** — `ruff` et `pytest` sur Python 3.10 et 3.12.
2. **Reproductibilité** — `dvc repro --force` depuis un dépôt propre. C'est le
   cœur de la promesse du projet : si une étape dépend d'un fichier local non
   déclaré, c'est là que ça casse. Le travail compare ensuite les métriques
   recalculées à celles versionnées et signale toute dérive.

## Limite connue sous Windows

Le cache de `dvc repro` peut dépasser la limite de 260 caractères par chemin
quand le dépôt est cloné dans une arborescence profonde : `dvc_objects` crée un
fichier temporaire à côté de sa destination, ce qui ajoute une trentaine de
caractères à des chemins de hachage déjà longs. Trois remèdes :

- cloner dans un dossier court (`C:\dev\silica`) ;
- activer les chemins longs : `LongPathsEnabled = 1` dans
  `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem` ;
- déplacer le cache hors du dépôt :
  `dvc cache dir --local C:\dvc-cache` (écrit dans `.dvc/config.local`, ignoré
  par Git).

Linux et macOS ne sont pas concernés ; la CI non plus.
