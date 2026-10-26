# Cas métier

## Le procédé

La flottation par mousse sépare le minerai de fer de sa gangue siliceuse. La
pulpe de minerai broyé entre dans des colonnes où l'on injecte de l'air et deux
réactifs :

- **l'amine**, qui rend les particules de silice hydrophobes — elles accrochent
  les bulles d'air et remontent dans la mousse, qui est évacuée ;
- **l'amidon**, qui déprime le fer — il l'empêche de partir avec la mousse.

Le concentré de fer sort par le bas. Ce qu'il reste de silice dedans, c'est
l'**impureté** : la variable que ce projet cherche à anticiper.

## Pourquoi c'est un problème

Le contrat avec l'aciérie fixe une teneur maximale en silice. Un lot au-dessus
est déclassé, pénalisé, ou retraité — dans tous les cas il coûte.

Or la teneur réelle n'est connue que par **analyse laboratoire, une fois par
heure**. Quand le résultat arrive, l'heure de production correspondante est
terminée : le minerai est sorti. L'opérateur a donc piloté soixante minutes
sans savoir ce qu'il produisait.

```
  t=0h          t=1h                      t=2h
   │             │                         │
   ├─ production ─┤                         │
   │             ├─ prélèvement + analyse ──┤
   │             │                          └─► résultat connu
   │                                             (trop tard)
   └─► 60 minutes de pilotage à l'aveugle
```

Face à cette incertitude, le comportement rationnel par défaut est de supposer
que l'heure en cours ressemble à la moyenne. **C'est exactement la référence
contre laquelle ce projet se mesure** : la colonne « référence moyenne » des
résultats n'est pas une convention statistique, c'est ce que fait réellement
l'exploitation en l'absence de modèle.

## Ce que le modèle apporte

Huit capteurs sont déjà installés sur la ligne et échantillonnés en continu :

| Capteur | Ce qu'il mesure | Levier d'action ? |
|---|---|---|
| `ave_flot_air_flow` | débit d'air de flottation | **oui** — réglable en direct |
| `ave_flot_level` | niveau de mousse dans la colonne | **oui** — réglable en direct |
| `amina_flow` | débit d'amine | **oui** — dosage |
| `starch_flow` | débit d'amidon | **oui** — dosage |
| `ore_pulp_pH` | pH de la pulpe | **oui** — correction chimique |
| `ore_pulp_density` | densité de la pulpe | partiellement |
| `ore_pulp_flow` | débit de pulpe | partiellement |
| `iron_feed` | teneur en fer à l'entrée | **non** — subi, dépend du gisement |

Cinq de ces huit variables sont des **leviers que l'opérateur peut actionner
dans la minute**. C'est ce qui rend l'exercice utile : une prédiction n'a de
valeur que si quelqu'un peut agir dessus avant que l'heure soit perdue.

Aucun investissement matériel n'est nécessaire — le projet n'exploite que des
données déjà produites et déjà stockées.

## La décision, pas la prédiction

Un modèle qui annonce « 2,4 % de silice » ne sert à rien en salle de contrôle.
La question réelle est binaire :

> **Je corrige le dosage maintenant, ou je laisse tourner ?**

Toute la conception du projet découle de cette traduction. Les métriques
statistiques (RMSE, R²) servent à arbitrer entre deux versions du modèle. Mais
ce qui détermine la valeur livrée, c'est le **seuil de déclenchement de
l'alerte** — un paramètre qui n'est pas statistique mais économique.

### L'erreur à ne pas commettre

Le réflexe naturel est d'alerter dès que la prédiction dépasse la limite
contractuelle. C'est un mauvais réglage, et pour une raison structurelle : un
modèle de régression minimise l'erreur quadratique moyenne, ce qui le pousse à
**tirer ses prédictions vers la moyenne**. Il sous-estime systématiquement les
pics — précisément les heures qui nous intéressent.

Mesuré sur le jeu de test : en alertant au seuil contractuel de 3 %, on ne
détecte que **21 des 76 heures réellement hors spécification**. Soit 72 % des
lots problématiques qui passent sans aucun signal.

### Le seuil est un arbitrage économique

En abaissant le déclenchement à 2,15 %, on détecte **54 heures sur 76 (71 %)**,
au prix de **105 fausses alertes** sur 288 heures conformes.

Le bon réglage dépend de deux coûts que seule l'exploitation connaît :

| | Ce que ça coûte |
|---|---|
| **Alerte manquée** | un lot potentiellement déclassé : pénalité contractuelle, retraitement, voire perte du lot |
| **Fausse alerte** | une correction de dosage inutile : surconsommation d'amine, quelques minutes d'opérateur, risque de sur-correction |

Tant que le coût d'un lot déclassé dépasse nettement celui d'une correction
inutile — ce qui est le cas usuel dans la minéralurgie — **le réglage bas est
le bon**.

Ce choix n'est pas figé dans le code : il se pilote par
`evaluation.alert_recall_target` dans [`params.yaml`](../params.yaml). Changer
l'objectif de détection et relancer `dvc repro` recalcule le seuil, les
métriques et les figures. L'exploitation peut donc déplacer le curseur sans
intervention d'un data scientist.

## Ce que ce projet ne prétend pas faire

- **Il ne remplace pas le laboratoire.** L'analyse reste la mesure de référence
  et la base contractuelle. Le modèle comble l'attente, il ne la supprime pas.
- **Il ne pilote rien automatiquement.** La sortie est une alerte destinée à un
  opérateur, pas une consigne envoyée à un automate. Avec un R² de 0,26, une
  boucle fermée serait irresponsable.
- **Il ne dit pas quoi corriger.** L'importance des variables indique où le
  modèle puise son signal, ce qui n'est pas une relation causale : augmenter le
  débit d'air parce que le modèle y est sensible serait un contresens
  méthodologique.

Voir [`03-model-card.md`](03-model-card.md) pour le périmètre d'usage complet
et [`04-resultats.md`](04-resultats.md) pour les chiffres détaillés.
