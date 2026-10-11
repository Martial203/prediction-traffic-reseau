# Migrer vers les données réelles

## Format attendu

Un CSV, une ligne par cellule et par heure :

| Colonne | Contenu |
|---|---|
| `timestamp` | Début de l'heure, en UTC (`2024-01-25 14:00:00`) |
| `cell_id` | Identifiant de cellule, format libre |
| `traffic_volume` | Trafic en Mbps (débit moyen sur l'heure), la valeur à prédire |
| `active_users`, `rsrp` (dBm), `sinr` (dB) | Moyennes horaires |
| `event` | `normal`, `congestion` ou `handover` |

Le chargement refuse les doublons, les valeurs manquantes et les heures manquantes. Ce
dernier point est important : les lags sont calculés par décalage de lignes, un trou les
fausserait sans provoquer d'erreur.

## Étapes

1. Dans `backend/app/prepare.py`, adapter `COLUMN_MAP` (noms des colonnes de l'export),
   `TRAFFIC_TO_MBPS` (conversion d'unité, par exemple Mo/h × 8/3600) et `SOURCE_TZ`
   (fuseau horaire de l'export).
2. Préparer le fichier : `python -m app.prepare export.csv C:\...\reel.csv`.
   Le script agrège à l'heure, comble les trous de 3 h au plus, écarte les cellules trop
   incomplètes et affiche un rapport par cellule.
3. Définir, avec des chemins absolus, pour l'entraînement comme pour l'API :
   `set TRAFFIC_DATA_PATH=C:\...\reel.csv`
   et éventuellement `set TRAFFIC_ARTIFACTS_DIR=C:\...\artifacts_reel` pour ne pas écraser
   les modèles actuels.
4. Lancer `python -m app.train` et les tests, puis redémarrer l'API.

Le frontend s'adapte de lui-même à la liste des cellules et à la période.

## Vérifications

- Les modèles doivent battre nettement le naïf saisonnier. Sinon, revoir les variables.
- Un trafic moyen incohérent trahit en général une erreur de conversion d'unité.
- Des pics décalés dans la journée trahissent un problème de fuseau horaire.
- Un R² supérieur à 0,99 doit faire suspecter une fuite d'information : toute nouvelle
  variable doit être décalée d'au moins une heure par rapport à l'heure prédite.

## Ajustements selon les données

- Avec plus de 6 semaines d'historique, ajouter le lag 168 (même heure la semaine
  précédente) à `TRAFFIC_LAGS` dans `data.py` et passer `LOOKBACK` à 168 dans `config.py`.
- Nouveaux types d'événements : `EVENTS` dans `data.py` et `overview.page.ts`, `cells()`
  dans `api.py`.
- Autre unité que le Mbps : libellés des pages `overview` et `cell`.
- Au-delà d'une dizaine de cellules, remplacer les onglets de cellules par une liste
  déroulante.
- Toute nouvelle cellule impose un réentraînement.
- Ne pas versionner les données réelles si elles sont sensibles.
