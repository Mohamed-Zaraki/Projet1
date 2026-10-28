# Rapports Datasets

Tableau de bord Streamlit pour lire, visualiser et conserver des rapports de datasets au format JSON.

## Lancer

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Utilisation

- **Importer** : déposer un ou plusieurs fichiers `.json` dans le panneau de gauche, ou utiliser « Coller du JSON ».
- **Enregistrement** : chaque rapport est stocké dans `data/rapports.db` (SQLite). À chaque ouverture, tous les rapports enregistrés sont listés et le dernier consulté s'affiche.
- **Visualisation** : l'onglet Aperçu repère automatiquement les valeurs simples (chiffres clés), les listes de nombres (histogramme), les objets de nombres (barres), les listes d'objets et les objets d'objets (tableaux avec graphique par colonne). Les onglets Structure et JSON brut montrent le contenu exact.
- **Suppression** : bouton « Supprimer ce rapport » sous la liste, avec confirmation.

## Source météo (weather.db)

Le sélecteur **Source** de la barre latérale bascule entre les rapports JSON et le rapport météo du Model (`weather.db`), affiché en lecture seule par `weather_view.py` via `weather_db.WeatherDB` : un onglet par section (description, manquants, extrêmes, bruit, tendances, corrélations, distributions), avec insights, tableaux et images.

- Fichiers à placer à côté de `app.py` : `weather_view.py`, `weather_db.py`, `weather.db` (autre chemin : `WEATHER_DB=/chemin/weather.db`).
- **Ajouter aux rapports JSON** copie le rapport (sans images) dans `rapports.db`. En ligne de commande : `python export_weather_json.py weather.db sortie.json`, puis import du fichier.
- Vue seule : `streamlit run weather_view.py`.

Pour utiliser une autre base de données : `RAPPORTS_DB=/chemin/vers/fichier.db streamlit run app.py`.
