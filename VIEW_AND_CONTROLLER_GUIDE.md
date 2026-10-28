# Guide Développeur : Intégration View & Controller (Phase Modèle)

Ce document est destiné à ton collaborateur en charge des couches **View** et **Controller** de l'architecture MVC du projet Météo.

---

## 1. Ce qui a été préparé et stocké dans le Model

La phase de modélisation a été exécutée et enregistrée dans [`weather.db`](file:///c:/Users/Moham/Desktop/files/weather.db).

### Clé de section : `"model_phase"`
Les données structurées et les graphiques sont accessibles via la section `"model_phase"`.

#### A. Équilibre des classes (`target_balance`)
- **Variable cible** : `Precip Type`
- **Résultat** : Données fortement déséquilibrées (**Severe Imbalance**)
  - `rain` : 85 224 (88,36 %)
  - `snow` : 10 712 (11,11 %)
  - `Missing / Not recorded` : 517 (0,54 %)
  - **Ratio de déséquilibre** : ~7,96 pour 1
- **Graphique généré** : `precip_type_balance` (barres annotées + donut de proportion)

#### B. Prétraitement (`preprocessing`)
- **Variables numériques** : `['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']`
  - Pipeline : `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`
- **Variables catégorielles** : `['Summary']`
  - Pipeline : `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`
- **Méthode** : `ColumnTransformer` (34 features après One-Hot Encoding)

#### C. Importance des caractéristiques (`feature_importance`)
- **Modèle** : `RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)`
- **Classement des variables (Top)** :
  1. `Temperature (C)` : **52,65 %**
  2. `Apparent Temperature (C)` : **33,64 %** *(Les 2 températures cumulent ~86,3 %)*
  3. `Visibility (km)` : **4,09 %**
  4. `Pressure (millibars)` : **3,97 %**
  5. `Summary` : **2,37 %**
  6. `Humidity` : **1,70 %**
  7. `Wind Speed (km/h)` : **0,86 %**
  8. `Wind Bearing (degrees)` : **0,71 %**
- **Graphique généré** : `rf_feature_importance` (diagramme horizontal trié par importance relative)

---

## 2. Ce que tu peux faire dans la Vue (`View`)

### Option 1 : Utilisation autonome (immédiate)
La vue générique actuelle dans [`weather_view.py`](file:///c:/Users/Moham/Desktop/files/weather_view.py) gère déjà dynamiquement toute section inconnue :
- Si tu lances `streamlit run weather_view.py` ou `streamlit run app.py`, un onglet **`model_phase`** apparaît automatiquement avec les puces d'insights, les données JSON et les deux graphiques PNG.

### Option 2 : Ajouter un rendu dédié et soigné dans `weather_view.py`
Pour donner à cet onglet le même design que les onglets existants (*Description*, *Bruit*, *Corrélations*), tu peux ajouter une fonction dédiée dans [`weather_view.py`](file:///c:/Users/Moham/Desktop/files/weather_view.py) :

#### Étape A : Définir le titre du nouvel onglet
Ajouter l'entrée dans le dictionnaire `SECTION_TITLES` :
```python
SECTION_TITLES = {
    "description": "Description",
    "missing_duplicates": "Manquants et doublons",
    "outliers": "Valeurs extrêmes",
    "noise": "Bruit",
    "monthly_trends": "Tendances mensuelles",
    "correlation": "Corrélations",
    "kde": "Distributions",
    "model_phase": "Préparation & Modèle ML",   # <-- Titre lisible
}
```

#### Étape B : Écrire la fonction de rendu `_sec_model_phase`
Exemple prêt à l'emploi :
```python
def _sec_model_phase(rep: dict, path: str):
    sec = rep["model_phase"]
    _head(sec)  # Affiche la description et les puces d'insights
    
    # 1. Cartes de métriques clés
    tb = sec["target_balance"]
    c = st.columns(4)
    c[0].metric("Variable cible", tb["target_feature"])
    c[1].metric("Statut", "Déséquilibré 🔴" if not tb["is_balanced"] else "Équilibré 🟢")
    c[2].metric("Ratio Rain / Snow", f"{tb['imbalance_ratio']} : 1")
    c[3].metric("Échantillons valides", f"{tb['valid_observations']:,}".replace(",", " "))
    
    # 2. Graphique d'équilibre des classes
    st.subheader("1. Équilibre des classes (Precip Type)")
    _plots(rep, "model_phase", path, only=["precip_type_balance"])
    
    # 3. Informations sur le Prétraitement (Pipeline & ColumnTransformer)
    st.subheader("2. Prétraitement des données")
    prep = sec["preprocessing"]
    col_num, col_cat = st.columns(2)
    with col_num:
        st.markdown("**Variables numériques (Pipeline) :**")
        st.caption("Imputation médiane + StandardScaler")
        st.write(prep["numerical_features"])
    with col_cat:
        st.markdown("**Variables catégorielles (Pipeline) :**")
        st.caption("Imputation mode + OneHotEncoder")
        st.write(prep["categorical_features"])
    
    # 4. Importance des variables (Random Forest)
    st.subheader("3. Importance des variables (Random Forest)")
    fi = sec["feature_importance"]
    st.caption(f"Modèle : {fi['model']} (n_estimators={fi['n_estimators']}, class_weight={fi['class_weight']})")
    
    # Tableau du classement des variables
    df_rank = pd.DataFrame(fi["ranked_features"])
    _stretch(st.dataframe, df_rank, hide_index=True)
    
    # Graphique d'importance
    _plots(rep, "model_phase", path, only=["rf_feature_importance"])
```

#### Étape C : Enregistrer la fonction dans `RENDERERS`
```python
RENDERERS = {
    "description": _sec_description,
    "missing_duplicates": _sec_missing,
    "outliers": _sec_outliers,
    "noise": _sec_noise,
    "monthly_trends": _sec_monthly,
    "correlation": _sec_correlation,
    "kde": _sec_kde,
    "model_phase": _sec_model_phase,   # <-- Brancher ici
}
```

---

## 3. Ce que tu peux faire dans le Contrôleur (`Controller`)

Le rôle du Contrôleur est de faire le lien entre le stockage SQLite / JSON et l'affichage utilisateur.

### A. Récupérer les données brutes ou les images depuis `WeatherDB`
```python
from weather_db import WeatherDB

db = WeatherDB("weather.db", read_only=True)
rid = db.latest_report_id()

# Obtenir uniquement la section model_phase
model_data = db.get_section(rid, "model_phase")

# Obtenir les octets PNG bruts d'un plot pour l'afficher ou le télécharger
png_balance = db.get_plot_png(rid, "precip_type_balance")
png_importance = db.get_plot_png(rid, "rf_feature_importance")

# Exécuter une requête SQL directe
results = db.query(
    "SELECT name, title FROM plots WHERE report_id = ? AND section = 'model_phase'",
    (rid,)
)
```

### B. Proposer un bouton d'export ou d'import JSON dans l'UI
Dans [`app.py`](file:///c:/Users/Moham\Desktop\files\app.py), tu peux charger directement [`latest_weather_report.json`](file:///c:/Users/Moham/Desktop/files/latest_weather_report.json) dans l'interface « Importer des rapports » pour le visualiser dans le tableau de bord générique.

En code Python :
```python
import json
from pathlib import Path

# Lecture du rapport JSON exporté
report_data = json.loads(Path("latest_weather_report.json").read_text(encoding="utf-8"))
```

### C. Permettre le ré-entraînement interactif depuis l'interface (Optionnel)
Si tu souhaites ajouter dans la barre latérale un bouton « Ré-entraîner le modèle avec d'autres paramètres » :
```python
import streamlit as st
from weather_model import WeatherEDAModel
from weather_db import WeatherDB

if st.sidebar.button("Relancer l'entraînement RF"):
    with st.spinner("Entraînement en cours..."):
        model = WeatherEDAModel("weatherHistory.csv")
        new_report = model.run()
        db = WeatherDB("weather.db")
        new_rid = db.save_report(new_report)
        st.sidebar.success(f"Nouveau rapport enregistré : {new_rid}")
        st.rerun()
```

---

## 4. Fichiers disponibles dans le projet
- [`weather_model.py`](file:///c:/Users/Moham/Desktop/files/weather_model.py) : Code source du Model (avec déséquilibre, Pipeline, ColumnTransformer, RF).
- [`weather.db`](file:///c:/Users/Moham/Desktop/files/weather.db) : Base SQLite contenant les tables `reports`, `sections`, `insights`, `plots` (images PNG BLOB).
- [`latest_weather_report.json`](file:///c:/Users/Moham/Desktop/files/latest_weather_report.json) : Export JSON complet prêt à l'import dans `app.py`.
- [`weather_view.py`](file:///c:/Users/Moham/Desktop/files/weather_view.py) : Composant de visualisation (intact, prêt pour tes ajouts).
- [`app.py`](file:///c:/Users/Moham/Desktop/files/app.py) : Tableau de bord principal Streamlit.
