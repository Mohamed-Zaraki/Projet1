"""weather_view.py : vue Streamlit en lecture seule du rapport météo stocké dans weather.db.

Ne dépend que de weather_db.WeatherDB (aucun calcul statistique ici).
Utilisation autonome :  streamlit run weather_view.py
Utilisation dans app.py :  from weather_view import render_weather ; render_weather()
"""
from __future__ import annotations

import html
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from weather_db import WeatherDB

DEFAULT_DB = Path(os.environ.get("WEATHER_DB", Path(__file__).parent / "weather.db"))
ACCENT = "#0E6B6F"

VERDICT = {
    "negligible": "🟢 négligeable",
    "low": "🟡 faible",
    "moderate": "🟠 modéré",
    "high": "🔴 élevé",
}
SECTION_TITLES = {
    "description": "Description",
    "missing_duplicates": "Manquants et doublons",
    "outliers": "Valeurs extrêmes",
    "noise": "Bruit",
    "monthly_trends": "Tendances mensuelles",
    "correlation": "Corrélations",
    "kde": "Distributions",
}


# ───────────────────────────── Accès aux données ─────────────────────────────

@st.cache_resource(show_spinner=False)
def _db(path: str) -> WeatherDB:
    return WeatherDB(path, read_only=True)


@st.cache_data(show_spinner=False)
def _reports(path: str) -> list[dict]:
    return _db(path).list_reports()


@st.cache_data(show_spinner=False)
def _report(path: str, rid: str) -> dict:
    return _db(path).get_report(rid, include_images=False)


@st.cache_data(show_spinner=False)
def _png(path: str, rid: str, name: str) -> bytes:
    return _db(path).get_plot_png(rid, name)


def load_light_report(path: str | Path, rid: str | None = None) -> dict:
    """Rapport sans images (≈150 Ko), pratique pour l'export JSON."""
    db = WeatherDB(path, read_only=True)
    return db.get_report(rid or db.latest_report_id(), include_images=False)


# ───────────────────────────── Helpers d'affichage ─────────────────────────────

def _stretch(fn, *args, **kwargs):
    """Largeur pleine, compatible avec les anciennes et nouvelles versions de Streamlit."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def _table(rows, **kwargs) -> None:
    if rows is None or len(rows) == 0:
        st.caption("Aucune donnée.")
        return
    _stretch(st.dataframe, pd.DataFrame(rows), hide_index=True, **kwargs)


def _insights(sec: dict) -> None:
    items = sec.get("insights") or []
    if items:
        st.markdown("\n".join(f"- {t.replace('$', chr(92) + '$')}" for t in items))


def _plots(rep: dict, section: str, path: str, only: list[str] | None = None) -> None:
    plots = [p for p in rep.get("plots", []) if p.get("section") == section]
    if only is not None:
        plots = [p for p in plots if p["name"] in only]
    for p in plots:
        try:
            st.image(_png(path, rep["meta"]["report_id"], p["name"]), caption=p.get("title") or p["name"])
        except KeyError:
            st.caption(f"Image manquante : {p['name']}")


def _head(sec: dict) -> None:
    if sec.get("description"):
        st.caption(sec["description"])
    _insights(sec)


# ───────────────────────────── Sections ─────────────────────────────

def _sec_description(rep, path):
    sec = rep["description"]
    _head(sec)
    m = rep["meta"]
    c = st.columns(4)
    c[0].metric("Lignes", f"{m['n_rows']:,}".replace(",", " "))
    c[1].metric("Colonnes", m["n_cols"])
    c[2].metric("Colonne de date", m.get("date_column") or "—")
    c[3].metric("Colonnes constantes", len(sec.get("constant_columns", [])))
    st.subheader("Colonnes")
    _table(sec["info"]["columns"])
    st.subheader("Statistiques numériques")
    _table(sec["describe_numeric"])
    st.subheader("Statistiques catégorielles")
    _table(sec["describe_categorical"])
    st.subheader("Premières lignes")
    _table(sec["head"])
    _plots(rep, "description", path)


def _sec_missing(rep, path):
    sec = rep["missing_duplicates"]
    _head(sec)
    dup = sec["duplicates"]
    c = st.columns(3)
    c[0].metric("Cellules manquantes", sec["missing"]["total_cells"])
    c[1].metric("Lignes en double", dup["count"])
    c[2].metric("Part de doublons (%)", dup["percent"])
    only_missing = st.checkbox("Colonnes avec valeurs manquantes seulement", True, key="w_missing_only")
    cols = sec["missing"]["columns"]
    _table([c for c in cols if c["count"] > 0] if only_missing else cols)
    _plots(rep, "missing_duplicates", path)


def _sec_outliers(rep, path):
    sec = rep["outliers"]
    _head(sec)
    st.caption("Les valeurs signalées sont à examiner, pas à supprimer automatiquement.")
    df = pd.DataFrame(sec["features"]).sort_values("iqr_percent", ascending=False)
    _stretch(st.dataframe, df, hide_index=True)
    _plots(rep, "outliers", path)


def _sec_noise(rep, path):
    sec = rep["noise"]
    _head(sec)
    s = sec["dataset_summary"]
    c = st.columns(4)
    c[0].metric("Variables analysées", s["features_checked"])
    c[1].metric("Avec bruit", s["features_with_noise"])
    c[2].metric("Avec erreurs", s["features_with_errors"])
    c[3].metric("Bruit/signal moyen", f"{s['mean_noise_to_signal']:.3f}".replace(".", ","))
    df = pd.DataFrame(sec["features"])
    df["verdict"] = df["verdict"].map(lambda v: VERDICT.get(v, v))
    cols = ["feature", "verdict", "noise_to_signal", "snr_db", "spikes", "spike_percent",
            "out_of_range", "has_noise", "has_errors"]
    _stretch(st.dataframe, df[[c for c in cols if c in df.columns]].sort_values("noise_to_signal", ascending=False),
             hide_index=True)
    problems = [f for f in sec["features"] if f.get("noise_reasons") or f.get("error_reasons")]
    for f in problems:
        with st.expander(f"{f['feature']} · {VERDICT.get(f['verdict'], f['verdict'])}"):
            for r in f.get("noise_reasons", []):
                st.markdown(f"- Bruit : {r}")
            for r in f.get("error_reasons", []):
                st.markdown(f"- Erreur : {r}")
    for sk in sec.get("skipped", []):
        st.caption(f"Ignorée : {sk['feature']} ({sk['reason']}).")
    with st.expander("Méthode et seuils"):
        for k, v in sec["method"].items():
            st.markdown(f"**{k}** : {v}")
        st.json(sec["thresholds"])
    _plots(rep, "noise", path)


def _sec_monthly(rep, path):
    sec = rep["monthly_trends"]
    _head(sec)
    feats = list(sec["monthly_stats"])
    feat = st.selectbox("Variable", feats, key="w_month_feat")
    df = pd.DataFrame(sec["monthly_stats"][feat])
    df["month"] = pd.to_datetime(df["month"])
    band = alt.Chart(df).mark_area(opacity=0.2, color=ACCENT).encode(x="month:T", y="low:Q", y2="high:Q")
    line = alt.Chart(df).mark_line(color=ACCENT).encode(
        x=alt.X("month:T", title=None), y=alt.Y("mean:Q", title=feat, scale=alt.Scale(zero=False)),
        tooltip=["month:T", "mean:Q", "low:Q", "high:Q", "std:Q"])
    st.altair_chart((band + line).properties(width="container", height=280))
    st.subheader("Moyenne par mois de l'année")
    seas = pd.DataFrame(sec["seasonality"][feat].items(), columns=["mois", "moyenne"])
    bars = alt.Chart(seas).mark_bar(color=ACCENT).encode(
        x=alt.X("mois:N", sort=list(seas["mois"]), title=None), y=alt.Y("moyenne:Q", title=feat),
        tooltip=["mois:N", alt.Tooltip("moyenne:Q", format=".2f")])
    st.altair_chart(bars.properties(width="container", height=240))
    _plots(rep, "monthly_trends", path)


def _sec_correlation(rep, path):
    sec = rep["correlation"]
    _head(sec)
    kind = st.radio("Coefficient", ["pearson", "spearman"], horizontal=True, key="w_corr_kind")
    mat = pd.DataFrame(sec[kind])
    long = mat.reset_index(names="a").melt("a", var_name="b", value_name="r")
    heat = alt.Chart(long).mark_rect().encode(
        x=alt.X("b:N", title=None, sort=list(mat.columns)), y=alt.Y("a:N", title=None, sort=list(mat.index)),
        color=alt.Color("r:Q", scale=alt.Scale(domain=[-1, 0, 1], range=["#B04A08", "#F4F4F4", ACCENT]), title="r"),
        tooltip=["a:N", "b:N", alt.Tooltip("r:Q", format=".3f")])
    st.altair_chart(heat.properties(width="container", height=340))
    st.subheader("Paires de variables")
    pairs = pd.DataFrame(sec["pairs"])
    _stretch(st.dataframe, pairs.reindex(pairs["pearson"].abs().sort_values(ascending=False).index), hide_index=True)
    bc = sec.get("by_category")
    if bc:
        st.subheader(f"Moyennes par « {bc['category']} »")
        if bc.get("note"):
            st.caption(bc["note"])
        _table(bc["means"])
    _plots(rep, "correlation", path)


def _sec_kde(rep, path):
    sec = rep["kde"]
    _head(sec)
    _table(sec["features"])
    for sk in sec.get("skipped", []):
        st.caption(f"Ignorée : {sk['feature']} ({sk['reason']}).")
    _plots(rep, "kde", path)


RENDERERS = {
    "description": _sec_description,
    "missing_duplicates": _sec_missing,
    "outliers": _sec_outliers,
    "noise": _sec_noise,
    "monthly_trends": _sec_monthly,
    "correlation": _sec_correlation,
    "kde": _sec_kde,
}


# ───────────────────────────── Point d'entrée ─────────────────────────────

def render_weather(db_path: str | Path = DEFAULT_DB) -> dict | None:
    """Affiche le rapport et le renvoie (sans images), ou None s'il n'y a rien à afficher."""
    path = str(db_path)
    if not Path(path).exists():
        st.error(f"Base introuvable : {path}")
        st.caption("Placez weather.db à côté de l'application ou définissez WEATHER_DB=/chemin/weather.db.")
        return None
    reports = _reports(path)
    if not reports:
        st.info("weather.db ne contient aucun rapport.")
        return None

    ids = [r["report_id"] for r in reports]
    labels = {r["report_id"]: f"{r['dataset']} · {r['generated_at'][:16].replace('T', ' ')}" for r in reports}
    rid = st.sidebar.selectbox("Rapport météo", ids, format_func=labels.get, key="w_report") if len(ids) > 1 else ids[0]
    rep = _report(path, rid)
    m = rep["meta"]

    when = m["generated_at"][:16].replace("T", " ")
    rows = f"{m['n_rows']:,}".replace(",", " ")
    st.markdown(f"<div class='eyebrow'>Rapport de dataset</div><h1 class='title'>{html.escape(m['dataset'])}</h1>"
                f"<div class='meta'>Généré le {when} · {rows} lignes · {m['n_cols']} colonnes</div>",
                unsafe_allow_html=True)

    names = [n for n in rep if n not in ("meta", "plots")]
    tabs = st.tabs([SECTION_TITLES.get(n, n) for n in names])
    for tab, name in zip(tabs, names):
        with tab:
            fn = RENDERERS.get(name)
            if fn is None:                       # section inconnue : affichage générique
                _head(rep[name])
                st.json(rep[name], expanded=1)
                _plots(rep, name, path)
            else:
                fn(rep, path)
    return rep


if __name__ == "__main__":
    st.set_page_config(page_title="Rapport météo", layout="wide")
    st.markdown("<style>.eyebrow{font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;font-weight:600;"
                "color:#0E6B6F}.title{font-size:2.6rem;margin:.3rem 0}.meta{font-size:.85rem;opacity:.7}</style>",
                unsafe_allow_html=True)
    render_weather()
