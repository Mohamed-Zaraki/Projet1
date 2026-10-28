"""Rapports Datasets : tableau de bord Streamlit pour des rapports de datasets au format JSON.

Les rapports importés sont enregistrés dans une base SQLite locale (data/rapports.db).
À chaque ouverture, la liste complète est relue et le dernier rapport consulté s'affiche.

Lancer :  streamlit run app.py
"""
from __future__ import annotations

import html
import json
import math
import os
import re
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DB_PATH = Path(os.environ.get("RAPPORTS_DB", Path(__file__).parent / "data" / "rapports.db"))
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ACCENT = "#0E6B6F"
WARN = "#B04A08"
MUTED = "#7A8A90"
ATTN = re.compile(r"missing|manquant|null|nan\b|duplic|doublon|error|erreur|invalid|outlier|aberr", re.I)

st.set_page_config(page_title="Rapports Datasets", layout="wide")

RID: int | None = None  # rapport affiché, sert à fabriquer des clés de widgets uniques


# ───────────────────────────── Base de données ─────────────────────────────

def conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute(
        "CREATE TABLE IF NOT EXISTS reports ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, "
        "added_at TEXT NOT NULL, size INTEGER NOT NULL, content TEXT NOT NULL)"
    )
    c.execute("CREATE TABLE IF NOT EXISTS settings (k TEXT PRIMARY KEY, v TEXT)")
    return c


def list_reports() -> list[sqlite3.Row]:
    with closing(conn()) as c:
        return c.execute("SELECT id, name, added_at, size FROM reports ORDER BY id DESC").fetchall()


def add_report(name: str, text: str) -> int:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON invalide (ligne {e.lineno}, colonne {e.colno}).") from e
    content = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    with closing(conn()) as c:
        cur = c.execute(
            "INSERT INTO reports (name, added_at, size, content) VALUES (?, ?, ?, ?)",
            (name, datetime.now().isoformat(timespec="minutes"), len(text.encode("utf-8")), content),
        )
        c.commit()
        return int(cur.lastrowid)


def delete_report(rid: int) -> None:
    with closing(conn()) as c:
        c.execute("DELETE FROM reports WHERE id = ?", (rid,))
        c.commit()


def get_setting(key: str) -> str | None:
    with closing(conn()) as c:
        row = c.execute("SELECT v FROM settings WHERE k = ?", (key,)).fetchone()
        return row["v"] if row else None


def set_setting(key: str, value: str) -> None:
    with closing(conn()) as c:
        c.execute("INSERT INTO settings (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v", (key, value))
        c.commit()


@st.cache_data(show_spinner=False)
def load_report(rid: int, db_path: str):
    with closing(conn()) as c:
        row = c.execute("SELECT content FROM reports WHERE id = ?", (rid,)).fetchone()
    return json.loads(row["content"]) if row else None


# ───────────────────────────── Formatage ─────────────────────────────

def esc(s) -> str:
    return html.escape(str(s))


def md_escape(s: str) -> str:
    return re.sub(r"([\\`*_\[\]~<>$|])", r"\\\1", s)


def is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def fmt_num(n) -> str:
    if isinstance(n, int) or (isinstance(n, float) and n.is_integer() and abs(n) < 1e15):
        s = f"{int(n):,}"
    else:
        s = f"{n:,.3f}".rstrip("0").rstrip(".")
    return s.replace(",", " ").replace(".", ",")


def fmt_val(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, bool):
        return "oui" if v else "non"
    if is_num(v):
        return fmt_num(v)
    if isinstance(v, (dict, list)):
        s = json.dumps(v, ensure_ascii=False)
        return s if len(s) <= 120 else s[:117] + "…"
    return str(v)


def fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} o"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} Ko".replace(".", ",")
    return f"{n / 1024 / 1024:.1f} Mo".replace(".", ",")


def fmt_date(s: str) -> str:
    try:
        return datetime.fromisoformat(s).strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return s


def stretch(fn, *args, **kwargs):
    """Largeur pleine, compatible avec les anciennes et nouvelles versions de Streamlit."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:  # anciennes versions : width n'accepte pas "stretch" (TypeError ou erreur Streamlit)
        return fn(*args, use_container_width=True, **kwargs)


# ───────────────────────────── Graphiques ─────────────────────────────

def bar_chart(items: list[tuple[str, float]], warn: bool = False) -> None:
    df = pd.DataFrame(items, columns=["label", "value"])
    df["label"] = df["label"].astype(str)
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=5, size=14, color=WARN if warn else ACCENT)
        .encode(
            y=alt.Y(
                "label:N",
                sort=alt.EncodingSortField(field="value", order="descending"),
                title=None,
                axis=alt.Axis(labelLimit=240, domain=False, ticks=False),
            ),
            x=alt.X("value:Q", title=None, axis=alt.Axis(domain=False)),
            tooltip=[alt.Tooltip("label:N", title="clé"), alt.Tooltip("value:Q", title="valeur", format=",.3~f")],
        )
        .properties(width="container", height=max(60, 28 * len(df)))
    )
    st.altair_chart(chart)


def cell_str(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return fmt_val(v)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c) for c in df.columns]
    for c in df.columns:
        vals = [v for v in df[c] if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if vals and all(is_num(v) for v in vals):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = df[c].map(cell_str)
    return df


# ───────────────────────────── Sections de l'aperçu ─────────────────────────────

def S(title: str, sub: str | None, wide: bool, body) -> dict:
    return {"title": title, "sub": sub, "wide": wide, "body": body}


def note_sec(title: str, msg: str) -> dict:
    return S(title, None, True, lambda: st.caption(msg))


def kpi_sec(title: str, entries: list[tuple[str, object]]) -> dict:
    small = [(k, v) for k, v in entries if v is None or isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= 28)]
    longs = [(k, v) for k, v in entries if (k, v) not in small]

    def body() -> None:
        per = 4
        for i in range(0, len(small), per):
            cols = st.columns(per)
            for col, (k, v) in zip(cols, small[i : i + per]):
                with col:
                    st.metric(k, fmt_val(v))
                    if ATTN.search(k) and is_num(v) and v > 0:
                        st.markdown('<span class="pill">à vérifier</span>', unsafe_allow_html=True)
        for k, v in longs:
            st.markdown(f"**{md_escape(str(k))}** · {md_escape(str(v))}")

    return S(title, f"{len(entries)} valeur{'s' if len(entries) > 1 else ''}", True, body)


def bars_sec(title: str, entries: list[tuple[str, float]], sub: str | None = None) -> dict:
    ordered = sorted(entries, key=lambda e: -abs(e[1]))

    def body() -> None:
        bar_chart(ordered[:30], warn=bool(ATTN.search(title)))
        if len(ordered) > 30:
            st.caption(f"+ {len(ordered) - 30} autres non affichées")

    return S(title, sub or f"{len(entries)} entrées", False, body)


def series_sec(title: str, arr: list[float]) -> dict:
    n = len(arr)
    mn, mx, mean = min(arr), max(arr), sum(arr) / n

    def body() -> None:
        if n <= 40:
            df = pd.DataFrame({"i": list(range(1, n + 1)), "value": arr})
            chart = (
                alt.Chart(df)
                .mark_bar(color=ACCENT, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("i:O", title=None, axis=alt.Axis(labels=n <= 20, labelAngle=0)),
                    y=alt.Y("value:Q", title=None),
                    tooltip=[alt.Tooltip("i:O", title="rang"), alt.Tooltip("value:Q", title="valeur", format=",.3~f")],
                )
                .properties(width="container", height=190)
            )
        else:
            counts, edges = np.histogram(arr, bins=24)
            df = pd.DataFrame({"start": edges[:-1], "end": edges[1:], "count": counts})
            bars = (
                alt.Chart(df)
                .mark_bar(color=ACCENT, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("start:Q", title=None),
                    x2="end:Q",
                    y=alt.Y("count:Q", title=None),
                    tooltip=[
                        alt.Tooltip("start:Q", title="de", format=",.3~f"),
                        alt.Tooltip("end:Q", title="à", format=",.3~f"),
                        alt.Tooltip("count:Q", title="effectif"),
                    ],
                )
            )
            rule = (
                alt.Chart(pd.DataFrame({"m": [mean]}))
                .mark_rule(strokeDash=[5, 4], color=MUTED, size=1.5)
                .encode(x="m:Q")
            )
            chart = (bars + rule).properties(width="container", height=190)
        st.altair_chart(chart)
        st.caption(f"n {fmt_num(n)} · min {fmt_num(mn)} · max {fmt_num(mx)} · moyenne {fmt_num(mean)}")

    return S(title, "Série" if n <= 40 else "Histogramme, 24 classes", False, body)


def text_list_sec(title: str, items: list[str]) -> dict:
    """Phrases (insights, noms de fichiers…) : une puce par élément, sans graphique de fréquences."""
    def body() -> None:
        for t in items[:50]:
            st.markdown(f"- {md_escape(t)}")
        if len(items) > 50:
            st.caption(f"+ {len(items) - 50} autres, voir l’onglet Structure.")

    return S(title, f"{len(items)} élément{'s' if len(items) > 1 else ''}", True, body)


def table_sec(title: str, rows: list[dict], sub: str) -> dict:
    def body() -> None:
        df = clean_df(pd.DataFrame(rows))
        nums = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) and df[c].notna().any()
        ]
        if nums and len(df) >= 2:
            col = st.selectbox("Graphique de la colonne", nums, key=f"col|{RID}|{title}")
            labcol = next((c for c in df.columns if c not in nums), None)
            labels = df[labcol].astype(str).tolist() if labcol else [f"#{i + 1}" for i in range(len(df))]
            items = [(lab, float(v)) for lab, v in zip(labels, df[col]) if pd.notna(v)]
            items.sort(key=lambda x: -x[1])
            bar_chart(items[:15], warn=bool(ATTN.search(title)))
            if len(items) > 15:
                st.caption(f"15 premières valeurs sur {len(items)}")
        stretch(st.dataframe, df.head(1000), hide_index=True)
        if len(df) > 1000:
            st.caption(f"1 000 premières lignes sur {len(df)}.")

    return S(title, sub, True, body)


def rows_from_object(o: dict) -> list[dict]:
    return [{"clé": k, **v} for k, v in o.items()]


def visit(path: list[str], v, depth: int, out: list[dict]) -> None:
    if len(out) >= 60:
        if len(out) == 60:
            out.append(note_sec("Aperçu limité", "Les autres blocs sont visibles dans l’onglet Structure."))
        return
    title = " › ".join(path)
    if isinstance(v, list):
        if not v:
            out.append(note_sec(title, "Liste vide."))
        elif all(is_num(x) for x in v):
            out.append(series_sec(title, v))
        elif all(isinstance(x, dict) for x in v):
            out.append(table_sec(title, v, f"{len(v)} ligne{'s' if len(v) > 1 else ''}"))
        elif all(isinstance(x, str) for x in v):
            cnt = Counter(v)
            if len(cnt) == len(v) or (len(v) >= 3 and len(cnt) > 0.5 * len(v)):   # presque tout est unique : des phrases
                out.append(text_list_sec(title, v))
                return
            out.append(bars_sec(title, list(cnt.items()), f"fréquences de {len(v)} valeurs"))
        else:
            out.append(note_sec(title, "Contenu mixte, voir l’onglet Structure."))
        return
    if isinstance(v, dict):
        ents = list(v.items())
        if not ents:
            out.append(note_sec(title, "Objet vide."))
        elif len(ents) >= 2 and all(is_num(x) for _, x in ents):
            out.append(bars_sec(title, ents))
        elif all(isinstance(x, dict) for _, x in ents):
            out.append(table_sec(title, rows_from_object(v), f"{len(ents)} entrée{'s' if len(ents) > 1 else ''}"))
        elif depth >= 3:
            out.append(note_sec(title, "Niveau trop profond, voir l’onglet Structure."))
        else:
            scalars = [(k, x) for k, x in ents if not isinstance(x, (dict, list))]
            if scalars:
                out.append(kpi_sec(title, scalars))
            for k, x in ents:
                if isinstance(x, (dict, list)):
                    visit(path + [k], x, depth + 1, out)
        return
    out.append(kpi_sec(title, [(path[-1], v)]))


def overview(data) -> list[dict]:
    out: list[dict] = []
    if isinstance(data, dict):
        ents = list(data.items())
        scalars = [(k, v) for k, v in ents if not isinstance(v, (dict, list))]
        if scalars:
            out.append(kpi_sec("Résumé", scalars))
        for k, v in ents:
            if isinstance(v, (dict, list)):
                visit([k], v, 1, out)
    else:
        visit(["Données"], data, 0, out)
    return out or [note_sec("Aperçu", "Rien à afficher pour ce rapport.")]


def draw(sec: dict) -> None:
    with st.container(border=True):
        sub = f'<span class="s">{esc(sec["sub"])}</span>' if sec["sub"] else ""
        st.markdown(f'<div class="card-h"><span class="t">{esc(sec["title"])}</span>{sub}</div>', unsafe_allow_html=True)
        sec["body"]()


def render_sections(secs: list[dict]) -> None:
    buf: list[dict] = []

    def flush() -> None:
        if len(buf) == 2:
            for col, s in zip(st.columns(2, gap="medium"), buf):
                with col:
                    draw(s)
        else:
            for s in buf:
                draw(s)
        buf.clear()

    for s in secs:
        if s["wide"]:
            flush()
            draw(s)
        else:
            buf.append(s)
            if len(buf) == 2:
                flush()
    flush()


# ───────────────────────────── Interface ─────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Hanken+Grotesk:wght@400;500;600&display=swap');
html, body, .stApp, [data-testid="stSidebar"] { font-family: 'Hanken Grotesk', system-ui, sans-serif; }
.block-container { padding-top: 2.4rem; max-width: 1240px; }
.brand { font-family: 'Instrument Serif', Georgia, serif; font-size: 2rem; line-height: 1; letter-spacing: -.01em; margin: 0 0 .2rem; }
.brand-sub { font-size: .8rem; opacity: .65; margin-bottom: .8rem; }
.eyebrow { font-size: .7rem; letter-spacing: .16em; text-transform: uppercase; font-weight: 600; color: #0E6B6F; }
.title { font-family: 'Instrument Serif', Georgia, serif; font-size: 3.2rem; line-height: 1.04; letter-spacing: -.015em; margin: .3rem 0 .5rem; overflow-wrap: anywhere; }
.meta { font-size: .85rem; opacity: .7; margin-bottom: 1.2rem; }
.card-h { display: flex; justify-content: space-between; align-items: baseline; gap: .8rem; flex-wrap: wrap; margin-bottom: .4rem; }
.card-h .t { text-transform: uppercase; letter-spacing: .09em; font-size: .74rem; font-weight: 600; overflow-wrap: anywhere; }
.card-h .s { font-size: .76rem; opacity: .6; }
[data-testid="stMetricValue"] { font-family: 'Instrument Serif', Georgia, serif; font-size: 2.5rem; font-weight: 400; letter-spacing: -.01em; }
[data-testid="stMetricLabel"] p { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .74rem; opacity: .7; }
.pill { font-size: .7rem; font-weight: 600; background: #FBEBD5; color: #B04A08; border-radius: 999px; padding: 2px 9px; }
.empty-title { font-family: 'Instrument Serif', Georgia, serif; font-size: 2.6rem; line-height: 1.05; letter-spacing: -.015em; margin: 2rem 0 .8rem; }
</style>
"""


def flash(kind: str, msg: str) -> None:
    st.session_state.setdefault("flash", []).append((kind, msg))


def sidebar() -> int | None:
    sb = st.sidebar
    sb.markdown('<div class="brand">Rapports</div><div class="brand-sub">de datasets, en JSON</div>', unsafe_allow_html=True)

    for kind, msg in st.session_state.pop("flash", []):
        (sb.error if kind == "error" else sb.success)(msg)

    # Import de fichiers
    st.session_state.setdefault("up_key", 0)
    files = sb.file_uploader(
        "Importer des rapports", type=["json"], accept_multiple_files=True,
        key=f"up_{st.session_state['up_key']}", help="Un ou plusieurs fichiers .json",
    )
    if files:
        last = None
        for f in files:
            try:
                raw = f.getvalue()
                if len(raw) > MAX_UPLOAD_BYTES:
                    raise ValueError("fichier de plus de 50 Mo.")
                last = add_report(Path(f.name).stem, raw.decode("utf-8-sig"))
                flash("ok", f"« {f.name} » enregistré.")
            except UnicodeDecodeError:
                flash("error", f"« {f.name} » : encodage non reconnu (UTF-8 attendu).")
            except ValueError as e:
                flash("error", f"« {f.name} » : {e}")
        if last is not None:
            st.session_state["selected_id"] = last
        st.session_state["up_key"] += 1
        st.rerun()

    # Collage de JSON
    with sb.expander("Coller du JSON"):
        with st.form("paste", clear_on_submit=True):
            name = st.text_input("Nom du rapport")
            text = st.text_area("Contenu JSON", height=160)
            if st.form_submit_button("Enregistrer"):
                if not text.strip():
                    flash("error", "Collez d’abord du JSON.")
                else:
                    try:
                        label = name.strip() or f"Rapport collé {datetime.now():%H:%M}"
                        st.session_state["selected_id"] = add_report(label, text)
                        flash("ok", f"« {label} » enregistré.")
                    except ValueError as e:
                        flash("error", str(e))
                st.rerun()

    # Liste des rapports enregistrés
    reps = list_reports()
    sb.divider()
    if not reps:
        sb.caption("Aucun rapport enregistré.")
        return None

    ids = [int(r["id"]) for r in reps]
    labels = {int(r["id"]): f"{md_escape(r['name'])} · {fmt_date(r['added_at'])}" for r in reps}
    if st.session_state.get("selected_id") not in ids:
        saved = get_setting("selected")
        st.session_state["selected_id"] = int(saved) if saved and int(saved) in ids else ids[0]
    sb.radio("Rapports enregistrés", ids, format_func=lambda i: labels[i], key="selected_id")
    rid = int(st.session_state["selected_id"])
    if get_setting("selected") != str(rid):
        set_setting("selected", str(rid))

    # Suppression en deux temps
    if st.session_state.get("confirm_delete") == rid:
        sb.warning("Supprimer définitivement ce rapport ?")
        c1, c2 = sb.columns(2)
        if c1.button("Confirmer", key="del_yes", type="primary"):
            delete_report(rid)
            st.session_state.pop("confirm_delete", None)
            st.session_state.pop("selected_id", None)
            flash("ok", "Rapport supprimé.")
            st.rerun()
        if c2.button("Annuler", key="del_no"):
            st.session_state.pop("confirm_delete", None)
            st.rerun()
    elif sb.button("Supprimer ce rapport", key="del_ask"):
        st.session_state["confirm_delete"] = rid
        st.rerun()

    sb.caption(f"{len(reps)} rapport{'s' if len(reps) > 1 else ''} enregistré{'s' if len(reps) > 1 else ''} dans {DB_PATH.name}.")
    return rid


def weather_mode() -> None:
    """Affiche weather.db (Model météo) via weather_view, et permet de copier le rapport dans les rapports JSON."""
    try:
        from weather_view import DEFAULT_DB, render_weather
    except ImportError as e:
        st.error(f"Vue météo indisponible : {e}. Placez weather_view.py et weather_db.py à côté de app.py.")
        return
    render_weather(DEFAULT_DB)


def main() -> None:
    global RID
    st.markdown(CSS, unsafe_allow_html=True)
    # Par défaut : le rapport lu directement dans la base weather.db (aucun JSON à importer).
    source = st.sidebar.radio("Source", ["Météo (weather.db)", "Rapports JSON"], key="source")
    if source == "Météo (weather.db)":
        weather_mode()
        return
    rid = sidebar()
    RID = rid

    if rid is None:
        st.markdown('<div class="empty-title">Aucun rapport enregistré</div>', unsafe_allow_html=True)
        st.write(
            "Importez un ou plusieurs fichiers JSON de rapport de dataset avec le panneau de gauche. "
            "Chaque rapport est enregistré : il réapparaît à chaque ouverture du tableau de bord."
        )
        st.write("Les listes, tableaux, statistiques et distributions sont repérés automatiquement et tracés.")
        return

    row = next(r for r in list_reports() if int(r["id"]) == rid)
    data = load_report(rid, str(DB_PATH))
    if isinstance(data, list):
        desc = f"liste · {len(data)} éléments"
    elif isinstance(data, dict):
        desc = f"objet · {len(data)} clés"
    else:
        desc = "valeur simple"

    st.markdown(
        f'<div class="eyebrow">Rapport de dataset</div><h1 class="title">{esc(row["name"])}</h1>'
        f'<div class="meta">Ajouté le {esc(fmt_date(row["added_at"]))} · {esc(fmt_size(row["size"]))} · {esc(desc)}</div>',
        unsafe_allow_html=True,
    )

    tab_ov, tab_tree, tab_raw = st.tabs(["Aperçu", "Structure", "JSON brut"])
    with tab_ov:
        render_sections(overview(data))
    with tab_tree:
        if row["size"] > 2_000_000:
            st.info("Rapport volumineux : la structure n’est pas affichée. Utilisez l’onglet JSON brut.")
        else:
            st.json(data, expanded=1)
    with tab_raw:
        text = json.dumps(data, ensure_ascii=False, indent=2)
        st.download_button("Télécharger le JSON", text, file_name=f"{row['name']}.json", mime="application/json")
        if len(text) > 250_000:
            st.caption("Affichage limité aux 250 000 premiers caractères.")
        st.code(text[:250_000], language="json")


main()