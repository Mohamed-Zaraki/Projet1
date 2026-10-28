"""
weather_db.py  -  MODEL layer (MVC): SQLite storage for the weather EDA report.

Uses only the Python standard library (sqlite3). No UI code.

  Controller :  db = WeatherDB("weather.db")
                rid = db.save_report(model.run())            # store a whole report
  View       :  db.list_reports()                            # [{report_id, dataset, generated_at, ...}]
                db.get_report(rid)                           # same dict as report.json (images as base64)
                db.get_report(rid, include_images=False)     # light version, no base64
                db.get_plot_png(rid, "correlation_heatmap")  # raw PNG bytes  -> serve as image/png
                db.query("SELECT ...")                       # any read-only SQL

Tables
  reports            one row per analysis run (meta as JSON)
  sections           one row per section (description, full section data as JSON)
  insights           one row per insight sentence               (section, position, text)
  plots              one row per plot, PNG stored as BLOB
  missing_columns    relational copy: missing values per column
  outlier_stats      relational copy: IQR / Z-score outliers per feature
  noise_features     relational copy: noise verdict + metrics per feature
  correlation_pairs  relational copy: Pearson / Spearman per feature pair

The JSON in `sections` keeps everything (nothing is lost); the relational tables make the
important results easy to query and filter with plain SQL.
"""

import json
import base64
import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS reports (
    report_id    TEXT PRIMARY KEY,
    dataset      TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    n_rows       INTEGER,
    n_cols       INTEGER,
    date_column  TEXT,
    meta_json    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id   TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    position    INTEGER NOT NULL,
    name        TEXT NOT NULL,
    description TEXT,
    data_json   TEXT NOT NULL,
    UNIQUE (report_id, name)
);

CREATE TABLE IF NOT EXISTS insights (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    section   TEXT NOT NULL,
    position  INTEGER NOT NULL,
    text      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plots (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    position  INTEGER NOT NULL,
    name      TEXT NOT NULL,
    title     TEXT,
    section   TEXT,
    mime      TEXT DEFAULT 'image/png',
    file_path TEXT,
    image     BLOB,
    UNIQUE (report_id, name)
);

CREATE TABLE IF NOT EXISTS missing_columns (
    report_id TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    feature   TEXT NOT NULL,
    count     INTEGER,
    percent   REAL,
    PRIMARY KEY (report_id, feature)
);

CREATE TABLE IF NOT EXISTS outlier_stats (
    report_id     TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    feature       TEXT NOT NULL,
    min_value     REAL, max_value REAL, q1 REAL, q3 REAL, iqr REAL,
    lower_fence   REAL, upper_fence REAL,
    iqr_outliers  INTEGER, iqr_percent REAL,
    z_outliers    INTEGER, z_percent REAL,
    PRIMARY KEY (report_id, feature)
);

CREATE TABLE IF NOT EXISTS noise_features (
    report_id       TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    feature         TEXT NOT NULL,
    noise_to_signal REAL, snr_db REAL, verdict TEXT,
    von_neumann     REAL, lag1_autocorr REAL,
    spikes          INTEGER, spike_percent REAL,
    out_of_range    INTEGER, longest_flat_run INTEGER, stuck_sensor INTEGER,
    has_noise       INTEGER, has_errors INTEGER,
    PRIMARY KEY (report_id, feature)
);

CREATE TABLE IF NOT EXISTS correlation_pairs (
    report_id TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    feature_a TEXT NOT NULL,
    feature_b TEXT NOT NULL,
    pearson   REAL, spearman REAL, strength TEXT, direction TEXT,
    PRIMARY KEY (report_id, feature_a, feature_b)
);

CREATE INDEX IF NOT EXISTS idx_insights_report ON insights(report_id, section, position);
CREATE INDEX IF NOT EXISTS idx_plots_report    ON plots(report_id, section);
"""

SECTION_KEYS_NOT_IN_JSON = {"description", "insights"}   # stored in their own column / table


class WeatherDB:
    def __init__(self, db_path="weather.db", read_only=False):
        """read_only=True : the database is opened with SQLite's read-only mode (the file is never
        touched, not even to create tables). Use it for every View/Controller."""
        self.db_path = str(db_path)
        self.read_only = read_only
        if read_only:
            if not Path(self.db_path).exists():
                raise FileNotFoundError(f"Database not found: {self.db_path}")
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as con:
                con.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        """Yield a connection, commit/rollback like `with con`, and ALWAYS close it
        (sqlite3's own `with` does not close, which keeps the file locked on Windows)."""
        if self.read_only:
            con = sqlite3.connect(Path(self.db_path).resolve().as_uri() + "?mode=ro", uri=True)
        else:
            con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        try:
            with con:
                yield con
        finally:
            con.close()

    # ------------------------------------------------------------------ write
    def save_report(self, report, replace=True):
        """Store one report (the dict returned by WeatherEDAModel.run()). Returns its report_id.
        Everything is written in ONE transaction: either the whole report is saved or nothing."""
        meta = report["meta"]
        rid = meta["report_id"]
        with self._connect() as con:                       # commits on success, rolls back on error
            if replace:
                con.execute("DELETE FROM reports WHERE report_id = ?", (rid,))   # cascades to children
            con.execute(
                "INSERT INTO reports VALUES (?,?,?,?,?,?,?)",
                (rid, meta["dataset"], meta["generated_at"], meta.get("n_rows"), meta.get("n_cols"),
                 meta.get("date_column"), json.dumps(meta, ensure_ascii=False)))

            sections = [k for k, v in report.items() if k not in ("meta", "plots") and isinstance(v, dict)]
            for pos, name in enumerate(sections):
                sec = report[name]
                data = {k: v for k, v in sec.items() if k not in SECTION_KEYS_NOT_IN_JSON}
                con.execute("INSERT INTO sections (report_id, position, name, description, data_json) "
                            "VALUES (?,?,?,?,?)",
                            (rid, pos, name, sec.get("description"), json.dumps(data, ensure_ascii=False)))
                con.executemany("INSERT INTO insights (report_id, section, position, text) VALUES (?,?,?,?)",
                                [(rid, name, i, t) for i, t in enumerate(sec.get("insights", []))])

            for pos, p in enumerate(report.get("plots", [])):
                con.execute(
                    "INSERT INTO plots (report_id, position, name, title, section, mime, file_path, image) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (rid, pos, p["name"], p.get("title"), p.get("section"), p.get("mime", "image/png"),
                     p.get("file"), self._image_bytes(p)))

            self._save_relational(con, rid, report)
        return rid

    @staticmethod
    def _image_bytes(p):
        if p.get("base64"):
            return base64.b64decode(p["base64"])
        f = p.get("file")
        return Path(f).read_bytes() if f and Path(f).exists() else None

    @staticmethod
    def _save_relational(con, rid, report):
        for c in report.get("missing_duplicates", {}).get("missing", {}).get("columns", []):
            con.execute("INSERT INTO missing_columns VALUES (?,?,?,?)", (rid, c["name"], c["count"], c["percent"]))
        for f in report.get("outliers", {}).get("features", []):
            con.execute("INSERT INTO outlier_stats VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (rid, f["feature"], f.get("min"), f.get("max"), f.get("q1"), f.get("q3"), f.get("iqr"),
                         f.get("lower_fence"), f.get("upper_fence"), f.get("iqr_outliers"),
                         f.get("iqr_percent"), f.get("z_outliers"), f.get("z_percent")))
        for f in report.get("noise", {}).get("features", []):
            con.execute("INSERT INTO noise_features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (rid, f["feature"], f["noise_to_signal"], f["snr_db"], f["verdict"], f["von_neumann"],
                         f["lag1_autocorr"], f["spikes"], f["spike_percent"], f["out_of_range"],
                         f["longest_flat_run"], int(f["stuck_sensor"]), int(f["has_noise"]), int(f["has_errors"])))
        for p in report.get("correlation", {}).get("pairs", []):
            con.execute("INSERT INTO correlation_pairs VALUES (?,?,?,?,?,?,?)",
                        (rid, p["a"], p["b"], p["pearson"], p["spearman"], p["strength"], p["direction"]))

    # ------------------------------------------------------------------ read
    def list_reports(self):
        with self._connect() as con:
            rows = con.execute("SELECT report_id, dataset, generated_at, n_rows, n_cols, date_column "
                               "FROM reports ORDER BY generated_at DESC").fetchall()
        return [dict(r) for r in rows]

    def latest_report_id(self):
        r = self.list_reports()
        return r[0]["report_id"] if r else None

    def get_report(self, report_id, include_images=True):
        """Rebuild the full report dict (identical in content to report.json)."""
        with self._connect() as con:
            head = con.execute("SELECT meta_json FROM reports WHERE report_id = ?", (report_id,)).fetchone()
            if head is None:
                raise KeyError(f"unknown report_id: {report_id}")
            report = {"meta": json.loads(head["meta_json"])}
            ins = {}
            for r in con.execute("SELECT section, text FROM insights WHERE report_id = ? "
                                 "ORDER BY section, position", (report_id,)):
                ins.setdefault(r["section"], []).append(r["text"])
            for r in con.execute("SELECT name, description, data_json FROM sections WHERE report_id = ? "
                                 "ORDER BY position", (report_id,)):
                sec = {"description": r["description"], **json.loads(r["data_json"])}
                sec["insights"] = ins.get(r["name"], [])
                report[r["name"]] = sec
            plots = []
            for r in con.execute("SELECT name, title, section, mime, file_path, image FROM plots "
                                 "WHERE report_id = ? ORDER BY position", (report_id,)):
                p = {"name": r["name"], "title": r["title"], "section": r["section"],
                     "file": r["file_path"], "mime": r["mime"]}
                if include_images and r["image"] is not None:
                    p["base64"] = base64.b64encode(r["image"]).decode()
                plots.append(p)
            report["plots"] = plots
        return report

    def get_section(self, report_id, name):
        """One section only (the View can load tabs lazily)."""
        with self._connect() as con:
            r = con.execute("SELECT description, data_json FROM sections WHERE report_id = ? AND name = ?",
                            (report_id, name)).fetchone()
            if r is None:
                raise KeyError(f"unknown section: {name}")
            sec = {"description": r["description"], **json.loads(r["data_json"])}
            sec["insights"] = [x["text"] for x in con.execute(
                "SELECT text FROM insights WHERE report_id = ? AND section = ? ORDER BY position",
                (report_id, name))]
        return sec

    def get_plot_png(self, report_id, name):
        """Raw PNG bytes of one plot (serve with Content-Type image/png)."""
        with self._connect() as con:
            r = con.execute("SELECT image FROM plots WHERE report_id = ? AND name = ?",
                            (report_id, name)).fetchone()
        if r is None or r["image"] is None:
            raise KeyError(f"unknown plot: {name}")
        return r["image"]

    def get_plot_base64(self, report_id, name):
        return base64.b64encode(self.get_plot_png(report_id, name)).decode()

    def query(self, sql, params=()):
        """Run a read-only SQL query; returns a list of dicts."""
        if not sql.lstrip().lower().startswith(("select", "with")):
            raise ValueError("only SELECT queries are allowed here")
        with self._connect() as con:
            con.execute("PRAGMA query_only = ON")      # blocks writes even in "WITH ... DELETE" tricks
            return [dict(r) for r in con.execute(sql, params)]

    # ------------------------------------------------------------------ delete
    def delete_report(self, report_id):
        with self._connect() as con:
            con.execute("DELETE FROM reports WHERE report_id = ?", (report_id,))   # cascades


if __name__ == "__main__":
    import sys
    from weather_model import WeatherEDAModel
    if len(sys.argv) < 2:
        sys.exit("Usage: python weather_db.py weatherHistory.csv [weather.db]")
    model = WeatherEDAModel(sys.argv[1])
    report = model.run()
    db = WeatherDB(sys.argv[2] if len(sys.argv) > 2 else "weather.db")
    rid = db.save_report(report)
    print("Saved report", rid, "->", db.db_path)

    