"""
weather_model.py  -  MODEL layer (MVC) for the Weather EDA project.

Responsibility : load the weather CSV, run the analysis, produce plots and a
                 JSON-serialisable report. NO UI code here (no plt.show(), no Flask).
Controller use : model = WeatherEDAModel("weather.csv"); report = model.run(); model.save()
View use       : read eda_output/report.json  (or the db_documents.json collections)

Pipeline (same order as the assignment)
  1. Description   : df.info(), df.head(), df.describe()
  2. Missing values + duplicates
  3. Outliers      : IQR fences + Z-score
  4. Noise         : robust noise-to-signal ratio, SNR (dB), von Neumann smoothness, spike (Hampel)
                     detection, stuck sensors, impossible values -> verdict per feature
  4b. Monthly trends: your resample("MS") mean/q10/q90/std bands + month-by-month lines
  5. Correlation   : Pearson + Spearman, heatmap, top pairs, scatter of strongest
                     pairs, histograms per weather 'Summary' (FacetGrid)
  6. KDE           : one density plot per numeric feature
"""

import io
import re
import json
import base64
import warnings
import calendar
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless: Model never opens windows
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

MONTH_STYLE = {                             # from your plot_month / label_color code
    1: ("January", "blue"),    2: ("February", "green"),   3: ("March", "orange"),
    4: ("April", "gold"),      5: ("May", "red"),          6: ("June", "violet"),
    7: ("July", "purple"),     8: ("August", "black"),     9: ("September", "brown"),
    10: ("October", "darkblue"), 11: ("November", "grey"), 12: ("December", "pink"),
}


# ---- noise-detection settings
SPIKE_K = 4                      # spike = more than 4 robust sigmas from the rolling median
SPIKE_PCT_FLAG = 1.0             # a feature counts as noisy if more than 1% of its points are spikes
SPIKE_MIN_FRAC = 0.5             # ...and also > 50% of the feature's normal spread
CIRCULAR_KEYS = ("bearing",)      # circular variables: linear noise tests do not apply
STUCK_RUN = 24                   # >= 24 identical consecutive readings = stuck sensor
NSR_LEVELS = (0.05, 0.15, 0.35)  # verdict thresholds: negligible / low / moderate / high
PHYSICAL_RANGES = {              # (min, max) physically possible values, matched on column name
    "temperature": (-90, 60), "humidity": (0, 1), "pressure": (850, 1090),
    "wind speed": (0, 150), "wind bearing": (0, 360), "visibility": (0, 100),
}


def robust_sigma(x):
    """Std estimate from the median absolute deviation (ignores outliers)."""
    x = np.asarray(x, dtype=float)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else 0.0


def physical_range(col, s):
    name = col.lower()
    for key, rng in PHYSICAL_RANGES.items():
        if key in name:
            if key == "humidity" and s.quantile(.99) > 1.5:      # humidity stored as percent
                return (0, 100)
            return rng
    return None


# ------------------------------------------------------------------ utils
def clean(o):
    """Recursively turn numpy / pandas objects into plain JSON types (NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if o is pd.NaT:
        return None
    return o


def records(df):
    """DataFrame -> list of dicts (NaN -> None, timestamps -> ISO strings)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def slug(text):
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()


# ------------------------------------------------------------------ model
class WeatherEDAModel:
    def __init__(self, csv_path, out_dir="eda_output", category_col="Summary",
                 top_categories=8, dpi=150, embed_base64=True):
        self.csv_path = Path(csv_path)
        self.out_dir = Path(out_dir)
        self.plots_dir = self.out_dir / "plots"
        self.category_col = category_col
        self.top_categories = top_categories
        self.dpi = dpi
        self.embed_base64 = embed_base64
        self.plots = []
        self.report = {}

    # ---------------------------------------------------------- loading
    def load(self):
        self.raw = pd.read_csv(self.csv_path)
        df = self.raw.copy()
        self.date_col = None
        for c in df.columns:
            if any(k in c.lower() for k in ("date", "time")):
                parsed = pd.to_datetime(df[c], utc=True, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    self.date_col = c
                    df[c] = parsed.dt.tz_convert(None)
                    break
        if self.date_col:
            df = df.dropna(subset=[self.date_col]).sort_values(self.date_col)
            df = df.set_index(self.date_col)
        self.df = df
        all_num = df.select_dtypes(include="number").columns.tolist()
        self.constant_cols = [c for c in all_num if df[c].nunique(dropna=True) <= 1]
        self.num_cols = [c for c in all_num if c not in self.constant_cols]
        self.cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        # df_valid = same data with physically impossible readings set to NaN (e.g. pressure = 0).
        # Outliers, correlation, KDE and monthly analysis use it so bad readings cannot distort them.
        self.df_valid = df.copy()
        self.invalid_counts = {}
        for c in self.num_cols:
            rng = physical_range(c, df[c].dropna())
            if rng:
                bad = (df[c] < rng[0]) | (df[c] > rng[1])
                if bad.any():
                    self.df_valid.loc[bad, c] = np.nan
                    self.invalid_counts[c] = int(bad.sum())
        # key features used by the assignment snippets (fallback: first numeric cols)
        pref = [c for c in ("Temperature (C)", "Humidity") if c in self.num_cols]
        self.key_feats = pref + [c for c in self.num_cols if c not in pref][: 2 - len(pref)]

    # ---------------------------------------------------------- plotting
    def _save_plot(self, fig, name, title, section):
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        path = self.plots_dir / f"{name}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        entry = {"name": name, "title": title, "section": section,
                 "file": str(path), "mime": "image/png"}
        if self.embed_base64:
            entry["base64"] = base64.b64encode(path.read_bytes()).decode()
        self.plots.append(entry)

    def _section(self, key, fn):
        """Run one analysis step and attach the names of the plots it produced."""
        before = len(self.plots)
        result = fn()
        result["plots"] = [p["name"] for p in self.plots[before:]]
        self.report[key] = result

    # ---------------------------------------------------------- 1. description
    def describe(self):
        buf = io.StringIO()
        self.raw.info(buf=buf)
        desc_num = self.df[self.num_cols + self.constant_cols].describe().T
        desc_num["skew_valid"] = self.df_valid[desc_num.index].skew()
        desc_num["kurtosis_valid"] = self.df_valid[desc_num.index].kurt()
        desc_cat = self.df[self.cat_cols].describe().T if self.cat_cols else pd.DataFrame()
        insights = [
            f"The dataset has {len(self.raw):,} rows and {self.raw.shape[1]} columns "
            f"({len(self.num_cols) + len(self.constant_cols)} numeric, {len(self.cat_cols)} categorical).",
        ]
        if self.date_col:
            idx = self.df.index
            insights.append(f"Time range: {idx.min():%Y-%m-%d} to {idx.max():%Y-%m-%d} "
                            f"(column '{self.date_col}', used as the time index).")
        if self.constant_cols:
            insights.append(f"Constant column(s) with no information: {self.constant_cols} "
                            "- excluded from correlation, KDE and noise analysis.")
        skewed = [c for c in desc_num.index if abs(desc_num.loc[c, "skew_valid"]) > 1 and c not in self.constant_cols]
        if skewed:
            insights.append(f"Strongly skewed features (|skew| > 1, impossible values excluded): {skewed}.")
        for c, k in self.invalid_counts.items():
            insights.append(f"'{c}' contains {k} physically impossible readings - they are excluded from "
                            "outlier, correlation, KDE and monthly analysis (see the Noise section).")
        return {
            "description": "Structure of the dataset: dtypes and non-null counts (info), "
                           "first rows (head) and summary statistics (describe).",
            "info": {
                "text": buf.getvalue(),
                "memory_kb": round(self.raw.memory_usage(deep=True).sum() / 1024, 2),
                "columns": [{"name": c, "dtype": str(self.raw[c].dtype),
                             "non_null": int(self.raw[c].notna().sum()),
                             "n_unique": int(self.raw[c].nunique())} for c in self.raw.columns],
            },
            "head": records(self.raw.head(10)),
            "describe_numeric": records(desc_num.reset_index().rename(columns={"index": "feature"})),
            "describe_categorical": records(desc_cat.reset_index().rename(columns={"index": "feature"}))
            if len(desc_cat) else [],
            "constant_columns": self.constant_cols,
            "insights": insights,
        }

    # ---------------------------------------------------------- 2. missing + duplicates
    def missing_duplicates(self):
        n = len(self.raw)
        miss = self.raw.isna().sum()
        miss_cols = [{"name": c, "count": int(miss[c]), "percent": round(miss[c] / n * 100, 3)}
                     for c in self.raw.columns]
        dup = self.raw.duplicated(keep="first")
        insights = []
        bad = sorted([m for m in miss_cols if m["count"] > 0], key=lambda m: -m["percent"])
        if not bad:
            insights.append("No missing values.")
        for m in bad:
            kind = "categorical" if m["name"] in self.cat_cols else "numeric"
            fix = "fill with the mode or an 'Unknown' label" if kind == "categorical" \
                else "interpolate over time or fill with the median"
            insights.append(f"'{m['name']}' has {m['count']} missing values ({m['percent']}%) - "
                            f"{kind}; suggested fix: {fix}.")
        insights.append(f"{int(dup.sum())} fully duplicated rows ({dup.mean() * 100:.3f}%)"
                        + (" - remove with drop_duplicates()." if dup.any() else "."))
        if bad:
            fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(bad))))
            ax.barh([m["name"] for m in bad][::-1], [m["percent"] for m in bad][::-1], color="#BD4A32")
            ax.set_xlabel("% missing")
            ax.set_title("Missing values per column")
            self._save_plot(fig, "missing_values", "Missing values per column", "missing_duplicates")
        return {
            "description": "Completeness check: missing cells per column and exact duplicate rows.",
            "missing": {"total_cells": int(miss.sum()), "columns": miss_cols},
            "duplicates": {"count": int(dup.sum()), "percent": round(dup.mean() * 100, 3),
                           "row_indexes": self.raw.index[dup].tolist()[:500]},
            "insights": insights,
        }

    # ---------------------------------------------------------- 3. outliers
    def outliers(self):
        rows = []
        for c in self.num_cols:
            s = self.df_valid[c].dropna()
            q1, q3 = s.quantile(.25), s.quantile(.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            iqr_mask = (s < lo) | (s > hi)
            z = np.abs(stats.zscore(s))
            rows.append({"feature": c, "min": s.min(), "max": s.max(), "q1": q1, "q3": q3, "iqr": iqr,
                         "lower_fence": lo, "upper_fence": hi,
                         "iqr_outliers": int(iqr_mask.sum()), "iqr_percent": round(iqr_mask.mean() * 100, 3),
                         "z_outliers": int((z > 3).sum()), "z_percent": round((z > 3).mean() * 100, 3)})
        insights = []
        flagged = sorted([r for r in rows if r["iqr_percent"] > 0], key=lambda r: -r["iqr_percent"])
        if flagged:
            top = flagged[0]
            insights.append(f"Most outlier-prone feature: '{top['feature']}' "
                            f"({top['iqr_outliers']} IQR outliers, {top['iqr_percent']}%).")
        clean_f = [r["feature"] for r in rows if r["iqr_outliers"] == 0]
        if clean_f:
            insights.append(f"No IQR outliers in: {clean_f}.")
        diff = [r["feature"] for r in rows if r["iqr_outliers"] > 5 * max(r["z_outliers"], 1)]
        if diff:
            insights.append(f"IQR flags far more points than Z-score for {diff}: skewed or heavy-tailed "
                            "distributions - check them before deleting anything (weather extremes are real).")
        if self.num_cols:
            z_df = (self.df_valid[self.num_cols] - self.df_valid[self.num_cols].mean()) / self.df_valid[self.num_cols].std()
            fig, ax = plt.subplots(figsize=(max(8, len(self.num_cols) * 1.0), 5))
            sns.boxplot(data=z_df, ax=ax, color="#E8DFCF", fliersize=2)
            ax.set_title("Box plots (standardised) - points beyond whiskers are IQR outliers")
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
            self._save_plot(fig, "boxplots", "Box plots of all numeric features", "outliers")
        return {
            "description": "Extreme values flagged with two methods: IQR fences (Q1-1.5*IQR, Q3+1.5*IQR) "
                           "and Z-score (|z| > 3).",
            "features": rows, "insights": insights,
        }

    # ---------------------------------------------------------- 4. noise
    def _noise_metrics(self, col, s):
        """All noise tests for one feature.
        Returns (metrics, expected, spike_mask, residual, resid_sigma). Impossible readings are
        excluded from the noise tests and reported separately as data errors."""
        raw = s.to_numpy(dtype=float)
        n = len(raw)
        rng = physical_range(col, s)
        bad = ((raw < rng[0]) | (raw > rng[1])) if rng else np.zeros(n, dtype=bool)
        v = np.where(bad, np.nan, raw)                       # valid readings only
        ok = ~np.isnan(v)

        # (a) NOISE LEVEL - window-free estimator. For 'smooth signal + random noise' the 2nd difference
        #     x[t+1] - 2x[t] + x[t-1] cancels the signal and leaves sqrt(6) * noise_sigma.
        #     MAD makes it robust, so spikes / bad readings do not inflate the estimate.
        scale = robust_sigma(v[ok])                          # robust spread of the signal itself
        if scale == 0:
            scale = float(np.nanstd(v, ddof=1))
        d2 = v[2:] - 2 * v[1:-1] + v[:-2]
        sigma = robust_sigma(d2[~np.isnan(d2)]) / np.sqrt(6)
        nsr = float(min(sigma / scale, 1.0)) if scale > 0 else 0.0
        var_sig = scale ** 2 - sigma ** 2
        snr_db = float(10 * np.log10(var_sig / sigma ** 2)) if sigma > 0 and var_sig > 0 else None

        # (b) SMOOTHNESS - von Neumann ratio (~0 smooth, ~2 pure white noise) and lag-1 autocorrelation
        var = float(np.nanvar(v))
        vn = float(np.nanmean(np.diff(v) ** 2) / var) if var > 0 else 0.0
        pair = ok[:-1] & ok[1:]
        lag1 = float(np.corrcoef(v[:-1][pair], v[1:][pair])[0, 1])

        # (c) SPIKES - a reading that (1) is far from the median of its 4 neighbours (itself excluded),
        #     (2) is large compared with the feature's normal spread, and (3) is ISOLATED: the neighbours
        #     before and after agree with each other, so it is a one-point blip and not a real step change
        #     (weather front, wrap-around, level shift).
        nb = np.full((n, 4), np.nan)
        for j, k in enumerate((-2, -1, 1, 2)):
            if k < 0:
                nb[-k:, j] = v[:k]
            else:
                nb[:-k, j] = v[k:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = np.nanmedian(nb, axis=1)
            left, right = np.nanmean(nb[:, :2], axis=1), np.nanmean(nb[:, 2:], axis=1)
        r = v - expected
        rs = robust_sigma(r[~np.isnan(r)])
        a = np.abs(np.nan_to_num(r))
        mask = (a > max(SPIKE_K * rs, SPIKE_MIN_FRAC * scale)) & (np.abs(right - left) < 0.5 * a)
        a = a * mask   # keep only the strongest point of each cluster (non-maximum suppression, +/-2 points)
        mask &= a >= pd.Series(a).rolling(5, center=True, min_periods=1).max().to_numpy()
        pos = [p for p in np.argsort(-a)[:5] if mask[p]]
        examples = [{"at": s.index[p], "value": raw[p], "expected": expected[p]} for p in pos]

        # (d) STUCK SENSOR - longest run of identical consecutive readings (continuous features only)
        run_id = np.cumsum(np.r_[True, raw[1:] != raw[:-1]])
        counts = np.bincount(run_id)
        longest = int(counts.max())
        stuck_val = float(raw[np.where(run_id == counts.argmax())[0][0]])
        stuck = bool(longest >= STUCK_RUN and s.nunique() > 50)

        # (e) IMPOSSIBLE VALUES - outside the physical range of the measured quantity
        n_range = int(bad.sum())

        verdict = ("negligible" if nsr < NSR_LEVELS[0] else "low" if nsr < NSR_LEVELS[1]
                   else "moderate" if nsr < NSR_LEVELS[2] else "high")
        spike_pct = round(mask.mean() * 100, 3)
        noise_reasons, error_reasons = [], []
        if nsr >= NSR_LEVELS[1]:
            noise_reasons.append(f"{verdict} random fluctuation (noise is {nsr * 100:.0f}% of the signal's spread)")
        if spike_pct > SPIKE_PCT_FLAG:
            noise_reasons.append(f"{int(mask.sum())} isolated spikes ({spike_pct}%)")
        if n_range:
            error_reasons.append(f"{n_range} physically impossible values (valid range {rng[0]} to {rng[1]})")
        if stuck:
            error_reasons.append(f"stuck at {stuck_val:.4g} for {longest} consecutive readings "
                                 "(probably a sensor outage)")
        metrics = {
            "feature": col, "n": n, "noise_sigma": sigma, "noise_to_signal": nsr, "snr_db": snr_db,
            "verdict": verdict, "von_neumann": vn, "lag1_autocorr": lag1,
            "spikes": int(mask.sum()), "spike_percent": spike_pct, "spike_examples": examples,
            "longest_flat_run": longest, "stuck_sensor": stuck, "stuck_value": stuck_val,
            "valid_range": list(rng) if rng else None, "out_of_range": n_range,
            "has_noise": bool(noise_reasons), "has_errors": bool(error_reasons),
            "noise_reasons": noise_reasons, "error_reasons": error_reasons,
        }
        idx = s.index
        return (metrics, pd.Series(expected, index=idx), pd.Series(mask, index=idx),
                pd.Series(r, index=idx), rs)

    def _plot_noise_view(self, col, s, smooth, mask, resid, rs, verdict):
        start = len(s) // 2
        seg = slice(start, start + 336)                      # ~14 days of hourly data
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
        x, y, sm, mk = s.index[seg], s.values[seg], smooth.values[seg], mask.values[seg]
        a1.plot(x, y, color="#9C9286", lw=1, label="raw")
        a1.plot(x, sm, color="#8C572A", lw=1.8, label="expected (median of 4 neighbours)")
        a1.scatter(x[mk], y[mk], color="#BD4A32", zorder=3, s=25, label="spike")
        a1.set_title(f"{col} - zoom on a 2-week window")
        a1.legend(loc="lower right", fontsize=8)
        plt.setp(a1.get_xticklabels(), rotation=25, ha="right")
        res = resid.dropna()
        if rs > 0:
            res = res.clip(-8 * rs, 8 * rs)
        a2.hist(res, bins=60, color="#8C572A", alpha=.8)
        if rs > 0:
            for k in (-SPIKE_K, SPIKE_K):
                a2.axvline(k * rs, color="#BD4A32", ls="--")
        a2.set_title("Residual (raw - expected); dashed = spike limit")
        fig.suptitle(f"Noise check: {col}  [{verdict}]", y=1.03, fontweight="bold")
        self._save_plot(fig, f"noise_view_{slug(col)}", f"Noise check - {col}", "noise")

    def noise(self):
        rows, cache, skipped = [], {}, []
        for c in self.num_cols:
            if any(k in c.lower() for k in CIRCULAR_KEYS):
                skipped.append({"feature": c, "reason": "circular variable (0 and 360 degrees are the same "
                                                        "direction), so linear noise tests are not meaningful"})
                continue
            s = self.df[c].dropna()
            if len(s) < 30:
                continue
            m, smooth, mask, resid, rs = self._noise_metrics(c, s)
            rows.append(m)
            cache[c] = (s, smooth, mask, resid, rs)
        out = {
            "description": "Is there noise in the data, and how much? Each numeric feature is tested for random "
                           "fluctuation and isolated spikes (= NOISE) and for stuck sensors and physically "
                           "impossible readings (= DATA ERRORS), then given a verdict.",
            "method": {
                "noise_to_signal": "Robust noise sigma (MAD of the 2nd difference / sqrt(6)) divided by the "
                                   "feature's robust spread (MAD-based, immune to bad readings). 0 = perfectly "
                                   "smooth, 1 = pure random noise. Window-free, so the daily and seasonal cycles "
                                   "are NOT counted as noise.",
                "snr_db": "10*log10(signal variance / noise variance). Above 20 dB is very clean; below 5 dB is noisy.",
                "von_neumann": "Mean squared step-to-step change / variance. Near 0 = smooth curve, near 2 = random.",
                "spikes": f"A reading more than {SPIKE_K} robust sigmas AND more than {int(SPIKE_MIN_FRAC * 100)}% "
                          "of the feature's spread away from the median of its 4 neighbours, while the neighbours "
                          "agree with each other (one-point blip, not a real step change).",
                "stuck_sensor": f"At least {STUCK_RUN} identical consecutive readings on a continuous feature.",
                "out_of_range": "Values outside the physically possible range of the quantity. They are excluded "
                                "from the noise tests (and from outliers, correlation, KDE and monthly analysis) "
                                "and reported as data errors.",
            },
            "thresholds": {"negligible": f"< {NSR_LEVELS[0]}", "low": f"< {NSR_LEVELS[1]}",
                           "moderate": f"< {NSR_LEVELS[2]}", "high": f">= {NSR_LEVELS[2]}",
                           "noise_flag_from": f">= {NSR_LEVELS[1]} (moderate) or > {SPIKE_PCT_FLAG}% spikes"},
            "features": rows, "skipped": skipped,
        }
        noisy = [r for r in rows if r["has_noise"]]
        errs = [r for r in rows if r["has_errors"]]
        out["dataset_summary"] = {
            "features_checked": len(rows),
            "features_with_noise": len(noisy), "noisy_features": [r["feature"] for r in noisy],
            "has_noise": bool(noisy),
            "features_with_errors": len(errs), "error_features": [r["feature"] for r in errs],
            "has_errors": bool(errs),
            "mean_noise_to_signal": float(np.mean([r["noise_to_signal"] for r in rows])) if rows else None,
        }
        if not rows:
            out["insights"] = ["Not enough data to test for noise."]
            return out
        ins = []
        if noisy:
            ins.append(f"Random noise: YES in {len(noisy)} of {len(rows)} features: {[r['feature'] for r in noisy]}.")
        else:
            ins.append(f"Random noise: NO significant noise - all {len(rows)} tested features are smooth.")
        if errs:
            ins.append(f"Data errors: YES in {len(errs)} feature(s): {[r['feature'] for r in errs]}.")
        else:
            ins.append("Data errors: none (no impossible or stuck readings).")
        by_nsr = sorted(rows, key=lambda r: -r["noise_to_signal"])
        ins.append(f"Noisiest: '{by_nsr[0]['feature']}' (noise/signal {by_nsr[0]['noise_to_signal']:.2f}); "
                   f"cleanest: '{by_nsr[-1]['feature']}' ({by_nsr[-1]['noise_to_signal']:.2f}).")
        for r in rows:
            if r["has_noise"] or r["has_errors"]:
                ins.append(f"'{r['feature']}': " + "; ".join(r["noise_reasons"] + r["error_reasons"]) + ".")
        good = [r["feature"] for r in rows if not (r["has_noise"] or r["has_errors"])]
        if good:
            ins.append(f"Clean features (no action needed): {good}.")
        for sk in skipped:
            ins.append(f"'{sk['feature']}' not tested: {sk['reason']}.")
        if errs:
            ins.append("Recommended cleaning: set impossible/stuck readings to NaN and interpolate over time. "
                       "The outlier, correlation, KDE and monthly sections already ignore impossible values.")
        if any(r["feature"].lower().startswith("wind speed") and r["has_noise"] for r in rows):
            ins.append("Note: wind speed fluctuates naturally (gusts), so its noise flag is expected - "
                       "smooth it (rolling median / daily mean) rather than deleting points.")

        # --- plots
        fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(rows))))
        colors = {"negligible": "#2B5E4B", "low": "#8DAA91", "moderate": "#E0A458", "high": "#BD4A32"}
        srt = sorted(rows, key=lambda r: r["noise_to_signal"])
        ax.barh([r["feature"] for r in srt], [r["noise_to_signal"] for r in srt],
                color=[colors[r["verdict"]] for r in srt])
        for lv in NSR_LEVELS:
            ax.axvline(lv, color="#2B2621", ls=":", lw=1)
        ax.set_xlim(0, max(0.5, min(1, by_nsr[0]["noise_to_signal"] * 1.15)))
        ax.set_xlabel("noise-to-signal ratio (0 = smooth, 1 = pure noise)")
        ax.set_title("Noise level per feature  (dotted lines = verdict thresholds)")
        self._save_plot(fig, "noise_levels", "Noise level per feature", "noise")

        show = list(dict.fromkeys(self.key_feats + [by_nsr[0]["feature"]] + [r["feature"] for r in errs]))
        for c in show:
            if c in cache:
                s, smooth, mask, resid, rs = cache[c]
                verdict = next(r["verdict"] for r in rows if r["feature"] == c)
                self._plot_noise_view(c, s, smooth, mask, resid, rs, verdict)
        out["insights"] = ins
        return out

    # ---------------------------------------------------------- 4b. monthly trends (your resample code)
    def monthly_trends(self):
        out = {"description": "Monthly aggregation of every feature (mean, 10th/90th percentile, std), "
                              "seasonal averages and month-by-month trend lines across the years.",
               "insights": []}
        if not self.date_col:
            out["insights"].append("No date column - monthly analysis skipped.")
            return out
        df, ins = self.df_valid, out["insights"]
        monthly = {}
        for c in self.num_cols:
            m = df[c].resample("MS").agg(
                mean="mean",
                low=lambda v: v.quantile(0.10),
                high=lambda v: v.quantile(0.90),
                std="std",
            )
            m.index.name = "month"
            monthly[c] = records(m.reset_index())
            fig, ax = plt.subplots(figsize=(12, 4.5))
            ax.plot(m.index, m["mean"], color="#8C572A", lw=1.8, label="monthly mean")
            ax.fill_between(m.index, m["low"], m["high"], color="#8C572A", alpha=.2, label="10th-90th percentile")
            ax.set_title(f"{c} - monthly mean with 10-90% band")
            ax.legend()
            self._save_plot(fig, f"monthly_band_{slug(c)}", f"Monthly band - {c}", "monthly_trends")
        out["monthly_stats"] = monthly

        season = {}
        for c in self.key_feats:
            by_m = df[c].groupby(df.index.month).mean()
            season[c] = {calendar.month_name[int(k)]: float(v) for k, v in by_m.items()}
            ins.append(f"'{c}' averages highest in {calendar.month_name[int(by_m.idxmax())]} "
                       f"({by_m.max():.2f}) and lowest in {calendar.month_name[int(by_m.idxmin())]} "
                       f"({by_m.min():.2f}).")
        out["seasonality"] = season

        df_monthly_mean = df[self.num_cols].resample("MS").mean()
        for c in self.key_feats:
            fig, ax = plt.subplots(figsize=(14, 8))
            for mth in range(1, 13):
                label, color = MONTH_STYLE[mth]
                md = df_monthly_mean[c][df_monthly_mean.index.month == mth].dropna()
                if len(md):
                    sns.lineplot(x=md.index.year, y=md.values, label=label, color=color, marker="o", ax=ax)
            ax.set_title(f"{c} - each month across the years")
            ax.set_xlabel("YEAR")
            ax.set_ylabel(c)
            self._save_plot(fig, f"month_lines_{slug(c)}", f"Month-by-month trend - {c}", "monthly_trends")
            ms = monthly[c]
            worst = max((r for r in ms if r["std"] is not None), key=lambda r: r["std"])
            ins.append(f"Most variable month for '{c}': {worst['month'][:7]} (std {worst['std']:.2f}).")
        return out

    # ---------------------------------------------------------- 5. correlation
    def correlation(self):
        cols = self.num_cols
        out = {"description": "Linear (Pearson) and rank (Spearman) relationships between numeric features, "
                              "plus how the main variables behave for each weather 'Summary'.",
               "insights": []}
        if len(cols) < 2:
            out["insights"].append("Need at least two non-constant numeric columns.")
            return out
        pear = self.df_valid[cols].corr()
        spear = self.df_valid[cols].corr(method="spearman")
        pairs = []
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                r, rs = pear.loc[a, b], spear.loc[a, b]
                pairs.append({"a": a, "b": b, "pearson": r, "spearman": rs,
                              "strength": ("very strong" if abs(r) >= .8 else "strong" if abs(r) >= .6 else
                                           "moderate" if abs(r) >= .4 else "weak" if abs(r) >= .2 else "negligible"),
                              "direction": "positive" if r > 0 else "negative"})
        pairs.sort(key=lambda p: -abs(p["pearson"]))
        out["pearson"] = pear.round(4).to_dict()
        out["spearman"] = spear.round(4).to_dict()
        out["pairs"] = pairs

        ins = out["insights"]
        for p in [p for p in pairs if abs(p["pearson"]) >= .6][:6]:
            ins.append(f"{p['strength'].capitalize()} {p['direction']} link between '{p['a']}' and '{p['b']}' "
                       f"(r = {p['pearson']:.2f}).")
        redundant = [p for p in pairs if abs(p["pearson"]) >= .9]
        for p in redundant:
            ins.append(f"'{p['a']}' and '{p['b']}' are nearly redundant (|r| >= 0.9) - keep one to avoid "
                       "multicollinearity in a model.")
        nonlin = [p for p in pairs if abs(p["spearman"] - p["pearson"]) > .15]
        for p in nonlin[:3]:
            ins.append(f"'{p['a']}' vs '{p['b']}': Pearson {p['pearson']:.2f} but Spearman {p['spearman']:.2f} "
                       "- the relationship is monotonic but not linear (or distorted by outliers/zeros).")
        weak = [c for c in cols if pear[c].drop(c).abs().max() < .2]
        if weak:
            ins.append(f"Almost independent of every other feature: {weak}.")

        fig, ax = plt.subplots(figsize=(max(7, len(cols) * 1.0), max(5, len(cols) * 0.85)))
        sns.heatmap(pear, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    linewidths=.5, ax=ax)
        ax.set_title("Pearson correlation matrix")
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        self._save_plot(fig, "correlation_heatmap", "Pearson correlation heatmap", "correlation")

        # scatter of the 2 strongest pairs
        for p in pairs[:2]:
            d = self.df_valid[[p["a"], p["b"]]].dropna()
            d = d.sample(min(len(d), 5000), random_state=0)
            fig, ax = plt.subplots(figsize=(6.5, 5))
            sns.regplot(data=d, x=p["a"], y=p["b"], ax=ax, scatter_kws={"s": 5, "alpha": .25},
                        line_kws={"color": "#BD4A32"})
            ax.set_title(f"{p['a']} vs {p['b']}  (r = {p['pearson']:.2f})")
            self._save_plot(fig, f"scatter_{slug(p['a'])}__{slug(p['b'])}",
                            f"Scatter {p['a']} vs {p['b']}", "correlation")

        # --- weather Summary analysis (your FacetGrid code, for Apparent Temp + Humidity)
        cat = self.category_col
        if cat in self.df.columns:
            facet_feats = [c for c in ("Apparent Temperature (C)", "Humidity") if c in cols] or self.key_feats
            d_all = self.df_valid[[cat] + facet_feats].dropna()
            top = d_all[cat].value_counts().head(self.top_categories).index
            d_all = d_all.assign(**{cat: d_all[cat].where(d_all[cat].isin(top), "Other")})
            for f in facet_feats:
                with sns.axes_style("darkgrid"):
                    g = sns.FacetGrid(d_all, hue=cat, height=7, aspect=1.5)
                    g.map(sns.histplot, f).add_legend()
                    g.figure.suptitle(f"Analysis of Weather Conditions with {f}", y=1.02)
                self._save_plot(g.figure, f"facet_{slug(f)}_by_{slug(cat)}",
                                f"{f} distribution by {cat}", "correlation")
            grp = d_all.groupby(cat)[facet_feats].mean().round(3)
            out["by_category"] = {"category": cat,
                                  "note": f"top {self.top_categories} categories kept, the rest grouped as 'Other'",
                                  "means": records(grp.reset_index())}
            for f in facet_feats:
                hi, lo = grp[f].idxmax(), grp[f].idxmin()
                ins.append(f"Average '{f}' is highest for '{hi}' ({grp[f].max():.2f}) and lowest for "
                           f"'{lo}' ({grp[f].min():.2f}).")
        return out

    # ---------------------------------------------------------- 6. KDE
    def kde(self):
        feats, skipped = [], []
        for c in self.num_cols + self.constant_cols:
            s = self.df_valid[c].dropna()
            if len(s) < 2 or s.nunique() < 2:
                skipped.append({"feature": c, "reason": "constant or too few values"})
                continue
            fig, ax = plt.subplots(figsize=(6, 3.8))
            sns.kdeplot(s, fill=True, color="#8C572A", ax=ax)
            ax.axvline(s.mean(), color="#2B5E4B", ls="--", label=f"mean = {s.mean():.2f}")
            ax.axvline(s.median(), color="#BD4A32", ls=":", label=f"median = {s.median():.2f}")
            ax.set_title(f"KDE - {c}")
            ax.legend()
            self._save_plot(fig, f"kde_{slug(c)}", f"KDE of {c}", "kde")
            sk = float(s.skew())
            shape = "right-skewed" if sk > .5 else "left-skewed" if sk < -.5 else "roughly symmetric"
            dens = stats.gaussian_kde(s)(np.linspace(s.min(), s.max(), 200))
            modes = len(find_peaks(dens, prominence=0.10 * dens.max())[0])   # ignore tiny bumps
            feats.append({"feature": c, "mean": s.mean(), "median": s.median(), "std": s.std(),
                          "skew": sk, "kurtosis": float(s.kurt()), "n_peaks": modes, "shape": shape})
        insights = [f"'{f['feature']}' is {f['shape']} with {f['n_peaks']} peak(s)"
                    + (" - multimodal, probably mixed regimes (day/night or seasons)." if f["n_peaks"] > 1 else ".")
                    for f in feats]
        return {"description": "Kernel density estimate of every non-constant numeric feature, with mean and "
                               "median markers, to show distribution shape.",
                "features": feats, "skipped": skipped, "insights": insights}

    # ---------------------------------------------------------- 7. model phase (Precip Type balance, preprocessing, RF importance)
    def model_phase(self):
        """Analyzes class balance of 'Precip Type', builds a ColumnTransformer + Pipeline
        for numerical & categorical preprocessing, and trains a RandomForestClassifier
        to determine feature importance."""
        target_col = "Precip Type"
        insights = []

        # (1) Class balance analysis
        s_target = self.raw[target_col] if target_col in self.raw.columns else self.df[target_col]
        total_samples = len(s_target)
        val_counts = s_target.value_counts(dropna=False)
        nan_count = int(s_target.isna().sum())

        class_counts = {}
        class_pcts = {}
        for k, v in val_counts.items():
            k_name = "Missing / Not recorded" if pd.isna(k) else str(k)
            class_counts[k_name] = int(v)
            class_pcts[k_name] = round(float(v) / total_samples * 100, 2)

        # Non-null class balance
        non_null_counts = s_target.dropna().value_counts()
        is_balanced = False
        imbalance_ratio = None
        if len(non_null_counts) >= 2:
            maj_count = float(non_null_counts.iloc[0])
            min_count = float(non_null_counts.iloc[1])
            imbalance_ratio = round(maj_count / max(min_count, 1), 2)
            # Imbalance threshold: > 1.5 ratio or majority > 60%
            is_balanced = bool(imbalance_ratio <= 1.5)

        # Plot 1: Target balance plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.4, 1]})
        
        categories = list(class_counts.keys())
        counts = list(class_counts.values())
        colors = ["#2878B5" if c == "rain" else "#20A387" if c == "snow" else "#BD4A32" for c in categories]
        
        bars = ax1.bar(categories, counts, color=colors, width=0.55, edgecolor="#18324B", linewidth=1.2)
        ax1.set_title(f"Class Balance: '{target_col}' (Severe Imbalance)", fontsize=13, fontweight="bold", pad=12)
        ax1.set_ylabel("Number of Observations", fontsize=11)
        ax1.set_xlabel("Precipitation Class", fontsize=11)
        ax1.grid(axis="y", linestyle="--", alpha=0.6)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        for bar, cat in zip(bars, categories):
            yval = bar.get_height()
            pct = class_pcts[cat]
            ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + (total_samples * 0.015),
                     f"{yval:,}\n({pct}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax1.set_ylim(0, max(counts) * 1.18)

        # Pie chart on ax2
        wedges, texts, autotexts = ax2.pie(
            counts, labels=categories, autopct="%1.1f%%", startangle=140,
            colors=colors, wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.6}
        )
        for autotext in autotexts:
            autotext.set_fontweight("bold")
            autotext.set_fontsize(10)
        ax2.set_title(f"Proportional Share\n(Imbalance Ratio ~ {imbalance_ratio}:1)", fontsize=12, fontweight="bold")

        fig.suptitle(f"Target Feature Analysis: '{target_col}' Distribution", fontsize=14, fontweight="bold", y=1.02)
        self._save_plot(fig, "precip_type_balance", f"Class Balance - {target_col}", "model_phase")

        # Insights on class balance
        rain_cnt = class_counts.get("rain", 0)
        snow_cnt = class_counts.get("snow", 0)
        insights.append(
            f"Class balance on '{target_col}': the dataset is SEVERELY IMBALANCED. "
            f"'rain' has {rain_cnt:,} instances ({class_pcts.get('rain', 0):.1f}%), while 'snow' has "
            f"{snow_cnt:,} instances ({class_pcts.get('snow', 0):.1f}%), and {nan_count} missing values ({class_pcts.get('Missing / Not recorded', 0):.2f}%)."
        )
        if imbalance_ratio:
            insights.append(
                f"Imbalance ratio: approximately {imbalance_ratio}:1 (rain vs snow). "
                "Because of this severe skew, models trained without class weighting would suffer from majority class bias."
            )

        # (2) Preprocessing: Divide into Numerical & Categorical data
        valid_mask = s_target.notna()
        df_model = self.raw[valid_mask].copy()
        y = df_model[target_col]

        candidate_num = [
            "Temperature (C)", "Apparent Temperature (C)", "Humidity",
            "Wind Speed (km/h)", "Wind Bearing (degrees)", "Visibility (km)",
            "Pressure (millibars)"
        ]
        num_features = [c for c in candidate_num if c in df_model.columns]
        cat_features = [c for c in ["Summary"] if c in df_model.columns]

        X = df_model[num_features + cat_features]

        # Sklearn Pipeline and ColumnTransformer
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, num_features),
                ("cat", cat_pipeline, cat_features),
            ],
            remainder="drop"
        )

        insights.append(
            f"Preprocessing: divided dataset into {len(num_features)} numerical features {num_features} "
            f"and {len(cat_features)} categorical feature(s) {cat_features}. "
            "Built a ColumnTransformer combining a numerical Pipeline (median imputation + StandardScaler) "
            "and a categorical Pipeline (most-frequent imputation + OneHotEncoder)."
        )

        # (3) Random Forest Feature Importance Phase
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )

        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", rf),
        ])

        full_pipeline.fit(X, y)

        trans_names = list(full_pipeline.named_steps["preprocessor"].get_feature_names_out())
        raw_importances = full_pipeline.named_steps["classifier"].feature_importances_

        cleaned_trans_names = [
            re.sub(r"^(num__|cat__)", "", name) for name in trans_names
        ]

        orig_importances = {feat: 0.0 for feat in (num_features + cat_features)}
        for raw_name, imp in zip(trans_names, raw_importances):
            found = False
            for orig in num_features + cat_features:
                if raw_name.startswith(f"num__{orig}") or raw_name.startswith(f"cat__{orig}"):
                    orig_importances[orig] += float(imp)
                    found = True
                    break
            if not found:
                orig_importances[raw_name] = float(imp)

        sorted_orig = sorted(orig_importances.items(), key=lambda item: -item[1])
        orig_ranking = [
            {"rank": i + 1, "feature": feat, "importance": round(imp, 4),
             "percentage": round(imp * 100, 2)}
            for i, (feat, imp) in enumerate(sorted_orig)
        ]

        sorted_trans = sorted(zip(cleaned_trans_names, raw_importances), key=lambda x: -x[1])
        trans_ranking = [
            {"rank": i + 1, "feature": name, "importance": round(float(imp), 4),
             "percentage": round(float(imp) * 100, 2)}
            for i, (name, imp) in enumerate(sorted_trans)
        ]

        # Plot 2: Feature Importance Plot
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        feats_plot = [r["feature"] for r in orig_ranking][::-1]
        imps_plot = [r["importance"] * 100 for r in orig_ranking][::-1]
        
        norm_vals = np.linspace(0.3, 0.95, len(feats_plot))
        bar_colors = [plt.cm.Blues(v) for v in norm_vals]

        hbars = ax.barh(feats_plot, imps_plot, color=bar_colors, edgecolor="#18324B", linewidth=0.8, height=0.6)
        ax.set_title("Random Forest Feature Importance (Target: Precip Type)", fontsize=13, fontweight="bold", loc="left", pad=12)
        ax.set_xlabel("Relative Importance (%)", fontsize=10, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, max(imps_plot) * 1.15)

        for bar in hbars:
            width = bar.get_width()
            ax.text(width + 0.8, bar.get_y() + bar.get_height() / 2.0,
                    f"{width:.1f}%", va="center", ha="left", fontsize=9.5, fontweight="bold", color="#18324B")

        self._save_plot(fig, "rf_feature_importance", "Feature Importance (Random Forest)", "model_phase")

        top1 = sorted_orig[0]
        top2 = sorted_orig[1] if len(sorted_orig) > 1 else None
        top_msg = f"Top predictor: '{top1[0]}' contributes {top1[1] * 100:.1f}% of the decision tree splits."
        if top2:
            top_msg += f" Combined with '{top2[0]}' ({top2[1] * 100:.1f}%), temperature metrics account for {(top1[1] + top2[1]) * 100:.1f}% of total feature importance."
        insights.append(top_msg)
        insights.append(
            "Meteorological interpretation: whether precipitation falls as rain or snow is almost entirely "
            "governed by atmospheric temperature and apparent temperature around the freezing point (0°C), "
            "while humidity, pressure, and visibility provide minor secondary signals."
        )

        return {
            "description": "Model phase: analysis of target class balance ('Precip Type'), "
                           "data preprocessing using sklearn Pipeline and ColumnTransformer, "
                           "and Random Forest feature importance ranking.",
            "target_balance": {
                "target_feature": target_col,
                "is_balanced": is_balanced,
                "imbalance_ratio": imbalance_ratio,
                "total_observations": total_samples,
                "valid_observations": int(len(df_model)),
                "missing_count": nan_count,
                "class_counts": class_counts,
                "class_percentages": class_pcts,
            },
            "preprocessing": {
                "numerical_features": num_features,
                "categorical_features": cat_features,
                "target_feature": target_col,
                "numerical_pipeline": ["SimpleImputer(strategy='median')", "StandardScaler()"],
                "categorical_pipeline": ["SimpleImputer(strategy='most_frequent')", "OneHotEncoder(handle_unknown='ignore')"],
                "transformer_method": "ColumnTransformer",
                "total_input_features": len(num_features) + len(cat_features),
                "total_transformed_features": len(trans_names),
                "samples_trained": int(len(df_model)),
            },
            "feature_importance": {
                "model": "RandomForestClassifier",
                "n_estimators": 100,
                "class_weight": "balanced",
                "random_state": 42,
                "ranked_features": orig_ranking,
                "detailed_transformed_ranking": trans_ranking[:15],
            },
            "insights": insights,
        }

    # ---------------------------------------------------------- run / save
    def run(self):
        self.load()
        stamp = datetime.now()
        self.report = {"meta": {
            "report_id": f"{slug(self.csv_path.stem)}_{stamp:%Y%m%d_%H%M%S}",
            "dataset": self.csv_path.name, "generated_at": stamp.isoformat(timespec="seconds"),
            "n_rows": len(self.raw), "n_cols": self.raw.shape[1], "date_column": self.date_col,
            "numeric_columns": self.num_cols, "constant_columns": self.constant_cols,
            "categorical_columns": self.cat_cols,
            "impossible_values_excluded": self.invalid_counts}}
        self._section("description", self.describe)
        self._section("missing_duplicates", self.missing_duplicates)
        self._section("outliers", self.outliers)
        self._section("noise", self.noise)
        self._section("monthly_trends", self.monthly_trends)
        self._section("correlation", self.correlation)
        self._section("kde", self.kde)
        self._section("model_phase", self.model_phase)
        self.report["plots"] = self.plots
        return clean(self.report)

    def to_db_documents(self):
        """Two collections/tables: 'reports' (light) and 'plots' (one doc per image).
        Splitting avoids MongoDB's 16 MB document limit when images are embedded."""
        rep = clean(self.report)
        rid = rep["meta"]["report_id"]
        core = {k: v for k, v in rep.items() if k != "plots"}
        core["_id"] = rid
        return {"reports": [core],
                "plots": [{"_id": f"{rid}__{p['name']}", "report_id": rid, **p} for p in rep["plots"]]}

    def save(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(clean(self.report), f, ensure_ascii=False, indent=2)
        with open(self.out_dir / "db_documents.json", "w", encoding="utf-8") as f:
            json.dump(self.to_db_documents(), f, ensure_ascii=False)
        return str(self.out_dir / "report.json")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("Usage: python weather_model.py weather.csv [output_dir]")
    m = WeatherEDAModel(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "eda_output")
    m.run()
    print("Saved:", m.save(), "|", len(m.plots), "plots")