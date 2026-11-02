#!/usr/bin/env python3
"""Synthetic twin of a restricted dataset, generated without Claude seeing the data.

Fits a Gaussian copula (per-column marginals + rank correlations) to the real
file and samples new rows. Only the synthetic file and a report computed ON THE
SYNTHETIC DATA are written; stdout carries paths and counts, never values.

Usage:
    python3 generate.py <real-file> [--config controls.json] [--out data/synthetic]
                        [--rows N] [--seed S] [--min-count 10]

Privacy measures (on by default):
    - numeric tails clipped to the real p1..p99 range (no exact extremes leak)
    - categories with fewer than --min-count rows merged into "Other"
    - near-unique text columns (IDs, names, free text) replaced by fake IDs
    - exceptions are reported by type only (tracebacks can quote data values)

Controls (JSON, all optional):
{
  "rows": 5000, "seed": 1,
  "correlation": 1.0,                 # 0 = independent columns, 1 = as estimated
  "correlations": {"age,income": 0.6},  # force pairwise latent correlation
  "columns": {
    "income": {"shift": 0.5, "scale": 1.2, "skew": 1.0, "missing_rate": 0.05},
    "region": {"weights": {"North": 0.7, "South": 0.3}},
    "ssn":    {"drop": true},
    "hh_id":  {"type": "id"}
  }
}
  shift: add N standard deviations    scale: multiply spread around the mean
  skew:  >0 right tail heavier, <0 left tail heavier (0 = as estimated)
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

READERS = {".csv": pd.read_csv, ".tsv": lambda p: pd.read_csv(p, sep="\t"),
           ".dta": pd.read_stata, ".parquet": pd.read_parquet,
           ".xlsx": pd.read_excel, ".xls": pd.read_excel, ".sav": pd.read_spss}
WRITERS = {".dta": lambda d, p: d.to_stata(p, write_index=False),
           ".parquet": lambda d, p: d.to_parquet(p, index=False)}
PROBS = np.linspace(0.01, 0.99, 99)  # ponytail: tails clipped at p1/p99 by design


def classify(s, override):
    if override:
        return override
    if pd.api.types.is_bool_dtype(s):
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "integer" if (s.dropna() % 1 == 0).all() and s.nunique() > 10 else (
            "categorical" if s.nunique() <= 10 else "numeric")
    nonnull = s.dropna()
    if len(nonnull) and nonnull.nunique() > 0.5 * len(nonnull):
        return "id"
    return "categorical"


def fit(df, cfg, min_count, rng):
    """Per-column marginal + latent normal scores for the copula."""
    cols, latent = {}, {}
    for name in df.columns:
        c = cfg.get("columns", {}).get(name, {})
        if c.get("drop"):
            continue
        s = df[name]
        kind = classify(s, c.get("type"))
        m = {"type": kind, "missing": float(s.isna().mean())}
        nonnull = s.dropna()
        if kind == "id":
            cols[name] = m
            continue
        if kind == "datetime":
            nonnull = nonnull.astype("int64")
        if kind in ("numeric", "integer", "datetime"):
            x = nonnull.astype(float)
            m["daily"] = kind == "datetime" and bool((nonnull % 86_400_000_000_000 == 0).all())
            m["knots"] = np.quantile(x, PROBS) if len(x) else np.zeros(len(PROBS))
            m["mean"], m["sd"] = float(x.mean()), float(x.std() or 1.0)
            z = pd.Series(norm.ppf(x.rank(method="average") / (len(x) + 1)), index=x.index)
        else:
            counts = nonnull.astype(str).value_counts()
            rare = counts[counts < min_count]
            counts = counts[counts >= min_count]
            if rare.sum():
                counts["Other"] = counts.get("Other", 0) + rare.sum()
            levels = list(counts.index)
            p = (counts / counts.sum()).to_numpy() if len(counts) else np.array([1.0])
            m["levels"], m["p"] = levels or ["Other"], p
            code = nonnull.astype(str).map({k: i for i, k in enumerate(levels)}).fillna(
                levels.index("Other") if "Other" in levels else 0).astype(int)
            lo = np.concatenate([[0], np.cumsum(p)[:-1]])
            u = lo[code] + rng.uniform(size=len(code)) * p[code]  # jitter inside the level's band
            z = pd.Series(norm.ppf(np.clip(u, 1e-6, 1 - 1e-6)), index=nonnull.index)
        latent[name] = z.reindex(df.index)  # NaN where missing; corr() below is pairwise-complete
        cols[name] = m
    return cols, pd.DataFrame(latent)


def nearest_psd(c):
    w, v = np.linalg.eigh(c)
    c = v @ np.diag(np.clip(w, 1e-6, None)) @ v.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def correlation(latent, cfg):
    c = np.nan_to_num(latent.corr().to_numpy()) if latent.shape[1] else np.zeros((0, 0))
    a = float(cfg.get("correlation", 1.0))
    c = a * c + (1 - a) * np.eye(len(c))
    names = list(latent.columns)
    for pair, r in cfg.get("correlations", {}).items():
        x, y = (t.strip() for t in pair.split(","))
        if x not in names or y not in names:
            raise KeyError(f"correlations: unknown column in '{pair}'")
        i, j = names.index(x), names.index(y)
        c[i, j] = c[j, i] = float(r)
    np.fill_diagonal(c, 1.0)
    return nearest_psd(c)


def skew_u(u, k):
    return u ** (1 + k) if k >= 0 else 1 - (1 - u) ** (1 - k)


def sample(cols, latent_names, corr, n, cfg, rng):
    z = rng.standard_normal((n, len(latent_names))) @ np.linalg.cholesky(corr).T
    out = {}
    for name, m in cols.items():
        c = cfg.get("columns", {}).get(name, {})
        if m["type"] == "id":
            out[name] = [f"ID_{i:07d}" for i in rng.permutation(n)]
        else:
            u = skew_u(norm.cdf(z[:, latent_names.index(name)]), float(c.get("skew", 0)))
            if m["type"] in ("numeric", "integer", "datetime"):
                x = np.interp(u, PROBS, m["knots"])
                x = m["mean"] + float(c.get("scale", 1)) * (x - m["mean"]) + float(c.get("shift", 0)) * m["sd"]
                if m["type"] == "integer":
                    x = pd.array(np.round(x).astype("int64"), dtype="Int64")
                if m["type"] == "datetime":
                    x = pd.to_datetime(x.astype("int64")).round("D" if m["daily"] else "s")
                col = pd.Series(x)
            else:
                levels, p = list(m["levels"]), np.asarray(m["p"], float)
                if "weights" in c:
                    unknown = set(c["weights"]) - set(levels)
                    levels += sorted(unknown)
                    p = np.array([float(c["weights"].get(k, 0)) for k in levels])
                    p = p / p.sum()
                if "weights" in c or c.get("skew"):
                    col = pd.Series(rng.choice(levels, size=n, p=p))  # overrides break the copula link for this column
                else:
                    col = pd.Series(np.array(levels)[np.searchsorted(np.cumsum(p), u, side="right").clip(0, len(levels) - 1)])
        col = pd.Series(out[name]) if m["type"] == "id" else col
        miss = float(c.get("missing_rate", m["missing"]))
        if miss:
            hole = rng.uniform(size=n) < miss
            col = col.astype(object).where(~hole, None) if col.dtype == object else col.mask(hole)
        out[name] = col
    return pd.DataFrame(out)


def report(syn, cols, cfg, src_name, md_path, json_path):
    """Describe the SYNTHETIC data only."""
    vars_ = []
    for name, m in cols.items():
        s = syn[name]
        v = {"name": name, "type": m["type"], "missing_pct": round(100 * float(s.isna().mean()), 1),
             "controls": cfg.get("columns", {}).get(name, {})}
        if m["type"] in ("numeric", "integer"):
            x = pd.to_numeric(s, errors="coerce").dropna()
            v.update({k: round(float(f(x)), 4) for k, f in [
                ("mean", np.mean), ("sd", np.std), ("min", np.min), ("p25", lambda a: np.quantile(a, .25)),
                ("median", np.median), ("p75", lambda a: np.quantile(a, .75)), ("max", np.max),
                ("skewness", lambda a: a.skew())]} if len(x) else {})
        elif m["type"] == "datetime":
            x = pd.to_datetime(s.dropna())
            v.update({"min": str(x.min()), "median": str(x.quantile(.5)), "max": str(x.max())} if len(x) else {})
        elif m["type"] == "categorical":
            freq = s.dropna().value_counts(normalize=True)
            v["levels"] = {str(k): round(100 * float(p), 1) for k, p in freq.items()}
        vars_.append(v)
    num = syn[[c for c, m in cols.items() if m["type"] in ("numeric", "integer")]].apply(pd.to_numeric, errors="coerce")
    corr = num.corr(method="spearman").round(2) if num.shape[1] > 1 else pd.DataFrame()
    json_path.write_text(json.dumps({"source": src_name, "rows": len(syn), "variables": vars_,
                                     "global_controls": {k: v for k, v in cfg.items() if k != "columns"},
                                     "spearman_correlation": corr.to_dict()}, indent=2, default=str))

    lines = [f"# Synthetic data report: {src_name}", "",
             "**SYNTHETIC DATA.** Every statistic below is computed on the synthetic file, not the real one. "
             "Use it to write and debug code; run final analyses on the real data outside Claude.", "",
             f"- Rows: {len(syn)}  ", f"- Variables: {len(cols)}  ",
             f"- Global controls: `{json.dumps({k: v for k, v in cfg.items() if k != 'columns'})}`", "",
             "| Variable | Type | Missing % | Distribution | Controls |", "|---|---|---|---|---|"]
    for v in vars_:
        if v["type"] in ("numeric", "integer") and "mean" in v:
            d = (f"mean {v['mean']:g}, sd {v['sd']:g}, min {v['min']:g}, p25 {v['p25']:g}, "
                 f"median {v['median']:g}, p75 {v['p75']:g}, max {v['max']:g}, skew {v['skewness']:.2f}")
        elif v["type"] == "datetime" and "min" in v:
            d = f"{v['min']} to {v['max']}, median {v['median']}"
        elif v["type"] == "categorical":
            d = ", ".join(f"{k} {p}%" for k, p in list(v["levels"].items())[:12])
            d += " ..." if len(v["levels"]) > 12 else ""
        elif v["type"] == "id":
            d = "fake unique IDs (real values never copied)"
        else:
            d = "all missing"
        lines.append(f"| `{v['name']}` | {v['type']} | {v['missing_pct']} | {d} | "
                     f"{json.dumps(v['controls']) if v['controls'] else ''} |")
    if not corr.empty:
        lines += ["", "## Spearman correlations (numeric, synthetic)", "",
                  "| | " + " | ".join(f"`{c}`" for c in corr.columns) + " |",
                  "|---" * (len(corr.columns) + 1) + "|"]
        lines += [f"| `{r}` | " + " | ".join(f"{x:.2f}" for x in corr.loc[r]) + " |" for r in corr.index]
    md_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("--config")
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--rows", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--min-count", type=int, default=10, help="merge categories rarer than this (default 10)")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")  # warning text can quote data values; stderr stays clean

    src = Path(a.source)
    cfg = json.loads(Path(a.config).read_text()) if a.config else {}
    seed = a.seed if a.seed is not None else cfg.get("seed", 0)
    rng = np.random.default_rng(seed)
    try:
        df = READERS[src.suffix.lower()](src)
        for c in df.columns[df.dtypes == object]:  # dates stored as text (csv) -> datetime
            parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
            if df[c].notna().any() and parsed.notna().sum() == df[c].notna().sum():
                df[c] = parsed
        cols, latent = fit(df, cfg, a.min_count, rng)
        corr = correlation(latent, cfg)
        n = a.rows or cfg.get("rows") or len(df)
        syn = sample(cols, list(latent.columns), corr, n, cfg, rng)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        data_path = out / f"{src.stem}_synthetic{src.suffix if src.suffix in WRITERS else '.csv'}"
        WRITERS.get(data_path.suffix, lambda d, p: d.to_csv(p, index=False))(syn, data_path)
        report(syn, cols, {**cfg, "seed": seed}, src.name, out / f"{src.stem}_report.md", out / f"{src.stem}_report.json")
    except Exception as e:  # never echo the message: it can quote real values
        sys.exit(f"generation failed: {type(e).__name__} (message suppressed; run it yourself to see details)")
    print(f"wrote {data_path} ({len(syn)} rows, {len(cols)} columns)")
    print(f"wrote {out / f'{src.stem}_report.md'} and .json")


if __name__ == "__main__":
    main()
