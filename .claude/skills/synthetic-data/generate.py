#!/usr/bin/env python3
"""Synthetic twin of a restricted dataset, generated without Claude seeing the data.

Fits a Gaussian copula (per-column marginals + rank correlations) to the real
file and samples new rows. Only the synthetic file and a report computed ON THE
SYNTHETIC DATA are written; stdout carries paths and counts, never values.

Usage:
    python3 generate.py <real-file> [--config controls.json] [--out data/synthetic]
                        [--rows N] [--seed S] [--min-count 10]
                        [--engine auto|python|stata] [--stata-above-mb 250]

Labels: Stata variable labels and value labels are read from .dta files, shown in
the report, and written back into a synthetic .dta (labelled codes stay codes).

Engine: a .dta larger than --stata-above-mb is fitted by Stata (batch mode) so the
data never loads into Python; Python only samples from Stata's fitted summaries.
Stata is found via $STATA_BIN, PATH (stata-mp/stata-se/stata) or /Applications.

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
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

READERS = {".csv": pd.read_csv, ".tsv": lambda p: pd.read_csv(p, sep="\t"),
           ".dta": lambda p: pd.read_stata(p, convert_categoricals=False), ".parquet": pd.read_parquet,
           ".xlsx": pd.read_excel, ".xls": pd.read_excel, ".sav": pd.read_spss}
PROBS = np.linspace(0.01, 0.99, 99)  # ponytail: tails clipped at p1/p99 by design


class SafeError(Exception):
    """An error whose message we wrote ourselves, so it is safe to print."""


def dta_labels(src):
    """Variable labels and per-variable value labels from a .dta header (no data read)."""
    with pd.io.stata.StataReader(src) as r:
        var_labels, sets = r.variable_labels(), r.value_labels()
        attached = getattr(r, "_lbllist", None) or [n if n in sets else "" for n in var_labels]
    return {"var_labels": {k: v for k, v in var_labels.items() if v},
            "value_labels": {n: {int(k): v for k, v in sets[l].items()}
                             for n, l in zip(var_labels, attached) if l in sets}}


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
            numeric = pd.api.types.is_numeric_dtype(nonnull) and not pd.api.types.is_bool_dtype(nonnull)
            keys = nonnull.astype(float) if numeric else nonnull.astype(str)
            counts = keys.value_counts()
            rare = counts[counts < min_count]
            counts = counts[counts >= min_count]
            other = (float(max(counts.index, default=0)) + 1) if numeric else "Other"  # same rule as the Stata engine
            if rare.sum():
                counts[other] = counts.get(other, 0) + rare.sum()
            levels = list(counts.index)
            p = (counts / counts.sum()).to_numpy() if len(counts) else np.array([1.0])
            m.update(levels=levels or [other], p=p, numeric=numeric, other=other if rare.sum() else None)
            code = keys.map({k: i for i, k in enumerate(levels)}).fillna(
                levels.index(other) if other in levels else 0).astype(int)
            lo = np.concatenate([[0], np.cumsum(p)[:-1]])
            u = lo[code] + rng.uniform(size=len(code)) * p[code]  # jitter inside the level's band
            z = pd.Series(norm.ppf(np.clip(u, 1e-6, 1 - 1e-6)), index=nonnull.index)
        latent[name] = z.reindex(df.index)  # NaN where missing; corr() below is pairwise-complete
        cols[name] = m
    latent = pd.DataFrame(latent)
    return cols, list(latent.columns), (latent.corr().to_numpy() if latent.shape[1] else np.zeros((0, 0)))


def nearest_psd(c):
    w, v = np.linalg.eigh(c)
    c = v @ np.diag(np.clip(w, 1e-6, None)) @ v.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def correlation(base, names, cfg):
    c = np.nan_to_num(np.asarray(base, float))
    a = float(cfg.get("correlation", 1.0))
    c = a * c + (1 - a) * np.eye(len(c))
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


def level_key(k, m, codes):
    """A weights key from JSON -> the level it names (label text or code for labelled numerics)."""
    if not m.get("numeric"):
        return str(k)
    if k in codes:
        return float(codes[k])
    try:
        return float(k)
    except ValueError:
        raise SafeError(f"weights: '{k}' is neither a code nor a value label")


def sample(cols, latent_names, corr, n, cfg, rng, value_labels):
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
                    codes = {v: k for k, v in value_labels.get(name, {}).items()}
                    w = {level_key(k, m, codes): float(v) for k, v in c["weights"].items()}
                    levels += sorted(set(w) - set(levels))
                    p = np.array([w.get(k, 0) for k in levels])
                    p = p / p.sum()
                if "weights" in c or c.get("skew"):
                    col = pd.Series(rng.choice(levels, size=n, p=p))  # overrides break the copula link for this column
                else:
                    col = pd.Series(np.array(levels)[np.searchsorted(np.cumsum(p), u, side="right").clip(0, len(levels) - 1)])
                if m.get("numeric"):
                    col = col.astype(float)
                    col = col.astype("Int64") if (col % 1 == 0).all() else col
        col = pd.Series(out[name]) if m["type"] == "id" else col
        miss = float(c.get("missing_rate", m["missing"]))
        if miss:
            hole = rng.uniform(size=n) < miss
            col = col.astype(object).where(~hole, None) if col.dtype == object else col.mask(hole)
        out[name] = col
    return pd.DataFrame(out)


# ---------------------------------------------------------------- Stata engine
# Stata reads the real file and writes only fitted summaries (quantile knots,
# level counts, latent correlations) to a temp dir; Python samples from those.

META_DO = r"""version 16
set more off
use `"{src}"', clear
file open fh using "meta.tsv", write replace text
file write fh "_N" _tab "`=_N'" _n
foreach v of varlist _all {{
    local t : type `v'
    local f : format `v'
    local vl : value label `v'
    local lab : variable label `v'
    local lab = subinstr(`"`lab'"', char(9), " ", .)
    quietly count if missing(`v')
    local nmiss = r(N)
    local nint = 0
    if substr("`t'", 1, 3) != "str" {{
        quietly count if `v' != floor(`v') & !missing(`v')
        local nint = r(N)
    }}
    capture egen byte _tag = tag(`v')
    if _rc local nuniq = _N
    else {{
        quietly count if _tag & !missing(`v')
        local nuniq = r(N)
        drop _tag
    }}
    file write fh "`v'" _tab "`t'" _tab "`f'" _tab "`vl'" _tab "`nmiss'" _tab "`nint'" _tab "`nuniq'" _tab `"`lab'"' _n
}}
file close fh
capture label save using "labels.do", replace
file open ok using "done", write replace
file close ok
"""

NUM_DO = r"""quietly count if !missing({v})
local nm = r(N)
if `nm' > 0 {{
    quietly _pctile {v}, percentiles(1(1)99)
    forvalues k = 1/99 {{
        file write st "{v}" _tab "k`k'" _tab %21.0g (r(r`k')) _n
    }}
    quietly summarize {v}
    file write st "{v}" _tab "mean" _tab %21.0g (r(mean)) _n "{v}" _tab "sd" _tab %21.0g (r(sd)) _n
    quietly egen double _r = rank({v})
    quietly generate double _zl{i} = invnormal(_r / (`nm' + 1))
    drop _r
}}
else quietly generate double _zl{i} = .
"""

CAT_DO = r"""quietly count if !missing({v})
local nm = r(N)
quietly bysort {v}: generate long _c = _N if !missing({v})
quietly egen long _g = group({v}) if !missing({v}) & _c >= {min_count}
quietly summarize _g
local other = cond(missing(r(max)), 1, r(max) + 1)
quietly replace _g = `other' if !missing({v}) & _c < {min_count}
quietly bysort _g: generate long _cg = _N if !missing(_g)
quietly by _g: generate byte _f = _n == 1 & !missing(_g)
quietly generate double _p = _cg / `nm' if _f
quietly generate double _lo = sum(_p) - _p if _f
quietly by _g: replace _lo = _lo[1]
quietly by _g: replace _p = _p[1]
quietly generate double _zl{i} = invnormal(min(max(_lo + runiform() * _p, 1e-6), 1 - 1e-6)) if !missing(_g)
quietly generate byte _other = _g == `other' & _c < {min_count}
frame put {v} _cg _other if _f, into(lv)
frame lv {{
    capture replace {v} = . if _other
    capture replace {v} = "" if _other
    export delimited using "levels_{i}.csv", nolabel replace
}}
frame drop lv
drop _c _g _cg _f _p _lo _other
"""


def find_stata():
    found = [os.environ.get("STATA_BIN")] + [shutil.which(b) for b in ("stata-mp", "stata-se", "stata")]
    found += sorted(glob.glob("/Applications/Stata*/Stata*.app/Contents/MacOS/stata-*"))
    found += sorted(glob.glob("/usr/local/stata*/stata-*"))
    return next((b for b in found if b and os.access(b, os.X_OK)), None)


def run_do(stata, tmp, name, code):
    (tmp / f"{name}.do").write_text(code)
    (tmp / "done").unlink(missing_ok=True)
    subprocess.run([stata, "-q", "-b", "do", f"{name}.do"], cwd=tmp, capture_output=True)
    if not (tmp / "done").exists():  # batch Stata exits 0 on error; the sentinel is the real signal
        log = (tmp / f"{name}.log").read_text(errors="ignore") if (tmp / f"{name}.log").exists() else ""
        rc = re.findall(r"^r\((\d+)\);", log, re.M)
        raise SafeError(f"Stata {name} step failed with r({rc[-1] if rc else '?'}); "
                        f"run {name}.do yourself to see why")


def stata_kind(t, fmt, vl, nmiss, nint, nuniq, n, override):
    if override:
        return override
    if fmt.startswith(("%td", "%tc", "%tC", "%d")):
        return "datetime"
    if vl:
        return "categorical"
    if t.startswith("str"):
        return "id" if n > nmiss and nuniq > 0.5 * (n - nmiss) else "categorical"
    if nuniq <= 10:
        return "categorical"
    return "integer" if nint == 0 else "numeric"


def fit_stata(src, cfg, min_count, seed, stata):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run_do(stata, tmp, "meta", META_DO.format(src=src.resolve()))
        lines = (tmp / "meta.tsv").read_text().splitlines()
        n = int(lines[0].split("\t")[1])
        sets = {}
        if (tmp / "labels.do").exists():
            for name, code, text in re.findall(r'^label define (\S+)\s+(-?\d+)\s+`"(.*)"\'', (tmp / "labels.do").read_text(), re.M):
                sets.setdefault(name, {})[int(code)] = text
        meta = {"var_labels": {}, "value_labels": {}}
        cols, latent, do = {}, [], ["version 16", "set more off", f"set seed {int(seed)}",
                                    f'use `"{src.resolve()}"\', clear', 'file open st using "stats.tsv", write replace text']
        for line in lines[1:]:
            v, t, fmt, vl, nmiss, nint, nuniq, lab = (line.split("\t", 7) + [""])[:8]
            nmiss, nint, nuniq = int(nmiss), int(nint), int(nuniq)
            if lab:
                meta["var_labels"][v] = lab
            if vl in sets:
                meta["value_labels"][v] = sets[vl]
            c = cfg.get("columns", {}).get(v, {})
            if c.get("drop"):
                continue
            kind = stata_kind(t, fmt, vl, nmiss, nint, nuniq, n, c.get("type"))
            cols[v] = {"type": kind, "missing": nmiss / n if n else 0.0}
            if kind == "id":
                continue
            latent.append(v)
            i = len(latent)
            cols[v]["fmt"] = fmt
            do.append((NUM_DO if kind in ("numeric", "integer", "datetime") else CAT_DO).format(v=v, i=i, min_count=min_count))
        do += ["file close st"]
        if latent:
            do += [f"quietly pwcorr {' '.join(f'_zl{i}' for i in range(1, len(latent) + 1))}",
                   "matrix C = r(C)", 'file open cm using "corr.tsv", write replace text',
                   "forvalues a = 1/`=rowsof(C)' {", "forvalues b = 1/`=colsof(C)' {",
                   "file write cm %21.0g (C[`a',`b']) _tab", "}", "file write cm _n", "}", "file close cm"]
        do += ['file open ok using "done", write replace', "file close ok"]
        run_do(stata, tmp, "fit", "\n".join(do) + "\n")

        stats = pd.read_csv(tmp / "stats.tsv", sep="\t", header=None, names=["v", "k", "x"])
        for i, v in enumerate(latent, 1):
            m = cols[v]
            if m["type"] in ("numeric", "integer", "datetime"):
                st = stats[stats.v == v].set_index("k").x
                knots = np.array([st.get(f"k{k}", 0.0) for k in range(1, 100)], float)
                mean, sd = float(st.get("mean", 0.0)), float(st.get("sd", 1.0) or 1.0)
                if m["type"] == "datetime":  # Stata epoch 1960 -> pandas ns since 1970
                    daily = m["fmt"].startswith(("%td", "%d"))
                    unit, off = (86_400e9, 3653) if daily else (1e6, 3653 * 86_400_000)
                    knots, mean, sd = (knots - off) * unit, (mean - off) * unit, sd * unit
                    m["daily"] = daily
                m.update(knots=knots, mean=mean, sd=sd)
            else:
                lv = pd.read_csv(tmp / f"levels_{i}.csv", dtype={v: str}, keep_default_na=False)
                numeric = not lv.empty and pd.to_numeric(lv[v][lv._other == 0], errors="coerce").notna().all() \
                    and not cols[v]["fmt"].startswith("%") or False
                numeric = cols[v]["fmt"][-1:] != "s"
                counts = lv._cg.to_numpy(float)
                keys = [None if o else (float(x) if numeric else x) for x, o in zip(lv[v], lv._other)]
                other = None
                if (lv._other == 1).any():  # the merged band's value was blanked inside Stata
                    other = (max([k for k in keys if k is not None], default=0.0) + 1) if numeric else "Other"
                    keys = [other if k is None else k for k in keys]
                m.update(levels=keys or ["Other"], p=counts / counts.sum() if counts.sum() else np.array([1.0]),
                         numeric=numeric, other=other)
        corr = np.loadtxt(tmp / "corr.tsv", ndmin=2) if latent else np.zeros((0, 0))
    return cols, latent, corr, n, meta


def kept_labels(m, labs):
    """Value labels for the codes the synthetic column can take; rare levels' labels never leave."""
    keep = {int(k) for k in m.get("levels", []) if isinstance(k, float) and k % 1 == 0}
    out = {k: v for k, v in labs.items() if k in keep}
    if m.get("other") is not None:
        out[int(m["other"])] = "Other (rare levels merged)"
    return out


def write_dta(syn, path, cols, meta):
    value_labels = {c: kept_labels(cols[c], labs) for c, labs in meta["value_labels"].items()
                    if c in cols and cols[c].get("numeric")}
    syn.to_stata(path, write_index=False, version=118,
                 variable_labels={c: l[:80] for c, l in meta["var_labels"].items() if c in syn},
                 value_labels=value_labels or None,
                 convert_dates={c: ("td" if m.get("daily") else "tc") for c, m in cols.items() if m["type"] == "datetime"})


def report(syn, cols, cfg, src_name, md_path, json_path, meta):
    """Describe the SYNTHETIC data only."""
    vars_ = []
    for name, m in cols.items():
        s = syn[name]
        v = {"name": name, "label": meta["var_labels"].get(name, ""), "type": m["type"],
             "missing_pct": round(100 * float(s.isna().mean()), 1),
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
            labs = kept_labels(m, meta["value_labels"].get(name, {})) if m.get("numeric") else {}
            show = lambda k: f"{k:g} {labs[int(k)]}" if m.get("numeric") and float(k) % 1 == 0 and int(k) in labs else (
                f"{k:g}" if m.get("numeric") else str(k))
            v["levels"] = {show(k): round(100 * float(p), 1) for k, p in freq.items()}
            if labs:
                v["value_labels"] = {str(k): l for k, l in labs.items()}
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
             "| Variable | Label | Type | Missing % | Distribution | Controls |", "|---|---|---|---|---|---|"]
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
        lab = v["label"].replace("|", "/")
        lines.append(f"| `{v['name']}` | {lab} | {v['type']} | {v['missing_pct']} | {d} | "
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
    ap.add_argument("--engine", choices=["auto", "python", "stata"], default="auto",
                    help="auto = Stata for .dta files above --stata-above-mb, else Python")
    ap.add_argument("--stata-above-mb", type=float, default=250)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")  # warning text can quote data values; stderr stays clean

    src = Path(a.source)
    cfg = json.loads(Path(a.config).read_text()) if a.config else {}
    seed = a.seed if a.seed is not None else cfg.get("seed", 0)
    rng = np.random.default_rng(seed)
    ext = src.suffix.lower()
    engine = a.engine
    if engine == "auto":
        big = ext == ".dta" and src.exists() and src.stat().st_size / 1e6 > a.stata_above_mb
        engine = "stata" if big else "python"
    stata = find_stata() if engine == "stata" else None
    if engine == "stata" and not stata:
        if a.engine == "stata":
            sys.exit("generation failed: Stata not found (set STATA_BIN to the stata-mp/stata-se binary)")
        print("note: large .dta but Stata not found; fitting in Python (set STATA_BIN to use Stata)")
        engine = "python"
    if engine == "stata" and ext != ".dta":
        sys.exit("generation failed: --engine stata needs a .dta file")
    try:
        meta = {"var_labels": {}, "value_labels": {}}
        if engine == "stata":
            cols, names, base, n_real, meta = fit_stata(src, cfg, a.min_count, seed, stata)
        else:
            df = READERS[ext](src)
            if ext == ".dta":
                meta = dta_labels(src)
            for c in df.columns[df.dtypes == object]:  # dates stored as text (csv) -> datetime
                parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
                if df[c].notna().any() and parsed.notna().sum() == df[c].notna().sum():
                    df[c] = parsed
            fit_cfg = json.loads(json.dumps(cfg))  # copy: the report shows the user's controls only
            for c in meta["value_labels"]:  # labelled codes are categories, not numbers
                fit_cfg.setdefault("columns", {}).setdefault(c, {}).setdefault("type", "categorical")
            cols, names, base = fit(df, fit_cfg, a.min_count, rng)
            n_real = len(df)
        corr = correlation(base, names, cfg)
        n = a.rows or cfg.get("rows") or n_real
        syn = sample(cols, names, corr, n, cfg, rng, meta["value_labels"])
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        suffix = ext if ext in (".dta", ".parquet") else ".csv"
        data_path = out / f"{src.stem}_synthetic{suffix}"
        if suffix == ".dta":
            write_dta(syn, data_path, cols, meta)
        elif suffix == ".parquet":
            syn.to_parquet(data_path, index=False)
        else:
            syn.to_csv(data_path, index=False)
        report(syn, cols, {**cfg, "seed": seed, "engine": engine}, src.name,
               out / f"{src.stem}_report.md", out / f"{src.stem}_report.json", meta)
    except SafeError as e:
        sys.exit(f"generation failed: {e}")
    except Exception as e:  # never echo the message: it can quote real values
        sys.exit(f"generation failed: {type(e).__name__} (message suppressed; run it yourself to see details)")
    print(f"wrote {data_path} ({len(syn)} rows, {len(cols)} columns, engine: {engine})")
    print(f"wrote {out / f'{src.stem}_report.md'} and .json")


if __name__ == "__main__":
    main()
