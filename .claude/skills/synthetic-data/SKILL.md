---
name: synthetic-data
description: Build a synthetic stand-in for a restricted or confidential dataset without Claude ever reading the real file. A local script fits a Gaussian copula (marginals plus rank correlations) and writes a synthetic file and a variable/distribution report computed on the synthetic data only. Controls can shift, rescale, skew, re-weight, add missingness, weaken or force correlations. Use when the user says "synthetic data", "fake version of my data", "make a dummy dataset", "I can't show you the data", "work on restricted data", "skew the data", or wants Claude to write analysis code for data it must not see.
argument-hint: "<path to real file under data/restricted/> [--config controls.json] [--rows N] [--seed S]"
allowed-tools: ["Bash", "Read", "Write"]
---

# `/synthetic-data` — work on data Claude never sees

**Core principle:** Claude writes and runs code *on* the real data; it never reads the data.
The real file stays in `data/restricted/` (or any `confidential/` folder). Claude reads only
`data/synthetic/`, which holds the synthetic copy and its report.

Enforced by [`restricted-data-guard.py`](../../hooks/restricted-data-guard.py) plus `Read`/`Edit`
deny rules in `.claude/settings.json`. Background: [`confidential-data.md`](../../rules/confidential-data.md).

## Hard rules for Claude

1. **Never** open, `cat`, `head`, `grep`, load in a REPL, or print anything from the real file.
   No "just the first row", no "just the column names". The report carries the column names.
2. The **only** command that touches the real file is the generator below, run as one plain
   command (no pipes, `;`, `&&`, redirection). The guard denies anything else.
3. If generation fails, the script prints only the exception *type*. Do not try to debug by
   reading the data: ask the user to run the command themselves and describe the problem
   in their own words, without pasting values.
4. Analysis code written against the synthetic file must take the path as an argument, so
   the user can rerun it on the real file outside Claude.

## Workflow

1. **Locate the file.** Ask the user for the filename, or `ls data/restricted/`.
   Supported: `.csv .tsv .dta .parquet .xlsx .xls .sav`.
2. **Agree the controls** (optional). Write `data/synthetic/controls.json` if the user wants
   anything other than a faithful twin. Format below.
3. **Generate:**
   ```bash
   python3 .claude/skills/synthetic-data/generate.py data/restricted/<file> --config data/synthetic/controls.json
   ```
4. **Read the report** `data/synthetic/<stem>_report.md` (and `.json` for machine use). It lists
   every variable with type, missing %, and distribution, plus the Spearman correlation matrix,
   all computed on the synthetic rows.
5. **Work** on `data/synthetic/<stem>_synthetic.<ext>`. Hand the user the script and the one
   command to run it on the real file.

## Controls (`controls.json`, every key optional)

```json
{
  "rows": 5000,
  "seed": 1,
  "correlation": 1.0,
  "correlations": {"age,income": 0.6},
  "columns": {
    "income": {"shift": 0.5, "scale": 1.2, "skew": 1.0, "missing_rate": 0.05},
    "region": {"weights": {"North": 0.7, "South": 0.3}},
    "ssn":    {"drop": true},
    "hh_id":  {"type": "id"}
  }
}
```

| Key | Effect |
|---|---|
| `rows`, `seed` | output size (default: same as real) and RNG seed (default 0) |
| `correlation` | 0 = all columns independent, 1 = as estimated, in between = shrink toward 0 |
| `correlations` | force a latent correlation for a pair; the matrix is repaired to stay valid |
| `shift` | add N standard deviations to a numeric column |
| `scale` | multiply its spread around the mean |
| `skew` | `>0` heavier right tail, `<0` heavier left tail; on a categorical it tilts toward later/earlier levels |
| `weights` | replace a categorical's level shares (new levels allowed) |
| `missing_rate` | replace the column's missing share |
| `drop` | leave the column out |
| `type` | force `numeric`, `integer`, `categorical`, `datetime` or `id` |

For a variable with Stata value labels, `weights` keys can be the label text (`"North"`) or
the code (`"1"`).

`weights` or `skew` on a categorical column samples it independently, so its correlation with
other columns is lost. Say so to the user if they ask for both.

Command-line `--rows` / `--seed` override the file. `--min-count N` (default 10, the usual
disclosure threshold) merges rarer categories into `Other`.

## Labels and large Stata files

- **Labels.** For `.dta` input, variable labels and value labels appear in the report (a
  `Label` column, and `1 North 40%` style level shares). The synthetic `.dta` keeps them, and
  labelled variables stay numeric codes, so do-files written against it run unchanged on the
  real file. Dates keep `%td` / `%tc`.
- **Stata engine.** A `.dta` larger than `--stata-above-mb` (default 250) is fitted by Stata in
  batch mode: Stata reads the data and writes only fitted summaries (percentile knots, level
  counts, latent correlations) to a temp folder that is deleted afterwards; Python samples from
  those, so every control above works the same. Force it with `--engine stata`, or stay in
  Python with `--engine python`. Stata is found via `$STATA_BIN`, `PATH`, or `/Applications`.
  The gain is memory, not speed: on 2M rows both engines take about 20 seconds, but Python
  has to hold the whole file in RAM. If Stata fails, the script reports only its `r()` code;
  ask the user to run the generated do-file themselves, never read its log.

## Built-in privacy measures

- Numeric values come from interpolated percentiles, clipped to the real p1–p99 range, so
  exact extremes never appear. Integer columns are rounded, so common values will recur.
- Rare categories (fewer than `--min-count` rows) become `Other`; for labelled codes, `Other`
  gets a new code and the rare levels' labels are dropped from the synthetic label set.
- Near-unique text columns (IDs, names, free text) become fake `ID_0000001` strings.
- Columns, level names and approximate shares *are* carried over: that is the point.
  If a column name or a category label is itself sensitive, `drop` it or rename it first.

## What this skill does NOT do

- It is **not** formal privacy (no differential-privacy guarantee). A synthetic twin of a
  small dataset can still leak; treat its outputs under the same disclosure rules
  ([`/disclosure-check`](../disclosure-check/SKILL.md)) before sharing them.
- Results on synthetic data are for building and debugging code. Estimates, p-values and
  effect sizes come from the real run, done by the user.
- It reads labels from `.dta` only. SPSS `.sav` labels are dropped (pandas does not keep them).
- It keeps pairwise rank dependence only. Nonlinear relations, panel structure, and
  within-group ordering are not reproduced.
- The guard does not catch a command that reads files without naming them (`grep -r .`).
  The rules above are the main protection; the hook is the backstop.
