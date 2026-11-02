#!/usr/bin/env python3
"""Self-check for generate.py and restricted-data-guard.py on fake data. Run: python3 check_generate.py"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GUARD = HERE.parents[1] / "hooks" / "restricted-data-guard.py"

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    rng = np.random.default_rng(1)
    n = 2000
    age = rng.integers(18, 80, n)
    real = pd.DataFrame({
        "pid": [f"P{i}" for i in range(n)], "age": age,
        "income": np.exp(9 + 0.02 * age + rng.normal(0, .5, n)),
        "region": rng.choice(["N", "S", "E", "W", "Tiny"], n, p=[.4, .3, .2, .098, .002]),
        "date": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), "D")})
    real.loc[rng.uniform(size=n) < .1, "income"] = np.nan
    real.to_csv(d / "real.csv", index=False)
    real.to_stata(d / "real.dta", write_index=False)
    (d / "c.json").write_text(json.dumps({"rows": 3000, "correlations": {"age,income": 0.9},
                                          "columns": {"income": {"shift": 1}, "age": {"missing_rate": .2}}}))

    for src, extra in [("real.csv", []), ("real.dta", ["--config", str(d / "c.json")])]:
        subprocess.run([sys.executable, HERE / "generate.py", d / src, "--out", d / src[-3:], *extra], check=True)

    s = pd.read_csv(d / "csv" / "real_synthetic.csv")
    assert list(s.columns) == list(real.columns) and len(s) == n
    assert s.income.max() <= real.income.max(), "tails must be clipped"
    assert "Tiny" not in set(s.region), "rare level must merge into Other"
    assert not set(s.pid) & set(real.pid), "IDs must be fake"
    assert abs(s.income.isna().mean() - .1) < .03
    assert abs(s.age.corr(s.income, method="spearman") - real.age.corr(real.income, method="spearman")) < .08
    assert (d / "csv" / "real_report.md").exists() and (d / "csv" / "real_report.json").exists()

    s2 = pd.read_stata(d / "dta" / "real_synthetic.dta")
    assert len(s2) == 3000 and abs(s2.age.isna().mean() - .2) < .03
    assert s2.age.corr(s2.income, method="spearman") > .8, "forced correlation"
    assert s2.income.mean() > real.income.mean() * 1.2, "shift"

    bad = subprocess.run([sys.executable, HERE / "generate.py", d / "missing.csv"], capture_output=True, text=True)
    assert bad.returncode and "missing.csv" not in bad.stderr, "errors must not echo details"

    # labels: variable + value labels survive into the report and the synthetic .dta, both engines
    real["region"] = rng.choice([1, 2, 3, 9], n, p=[.5, .3, .198, .002]).astype("int8")
    real.to_stata(d / "lab.dta", write_index=False, version=118,
                  variable_labels={"age": "Age in years", "region": "Region of residence"},
                  value_labels={"region": {1: "North", 2: "South", 3: "East", 9: "Islands"}})
    (d / "w.json").write_text(json.dumps({"columns": {"region": {"weights": {"North": 0.8, "South": 0.2}}}}))
    sys.path.insert(0, str(HERE))
    import generate
    engines = ["python"] + (["stata"] if generate.find_stata() else [])
    for eng in engines:
        flags = ["--engine", "auto", "--stata-above-mb", "0"] if eng == "stata" else ["--engine", "python"]
        r = subprocess.run([sys.executable, HERE / "generate.py", d / "lab.dta", "--out", d / eng, *flags],
                           check=True, capture_output=True, text=True)
        assert f"engine: {eng}" in r.stdout
        with pd.io.stata.StataReader(d / eng / "lab_synthetic.dta") as rd:
            assert rd.variable_labels()["age"] == "Age in years"
            labs = rd.value_labels()["region"]
        assert labs[1] == "North" and "Islands" not in labs.values(), "rare level's label must not leak"
        assert "Region of residence" in (d / eng / "lab_report.md").read_text()
        syn = pd.read_stata(d / eng / "lab_synthetic.dta", convert_categoricals=False)
        assert set(syn.region.dropna()) <= {1, 2, 3, 4}, "labelled codes stay codes"
        subprocess.run([sys.executable, HERE / "generate.py", d / "lab.dta", "--out", d / f"{eng}w",
                        "--config", d / "w.json", *flags], check=True, capture_output=True)
        syn = pd.read_stata(d / f"{eng}w" / "lab_synthetic.dta", convert_categoricals=False)
        assert abs((syn.region == 1).mean() - .8) < .04, "weights keyed by label text"
    if "stata" in engines:
        (d / "broken.dta").write_bytes(b"not a stata file")
        r = subprocess.run([sys.executable, HERE / "generate.py", d / "broken.dta", "--engine", "stata"],
                           capture_output=True, text=True)
        assert r.returncode and "Stata meta step failed" in r.stderr
    print(f"labels checked on engines: {', '.join(engines)}")


def guard(event):
    out = subprocess.run([sys.executable, GUARD], input=json.dumps(event), capture_output=True, text=True).stdout
    return "deny" in out


bash = lambda c: {"tool_name": "Bash", "tool_input": {"command": c}}
assert guard({"tool_name": "Read", "tool_input": {"file_path": "data/restricted/a.csv"}})
assert not guard({"tool_name": "Read", "tool_input": {"file_path": "data/synthetic/a.csv"}})
assert guard(bash("head data/restricted/a.csv"))
assert guard(bash("python3 .claude/skills/synthetic-data/generate.py data/restricted/a.csv; cat data/restricted/a.csv"))
assert not guard(bash("python3 .claude/skills/synthetic-data/generate.py data/restricted/a.csv --rows 10"))
assert not guard({"tool_name": "Write", "tool_input": {"file_path": "x.md", "content": "data/restricted/"}})
print("synthetic-data: all checks passed")
