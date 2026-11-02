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
