"""Statistics for chapter 2: per-strategy summaries, Friedman test across the five strategies,
paired Wilcoxon tests and bootstrap CIs versus the rule-based EMS, start-cost and horizon sensitivity."""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats

import h2system as hs
from run_ch2 import DESIGN

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
d = pd.read_csv(os.path.join(RES, "ch2_runs.csv"))
d["backup_cost"] = d["short"] * hs.P["backup_price"]
d["opcost"] = d["backup_cost"] + d["wear"]
DEM = hs.P["demand_kg_day"] * 365
for name, x in DESIGN.items():
    fixed = float(hs.annual_fixed_cost(np.array([x]))[0])
    m = d.site == name
    d.loc[m, "lcoh"] = (fixed + d.loc[m, "opcost"] + d.loc[m, "prod"] * 15e-3 * hs.WATER_PRICE[name]) / DEM
d["rel"] = 1 - d["short"] / DEM
LAB = {("rule", "perfect"): "RB", ("mpc_energy", "perfect"): "MPC-E", ("mpc_deg", "perfect"): "MPC-D",
       ("mpc_deg", "persistence"): "MPC-D-P", ("mpc_deg", "noisy"): "MPC-D-N"}
d["code"] = [LAB.get((s, f), s) for s, f in zip(d.strategy, d.forecast)]
base = d[(d.start_h == 10.0) & (d.horizon == 48)]
out = {}

summ = base.groupby(["site", "code"]).agg(n=("year", "count"), prod=("prod", "mean"), short=("short", "mean"),
                                          rel=("rel", "mean"), starts=("starts", "mean"), hours=("hours", "mean"),
                                          load=("mean_load", "mean"), wear=("wear", "mean"), backup=("backup_cost", "mean"),
                                          opcost=("opcost", "mean"), opcost_sd=("opcost", "std"), lcoh=("lcoh", "mean"),
                                          curt=("curt", "mean")).reset_index()
summ.to_csv(os.path.join(RES, "ch2_summary.csv"), index=False)
out["summary"] = summ.to_dict(orient="records")

tests, fried = [], []
for name in DESIGN:
    w = base[base.site == name].pivot(index="year", columns="code", values="opcost").dropna()
    f = stats.friedmanchisquare(*[w[c] for c in w.columns])
    fried.append(dict(site=name, n_years=len(w), chi2=float(f.statistic), p=float(f.pvalue),
                      mean_rank={c: float(v) for c, v in w.rank(axis=1).mean().items()}))
    rng = np.random.default_rng(1)
    for metric in ["opcost", "wear", "backup_cost", "starts", "hours", "prod", "lcoh"]:
        p = base[base.site == name].pivot(index="year", columns="code", values=metric).dropna()
        for c in [k for k in ["MPC-E", "MPC-D", "MPC-D-P", "MPC-D-N"] if k in p.columns]:
            for ref in ["RB", "MPC-E"]:
                if ref == c or ref not in p.columns:
                    continue
                diff = (p[c] - p[ref]).values
                bs = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
                wt = stats.wilcoxon(p[c], p[ref]) if np.any(diff != 0) else None
                tests.append(dict(site=name, metric=metric, strategy=c, vs=ref, n=len(diff), mean_ref=float(p[ref].mean()),
                                  mean_diff=float(diff.mean()), pct=float(diff.mean() / p[ref].mean() * 100),
                                  ci_lo=float(np.quantile(bs, 0.025)), ci_hi=float(np.quantile(bs, 0.975)),
                                  wilcoxon_p=float(wt.pvalue) if wt else 1.0, dz=float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0,
                                  years_better=int((diff < 0).sum())))
pd.DataFrame(tests).to_csv(os.path.join(RES, "ch2_tests.csv"), index=False)
out["friedman"] = fried

# start-cost sensitivity: RB vs MPC-D (perfect) under k_s = 0, 10, 30
sens = []
for name in DESIGN:
    for sh in [0.0, 10.0, 30.0]:
        g = d[(d.site == name) & (d.start_h == sh) & (d.horizon == 48) & (d.forecast == "perfect")]
        p = g.pivot(index="year", columns="code", values="opcost").dropna()
        st = g.pivot(index="year", columns="code", values="starts").dropna()
        if {"RB", "MPC-D"} <= set(p.columns):
            diff = p["MPC-D"] - p["RB"]
            sens.append(dict(site=name, start_h=sh, rb_opcost=float(p["RB"].mean()), mpcd_opcost=float(p["MPC-D"].mean()),
                             saving=float(-diff.mean()), saving_pct=float(-diff.mean() / p["RB"].mean() * 100),
                             rb_starts=float(st["RB"].mean()), mpcd_starts=float(st["MPC-D"].mean()),
                             wilcoxon_p=float(stats.wilcoxon(p["MPC-D"], p["RB"]).pvalue), n=len(p)))
pd.DataFrame(sens).to_csv(os.path.join(RES, "ch2_startcost.csv"), index=False)
out["startcost"] = sens

# horizon study (site A, years 0-4)
hz = d[(d.site == "A_semiarid_CO") & (d.code == "MPC-D") & (d.start_h == 10.0) & (d.year < 5)]
hzs = hz.groupby("horizon").agg(opcost=("opcost", "mean"), wear=("wear", "mean"), short=("short", "mean"),
                                starts=("starts", "mean"), n=("year", "count")).reset_index()
p = hz.pivot(index="year", columns="horizon", values="opcost").dropna()
out["horizon"] = hzs.to_dict(orient="records")
out["horizon_friedman_p"] = float(stats.friedmanchisquare(*[p[c] for c in p.columns]).pvalue) if p.shape[1] >= 3 else None

with open(os.path.join(RES, "ch2_stats.json"), "w") as fh:
    json.dump(out, fh, indent=1, default=float)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(summ.round(1).to_string())
print(pd.DataFrame(fried).to_string())
print(pd.DataFrame(tests)[lambda t: t.metric.isin(["opcost", "wear", "starts", "backup_cost", "lcoh", "prod", "hours"])].round(4).to_string())
print(pd.DataFrame(sens).round(3).to_string())
print(hzs.round(1).to_string(), out["horizon_friedman_p"])
