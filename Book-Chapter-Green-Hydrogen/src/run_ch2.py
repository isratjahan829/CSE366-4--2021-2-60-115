"""Chapter 2 experiment: rule-based vs MPC energy management (perfect / persistence / noisy forecasts),
start-cost sensitivity and horizon study. Weather: observed year + 9 seasonal moving-block-bootstrap years
(5-day blocks, +/-15-day window) per site. Writes ../results/ch2_runs.csv"""
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool

import data_prep as dp
import scenarios as sc
import h2system as hs
import mpc

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DESIGN = {"A_semiarid_CO": (1780.0, 906.0, 600.0, 224.0), "B_humid_NC": (2061.0, 866.0, 600.0, 257.0)}
NYEARS = 10


def block_bootstrap(site, rng, block=5, window=15):
    kt, ta = np.zeros((365, 24)), np.zeros((365, 24))
    days = np.arange(365)
    for d0 in range(0, 365, block):
        lo, hi = max(d0 - window, 0), min(d0 + window, 365 - block)
        s = rng.integers(lo, hi + 1)
        n = min(block, 365 - d0)
        kt[d0:d0 + n] = site.kt[s:s + n]
        ta[d0:d0 + n] = site.ta[s:s + n]
    _ = days
    return kt * site.cs, ta


def build_years():
    df = dp.build()
    out = {}
    for i, name in enumerate(sc.SITES):
        s = sc.SiteData(*dp.daily_arrays(df, name), i)
        geo = hs.geometry(dp.SITES[name])
        rng = np.random.default_rng(100 + i)
        yrs = [(s.ghi, s.ta)] + [block_bootstrap(s, rng) for _ in range(NYEARS - 1)]
        pv = [hs.pv_per_kwp(g.reshape(-1), t.reshape(-1), geo) for g, t in yrs]
        cs = [hs.pv_per_kwp(s.cs.reshape(-1), t.reshape(-1), geo) for g, t in yrs]
        out[name] = (pv, cs)
    return out


YEARS = None


def job(args):
    name, yr, strategy, fckind, start_h, horizon = args
    pv1, cs1 = YEARS[name][0][yr], YEARS[name][1][yr]
    x = DESIGN[name]
    deg = mpc.deg_params(x[1], start_hours=start_h)
    fc = None
    if fckind != "perfect":
        pers, noisy = mpc.make_forecasts(pv1, cs1, np.random.default_rng(1000 + yr))
        fc = pers if fckind == "persistence" else noisy
    r = mpc.run_year(pv1, x, strategy, fc=fc, horizon=horizon, deg=deg)
    return dict(site=name, year=yr, strategy=strategy, forecast=fckind, start_h=start_h, horizon=horizon, **r)


def init():
    global YEARS
    YEARS = build_years()


if __name__ == "__main__":
    jobs = []
    for name in sc.SITES:
        for yr in range(NYEARS):
            jobs += [(name, yr, "rule", "perfect", 10.0, 48), (name, yr, "mpc_energy", "perfect", 10.0, 48),
                     (name, yr, "mpc_deg", "perfect", 10.0, 48), (name, yr, "mpc_deg", "persistence", 10.0, 48),
                     (name, yr, "mpc_deg", "noisy", 10.0, 48)]
            for sh in [0.0, 30.0]:
                jobs += [(name, yr, "rule", "perfect", sh, 48), (name, yr, "mpc_deg", "perfect", sh, 48)]
    for yr in range(5):
        for hz in [24, 72]:
            jobs.append(("A_semiarid_CO", yr, "mpc_deg", "perfect", 10.0, hz))
    jobs.sort(key=lambda j: j[2] == "rule")  # slow jobs first
    with Pool(4, initializer=init) as pool:
        rows = []
        for k, r in enumerate(pool.imap_unordered(job, jobs)):
            rows.append(r)
            if k % 10 == 0:
                print(k, len(jobs), flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(RES, "ch2_runs.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "ch2_runs.csv"), index=False)
    print("done")
