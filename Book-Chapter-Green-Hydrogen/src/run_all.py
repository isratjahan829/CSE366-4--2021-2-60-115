"""End-to-end experiment: train CVAE scenario generator, score it, then size the hydrogen
system with a GA under deterministic, risk-neutral and risk-averse formulations.
Writes ../results/results.json, ../results/*.csv and ../figures/*.png"""
import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_prep
import scenarios as sc
import h2system as hs
from ga import run_ga, random_search

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results"); FIG = os.path.join(HERE, "..", "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(2026)
R = {}

# ----------------------------------------------------------------- 1. data + CVAE
df = data_prep.build()
sites = [sc.SiteData(*data_prep.daily_arrays(df, s), i) for i, s in enumerate(sc.SITES)]
train, test = sc.split_days()
t0 = time.time()
model, hist = sc.train_cvae(sites, train, epochs=3000)
R["train_seconds"] = round(time.time() - t0, 1)
R["train_days"], R["test_days"] = int(len(train) * 2), int(len(test) * 2)
pd.DataFrame(hist, columns=["epoch", "recon", "kl"]).to_csv(os.path.join(RES, "cvae_training.csv"), index=False)

scores = []
for s, name in zip(sites, sc.SITES):
    gpar = sc.fit_gaussian(s, train)
    for meth, (kt, ta) in {
        "CVAE": sc.sample_cvae(model, s, test, 100, rng),
        "Gaussian (monthly MVN)": sc.sample_gaussian(s, gpar, test, 100, rng),
        "Bootstrap (+/-15 d)": sc.sample_bootstrap(s, train, test, 100, rng),
    }.items():
        scores.append(dict(site=name, method=meth, **sc.evaluate(s, test, kt, ta)))
    clim = np.array([s.ghi[train].sum(1).mean() / 1000.0] * len(test))
    scores.append(dict(site=name, method="Climatology (point)",
                       ES=float(np.mean([np.linalg.norm(s.ghi[train].mean(0) / 1000 - s.ghi[d] / 1000) for d in test])),
                       CRPS=float(np.mean(np.abs(clim - s.ghi[test].sum(1) / 1000.0))), PICP90=0.0))
R["scores"] = scores
pd.DataFrame(scores).to_csv(os.path.join(RES, "scenario_scores.csv"), index=False)
print(pd.DataFrame(scores))

# annual-insolation spread of synthetic years and latent persistence
R["rho"] = {}
YEARS = {}
for s, name in zip(sites, sc.SITES):
    rho = sc.latent_rho(model, s)
    R["rho"][name] = rho
    g_in, t_in = sc.year_scenarios(model, s, 30, rng, rho)          # in-sample (optimisation)
    g_out, t_out = sc.year_scenarios(model, s, 200, rng, rho)       # out-of-sample (CVAE)
    g_bs, t_bs = sc.bootstrap_years(s, 200, rng)                    # out-of-sample (bootstrap)
    g_hist, t_hist = s.ghi.reshape(1, -1), s.ta.reshape(1, -1)
    geo = hs.geometry(data_prep.SITES[name])
    YEARS[name] = dict(insample=hs.pv_per_kwp(g_in, t_in, geo), cvae=hs.pv_per_kwp(g_out, t_out, geo),
                       boot=hs.pv_per_kwp(g_bs, t_bs, geo), hist=hs.pv_per_kwp(g_hist, t_hist, geo))
    R.setdefault("pv_yield_kwh_kwp", {})[name] = dict(hist=float(YEARS[name]["hist"].sum()),
                                                       hist_horizontal=float(hs.pv_per_kwp(g_hist, t_hist).sum()),
                                                       cvae_mean=float(YEARS[name]["cvae"].sum(1).mean()))
    ann = g_out.sum(1) / 1000.0
    # longest run of consecutive days below 50 % of the clear-sky energy ("dunkelflaute" proxy)
    def longest_dark(g):
        daily = g.reshape(len(g), 365, 24).sum(2) / s.cs.sum(1)[None]
        runs = []
        for row in daily < 0.5:
            best = cur = 0
            for v in row:
                cur = cur + 1 if v else 0; best = max(best, cur)
            runs.append(best)
        return np.array(runs)
    R.setdefault("years", {})[name] = dict(
        hist_kwh_m2=float(s.ghi.sum() / 1000), cvae_mean=float(ann.mean()), cvae_p5=float(np.quantile(ann, 0.05)),
        cvae_p95=float(np.quantile(ann, 0.95)), hist_dark_run=int(longest_dark(g_hist)[0]),
        cvae_dark_run_mean=float(longest_dark(g_out).mean()), cvae_dark_run_p95=float(np.quantile(longest_dark(g_out), 0.95)),
        boot_dark_run_mean=float(longest_dark(g_bs).mean()))
print(R["years"], R["rho"])

# ----------------------------------------------------------------- 2. GA sizing
LO, HI = [100, 50, 0, 10], [4000, 2000, 4000, 3000]   # PV kWp, EL kW, battery kWh, tank kg
REL_MIN = 0.95


def make_fitness(pv1, water, mode):
    def f(X):
        c = hs.costs(X, hs.simulate(X, pv1), water)
        if mode == "risk_averse":
            obj = 0.5 * c["lcoh"].mean(1) + 0.5 * hs.cvar(c["lcoh"], 0.9)
        else:
            obj = c["lcoh"].mean(1)
        rel_ok = (c["reliability"] >= REL_MIN).mean(1)              # chance constraint P(rel>=95%)>=0.9
        return obj + 20.0 * np.maximum(0.9 - rel_ok, 0.0)
    return f


def summarise(x, pv1, water):
    X = np.asarray(x)[None]
    sim = hs.simulate(X, pv1)
    c = hs.costs(X, sim, water)
    l = c["lcoh"][0]
    return dict(lcoh_mean=float(l.mean()), lcoh_p95=float(np.quantile(l, 0.95)), lcoh_cvar90=float(hs.cvar(c["lcoh"], 0.9)[0]),
                rel_mean=float(c["reliability"][0].mean()), rel_p5=float(np.quantile(c["reliability"][0], 0.05)),
                p_rel95=float((c["reliability"][0] >= REL_MIN).mean()),
                curtail_pct=float((sim["curt"][0] / (X[0, 0] * pv1.sum(1))).mean() * 100),
                el_cf=float((sim["prod"][0] * 56.5 / (X[0, 1] * 8760)).mean() * 100),
                el_hours=float(sim["hours"][0].mean()))


designs, conv = [], {}
for name in sc.SITES:
    Y, water = YEARS[name], hs.WATER_PRICE[name]
    for mode, pv1 in [("deterministic", Y["hist"]), ("risk_neutral", Y["insample"]), ("risk_averse", Y["insample"])]:
        t0 = time.time()
        f = make_fitness(pv1, water, "risk_averse" if mode == "risk_averse" else "mean")
        if mode == "deterministic":  # one year: the chance constraint collapses to rel>=95 % in that year
            f = make_fitness(pv1, water, "mean")
        x, fx, h = run_ga(f, LO, HI, pop=40, gens=60, seed=11, log=print)
        conv[f"{name}|{mode}|GA"] = h.tolist()
        row = dict(site=name, formulation=mode, PV_kWp=x[0], EL_kW=x[1], BATT_kWh=x[2], TANK_kg=x[3],
                   capex_musd=float((x[0] * hs.P["pv_capex"] + x[1] * hs.P["el_capex"] + x[2] * hs.P["b_capex"] + x[3] * hs.P["t_capex"]) / 1e6),
                   ga_seconds=round(time.time() - t0, 1))
        for ev in ["cvae", "boot", "hist"]:
            for k, v in summarise(x, Y[ev], water).items():
                row[f"{ev}_{k}"] = v
        designs.append(row)
        print(row)
    # optimiser benchmark on the risk-neutral problem: GA vs random search, same evaluation budget
    f = make_fitness(Y["insample"], water, "mean")
    for seed in [1, 2, 3]:
        _, _, hg = run_ga(f, LO, HI, pop=40, gens=60, seed=100 + seed)
        _, _, hr = random_search(f, LO, HI, n_eval=40 + 60 * 38, seed=200 + seed)
        conv[f"{name}|seed{seed}|GA"] = hg.tolist(); conv[f"{name}|seed{seed}|RS"] = hr.tolist()

R["designs"] = designs
R["convergence"] = conv
pd.DataFrame(designs).to_csv(os.path.join(RES, "designs.csv"), index=False)

# ----------------------------------------------------------------- 3. sensitivity (risk-neutral design, site A)
sens = []
base = hs.P.copy()
for key, lab in [("el_capex", "Electrolyzer CAPEX"), ("pv_capex", "PV CAPEX"), ("backup_price", "Backup H2 price"),
                 ("b_capex", "Battery CAPEX"), ("rate", "Discount rate")]:
    for fac in [0.7, 1.3]:
        p = base.copy(); p[key] = base[key] * fac
        for name in sc.SITES:
            d = [r for r in designs if r["site"] == name and r["formulation"] == "risk_neutral"][0]
            X = np.array([[d["PV_kWp"], d["EL_kW"], d["BATT_kWh"], d["TANK_kg"]]])
            c = hs.costs(X, hs.simulate(X, YEARS[name]["cvae"], p), hs.WATER_PRICE[name], p)
            sens.append(dict(site=name, param=lab, factor=fac, lcoh_mean=float(c["lcoh"].mean())))
R["sensitivity"] = sens
pd.DataFrame(sens).to_csv(os.path.join(RES, "sensitivity.csv"), index=False)

# ----------------------------------------------------------------- 4. figures (PNG for the manuscript)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for i, s in enumerate(sites):
    d = test[20]
    kt, ta = sc.sample_cvae(model, s, [d], 30, rng)
    for k in kt[0]:
        ax[i].plot(k * s.cs[d], color="0.7", lw=0.6)
    ax[i].plot(s.ghi[d], color="C0", lw=2, label="observed (held-out)")
    ax[i].plot(s.cs[d], "k--", lw=1, label="clear-sky")
    ax[i].set_title(f"{sc.SITES[i]} - day {d}"); ax[i].set_xlabel("hour"); ax[i].set_ylabel("GHI [W/m$^2$]")
ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_scenarios.png"), dpi=200)

fig, ax = plt.subplots(figsize=(5, 3.2))
for name in sc.SITES:
    for seed in [1, 2, 3]:
        ax.plot(np.arange(61) * 38 + 40, conv[f"{name}|seed{seed}|GA"], color="C0" if name.startswith("A") else "C1", lw=1)
        ax.plot((np.arange(len(conv[f"{name}|seed{seed}|RS"])) + 1) * 40, conv[f"{name}|seed{seed}|RS"], "--",
                color="C0" if name.startswith("A") else "C1", lw=1)
ax.set_xlabel("fitness evaluations"); ax.set_ylabel("objective [USD/kg]"); ax.set_title("GA (solid) vs random search (dashed)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_convergence.png"), dpi=200)

# one-week dispatch trace of the risk-neutral design at site A (first CVAE year, July)
d = [r for r in designs if r["site"] == sc.SITES[0] and r["formulation"] == "risk_neutral"][0]
X = np.array([[d["PV_kWp"], d["EL_kW"], d["BATT_kWh"], d["TANK_kg"]]])
tr = hs.simulate(X, YEARS[sc.SITES[0]]["cvae"][:1], trace=True)["trace"]
w = slice(181 * 24, 188 * 24)
fig, ax = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True)
ax[0].plot(tr["pv"][w], label="PV [kW]"); ax[0].plot(tr["el"][w], label="electrolyzer [kW]"); ax[0].legend(fontsize=8)
ax[1].plot(tr["soc"][w] / max(d["BATT_kWh"], 1) * 100, label="battery SOC [%]")
ax[1].plot(tr["tank"][w] / d["TANK_kg"] * 100, label="H$_2$ tank [%]"); ax[1].legend(fontsize=8); ax[1].set_xlabel("hour of week")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_dispatch.png"), dpi=200)
R["dispatch_week"] = {k: np.round(v[w], 2).tolist() for k, v in tr.items()}

with open(os.path.join(RES, "results.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=float)
print("done")
