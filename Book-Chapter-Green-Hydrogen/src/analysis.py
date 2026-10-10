"""Deep statistical analysis for the chapter (run after run_all.py).

A  descriptive statistics of the two solar datasets
B  statistical tests of the scenario generator (paired Wilcoxon, bootstrap CIs, PIT/chi-square,
   KS / Wasserstein, hourly correlation structure, day-to-day persistence)
C  Monte Carlo statistics of the designs (CIs, Wilson intervals, McNemar, paired Wilcoxon, monthly shortfall)
D  global economic sensitivity (Latin hypercube, Spearman, standardised regression coefficients) and
   weather-vs-economic variance decomposition
E  cost-reliability trade-off (GA re-run for several reliability targets)
F  GA seed robustness (5 seeds)
G  sample-average-approximation (scenario count) convergence
Writes ../results/analysis.json and ../results/analysis_*.csv
"""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats

import data_prep
import scenarios as sc
import h2system as hs
from ga import run_ga

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
A = {}

# ------------------------------------------------ rebuild stage 1 exactly as run_all.py (same seeds)
rng = np.random.default_rng(2026)
df = data_prep.build()
sites = [sc.SiteData(*data_prep.daily_arrays(df, s), i) for i, s in enumerate(sc.SITES)]
train, test = sc.split_days()
model, _ = sc.train_cvae(sites, train, epochs=3000)
samples = {}
for s, name in zip(sites, sc.SITES):
    gpar = sc.fit_gaussian(s, train)
    samples[name] = {
        "CVAE": sc.sample_cvae(model, s, test, 100, rng),
        "Gaussian": sc.sample_gaussian(s, gpar, test, 100, rng),
        "Bootstrap": sc.sample_bootstrap(s, train, test, 100, rng),
    }
YEARS, RAW = {}, {}
for s, name in zip(sites, sc.SITES):
    rho = sc.latent_rho(model, s)
    g_in, t_in = sc.year_scenarios(model, s, 30, rng, rho)
    g_out, t_out = sc.year_scenarios(model, s, 200, rng, rho)
    g_bs, t_bs = sc.bootstrap_years(s, 200, rng)
    geo = hs.geometry(data_prep.SITES[name])
    YEARS[name] = dict(insample=hs.pv_per_kwp(g_in, t_in, geo), cvae=hs.pv_per_kwp(g_out, t_out, geo),
                       boot=hs.pv_per_kwp(g_bs, t_bs, geo), hist=hs.pv_per_kwp(s.ghi.reshape(1, -1), s.ta.reshape(1, -1), geo))
    RAW[name] = dict(cvae=g_out, boot=g_bs)
designs = pd.read_csv(os.path.join(RES, "designs.csv"))


def xvec(site, form):
    d = designs[(designs.site == site) & (designs.formulation == form)].iloc[0]
    return np.array([[d.PV_kWp, d.EL_kW, d.BATT_kWh, d.TANK_kg]])


def daily(g):  # (..., 8760) W/m2 -> (..., 365) kWh/m2
    return g.reshape(*g.shape[:-1], 365, 24).sum(-1) / 1000.0


def lag1(x):
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def deseason(e):
    k = 31
    pad = np.concatenate([e[-k:], e, e[:k]])
    clim = np.convolve(pad, np.ones(k) / k, mode="same")[k:-k]
    return e - clim


# ------------------------------------------------ A. descriptive statistics
desc, monthly = [], []
for s, name in zip(sites, sc.SITES):
    e = s.ghi.sum(1) / 1000.0
    kd = e / (s.cs.sum(1) / 1000.0)
    desc.append(dict(site=name, annual_kwh_m2=e.sum(), daily_mean=e.mean(), daily_sd=e.std(ddof=1),
                     daily_cv=e.std(ddof=1) / e.mean(), daily_min=e.min(), daily_max=e.max(),
                     skew=stats.skew(e), kurt=stats.kurtosis(e), kd_mean=kd.mean(),
                     clear_days=int((kd >= 0.7).sum()), cloudy_days=int((kd < 0.4).sum()),
                     lag1_anomaly=lag1(deseason(e)), ta_mean=s.ta.mean(), ta_min=s.ta.min(), ta_max=s.ta.max(),
                     corr_e_ta_anom=float(np.corrcoef(deseason(e), deseason(s.ta.mean(1)))[0, 1])))
    for m in range(12):
        dd = np.arange(365)[np.minimum(np.arange(365) * 12 // 365, 11) == m]
        monthly.append(dict(site=name, month=m + 1, ghi_kwh_m2=float(e[dd].sum()), kd_mean=float(kd[dd].mean()),
                            ta_mean=float(s.ta[dd].mean())))
A["descriptive"] = desc
A["monthly"] = monthly
pd.DataFrame(desc).to_csv(os.path.join(RES, "analysis_descriptive.csv"), index=False)
pd.DataFrame(monthly).to_csv(os.path.join(RES, "analysis_monthly.csv"), index=False)
print(pd.DataFrame(desc).T)

# ------------------------------------------------ B. generator tests
gen = []
for s, name in zip(sites, sc.SITES):
    per = {}
    obs_e = s.ghi[test].sum(1) / 1000.0
    for meth, (kt, ta) in samples[name].items():
        es, cr, pit = [], [], []
        for i, d in enumerate(test):
            g = kt[i] * s.cs[d] / 1000.0
            es.append(sc.energy_score(g, s.ghi[d] / 1000.0))
            cr.append(sc.crps(g.sum(1), obs_e[i]))
            pit.append((g.sum(1) < obs_e[i]).mean() + 0.5 * (g.sum(1) == obs_e[i]).mean())
        per[meth] = dict(es=np.array(es), crps=np.array(cr), pit=np.array(pit))
        hist_c, _ = np.histogram(pit, bins=5, range=(0, 1))
        chi = stats.chisquare(hist_c)
        pooled = (kt * s.cs[test][:, None, :]).sum(-1).ravel() / 1000.0
        ks = stats.ks_2samp(pooled, obs_e)
        # hourly structure: lag-1 autocorrelation of kt inside daylight hours and inter-hour correlation matrix
        day = s.cs[test].mean(0) > 50
        k_gen = kt[:, :, day].reshape(-1, day.sum())
        k_obs = s.kt[test][:, day]
        c_gen, c_obs = np.corrcoef(k_gen.T), np.corrcoef(k_obs.T)
        gen.append(dict(site=name, method=meth, ES=float(np.mean(es)), CRPS=float(np.mean(cr)),
                        pit_chi2=float(chi.statistic), pit_p=float(chi.pvalue),
                        ks_D=float(ks.statistic), ks_p=float(ks.pvalue),
                        wasserstein=float(stats.wasserstein_distance(pooled, obs_e)),
                        mean_gen=float(pooled.mean()), mean_obs=float(obs_e.mean()),
                        sd_gen=float(pooled.std()), sd_obs=float(obs_e.std()),
                        hourcorr_err=float(np.abs(c_gen - c_obs)[np.triu_indices(day.sum(), 1)].mean()),
                        pit_hist=hist_c.tolist()))
    for base in ["Gaussian", "Bootstrap"]:
        for metric in ["es", "crps"]:
            dlt = per["CVAE"][metric] - per[base][metric]
            w = stats.wilcoxon(per["CVAE"][metric], per[base][metric])
            bs = [rng.choice(dlt, len(dlt)).mean() for _ in range(5000)]
            gen.append(dict(site=name, method=f"CVAE - {base}", metric=metric, mean_diff=float(dlt.mean()),
                            ci_lo=float(np.quantile(bs, 0.025)), ci_hi=float(np.quantile(bs, 0.975)),
                            wilcoxon_p=float(w.pvalue), cvae_better_days=int((dlt < 0).sum()), n=len(dlt)))
A["generator_tests"] = gen
pd.DataFrame(gen).drop(columns=["pit_hist"], errors="ignore").to_csv(os.path.join(RES, "analysis_generator.csv"), index=False)

# year-level persistence statistics
pers = []
for s, name in zip(sites, sc.SITES):
    obs = s.ghi.sum(1) / 1000.0
    row = dict(site=name, obs_lag1=lag1(deseason(obs)), obs_annual=float(obs.sum()))
    for k in ["cvae", "boot"]:
        e = daily(RAW[name][k])
        l1 = np.array([lag1(deseason(y)) for y in e])
        ann = e.sum(1)
        row.update({f"{k}_lag1_mean": float(l1.mean()), f"{k}_lag1_p5": float(np.quantile(l1, 0.05)),
                    f"{k}_lag1_p95": float(np.quantile(l1, 0.95)), f"{k}_annual_sd": float(ann.std(ddof=1)),
                    f"{k}_annual_mean": float(ann.mean())})
        # 3-day minimum insolation (worst 3-day spell) per year, winter matters most for storage
        w3 = np.array([np.convolve(y, np.ones(3) / 3, mode="valid").min() for y in e])
        row[f"{k}_worst3_mean"] = float(w3.mean())
        row[f"{k}_worst3_p5"] = float(np.quantile(w3, 0.05))
        pers.append({"site": name, "set": k, "lag1": l1.tolist(), "worst3": w3.tolist()})
    row["obs_worst3"] = float(np.convolve(obs, np.ones(3) / 3, mode="valid").min())
    ksl = stats.ks_2samp(pers[-2]["lag1"], pers[-1]["lag1"])
    row["lag1_ks_p"] = float(ksl.pvalue)
    A.setdefault("persistence", []).append(row)
print(pd.DataFrame(A["persistence"]).T)

# ------------------------------------------------ C. Monte Carlo statistics of the designs
mc, month_rows = [], []
for name in sc.SITES:
    water = hs.WATER_PRICE[name]
    res = {}
    for form in ["deterministic", "risk_neutral", "risk_averse"]:
        X = xvec(name, form)
        for ev in ["boot", "cvae"]:
            sim = hs.simulate(X, YEARS[name][ev], monthly=True)
            c = hs.costs(X, sim, water)
            l, r = c["lcoh"][0], c["reliability"][0]
            ok = r >= 0.95
            k, n = int(ok.sum()), len(ok)
            wl = stats.binomtest(k, n).proportion_ci(method="wilson")
            tci = stats.t.interval(0.95, n - 1, loc=l.mean(), scale=stats.sem(l))
            res[(form, ev)] = dict(l=l, r=r, ok=ok)
            mc.append(dict(site=name, formulation=form, eval_set=ev, lcoh_mean=float(l.mean()), lcoh_sd=float(l.std(ddof=1)),
                           lcoh_ci_lo=float(tci[0]), lcoh_ci_hi=float(tci[1]), lcoh_p5=float(np.quantile(l, 0.05)),
                           lcoh_p95=float(np.quantile(l, 0.95)), rel_mean=float(r.mean()), rel_sd=float(r.std(ddof=1)),
                           rel_min=float(r.min()), comply=k / n, comply_lo=float(wl.low), comply_hi=float(wl.high),
                           short_kg_mean=float(sim["short"][0].mean())))
            if ev == "boot" and form in ["deterministic", "risk_neutral"]:
                sm = sim["short_m"][0]
                for m in range(12):
                    month_rows.append(dict(site=name, formulation=form, month=m + 1, short_kg_mean=float(sm[:, m].mean()),
                                           short_kg_p95=float(np.quantile(sm[:, m], 0.95)),
                                           prod_kg_mean=float(sim["prod_m"][0][:, m].mean())))
    for ev in ["boot", "cvae"]:
        for alt in ["risk_neutral", "risk_averse"]:
            a, b = res[("deterministic", ev)], res[(alt, ev)]
            n01 = int((~a["ok"] & b["ok"]).sum()); n10 = int((a["ok"] & ~b["ok"]).sum())
            mcn = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            wl = stats.wilcoxon(b["l"], a["l"])
            dl = b["l"] - a["l"]
            mc.append(dict(site=name, formulation=f"{alt} vs deterministic", eval_set=ev,
                           lcoh_diff_mean=float(dl.mean()), lcoh_diff_pct=float(dl.mean() / a["l"].mean() * 100),
                           lcoh_wilcoxon_p=float(wl.pvalue), rel_diff_pp=float((b["r"] - a["r"]).mean() * 100),
                           mcnemar_only_alt_ok=n01, mcnemar_only_det_ok=n10, mcnemar_p=float(mcn),
                           cohens_d_rel=float((b["r"] - a["r"]).mean() / (b["r"] - a["r"]).std(ddof=1))))
A["mc"] = mc
A["monthly_shortfall"] = month_rows
pd.DataFrame(mc).to_csv(os.path.join(RES, "analysis_montecarlo.csv"), index=False)
pd.DataFrame(month_rows).to_csv(os.path.join(RES, "analysis_monthly_shortfall.csv"), index=False)

# ------------------------------------------------ D. global economic sensitivity + variance decomposition
KEYS = [("pv_capex", 0.7, 1.3), ("el_capex", 0.7, 1.3), ("b_capex", 0.7, 1.3), ("t_capex", 0.7, 1.3),
        ("rate", 0.7, 1.3), ("backup_price", 0.7, 1.3), ("water", 0.5, 2.0)]
gsa, vdec = [], []
lhs = stats.qmc.LatinHypercube(d=len(KEYS), seed=7).random(2000)
for name in sc.SITES:
    X = xvec(name, "risk_neutral")
    sim = hs.simulate(X, YEARS[name]["boot"])
    L = np.zeros((2000, sim["short"].shape[1]))
    F = np.zeros((2000, len(KEYS)))
    for i, u in enumerate(lhs):
        p = hs.P.copy(); w = hs.WATER_PRICE[name]
        for j, (k, lo, hi) in enumerate(KEYS):
            f = lo + u[j] * (hi - lo); F[i, j] = f
            if k == "water":
                w = w * f
            else:
                p[k] = hs.P[k] * f
        L[i] = hs.costs(X, sim, w, p)["lcoh"][0]
    y = L.mean(1)
    Z = (F - F.mean(0)) / F.std(0)
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(y)), Z]), (y - y.mean()) / y.std(), rcond=None)
    pred = np.column_stack([np.ones(len(y)), Z]) @ beta
    r2 = 1 - ((y - y.mean()) / y.std() - pred).var() / 1.0
    for j, (k, lo, hi) in enumerate(KEYS):
        gsa.append(dict(site=name, param=k, range=f"x{lo}-x{hi}", spearman=float(stats.spearmanr(F[:, j], y).statistic),
                        src=float(beta[j + 1]), src2_share=float(beta[j + 1] ** 2 / np.sum(beta[1:] ** 2))))
    tot = L.var()
    weather = L.var(1).mean()        # E_econ[ Var_weather ]
    econ = L.mean(1).var()           # Var_econ[ E_weather ]
    vdec.append(dict(site=name, lcoh_p5=float(np.quantile(L, 0.05)), lcoh_p50=float(np.median(L)), lcoh_p95=float(np.quantile(L, 0.95)),
                     var_total=float(tot), share_weather=float(weather / tot), share_economic=float(econ / tot), src_r2=float(r2)))
A["gsa"] = gsa
A["variance_decomposition"] = vdec
pd.DataFrame(gsa).to_csv(os.path.join(RES, "analysis_gsa.csv"), index=False)
pd.DataFrame(vdec).to_csv(os.path.join(RES, "analysis_variance.csv"), index=False)
print(pd.DataFrame(gsa)); print(pd.DataFrame(vdec))

# ------------------------------------------------ G. SAA convergence (scenario count)
saa = []
for name in sc.SITES:
    X = xvec(name, "risk_neutral")
    c = hs.costs(X, hs.simulate(X, YEARS[name]["cvae"]), hs.WATER_PRICE[name])
    l, ok = c["lcoh"][0], c["reliability"][0] >= 0.95
    for S in [5, 10, 20, 30, 50, 100]:
        means = [l[rng.choice(200, S, replace=False)].mean() for _ in range(1000)]
        comp = [ok[rng.choice(200, S, replace=False)].mean() for _ in range(1000)]
        saa.append(dict(site=name, S=S, lcoh_se=float(np.std(means)), comply_se=float(np.std(comp)),
                        comply_mean=float(np.mean(comp))))
A["saa"] = saa
pd.DataFrame(saa).to_csv(os.path.join(RES, "analysis_saa.csv"), index=False)

# ------------------------------------------------ E/F. GA trade-off and seed robustness
LO, HI = [100, 50, 0, 10], [4000, 2000, 4000, 3000]


def fitness_for(pv1, water, rel_min):
    def f(X):
        c = hs.costs(X, hs.simulate(X, pv1), water)
        ok = (c["reliability"] >= rel_min).mean(1)
        return c["lcoh"].mean(1) + 20.0 * np.maximum(0.9 - ok, 0.0)
    return f


trade, seeds = [], []
for name in sc.SITES:
    water = hs.WATER_PRICE[name]
    for rel in [0.80, 0.90, 0.95, 0.97, 0.99]:
        x, fx, _ = run_ga(fitness_for(YEARS[name]["insample"], water, rel), LO, HI, seed=11)
        X = x[None]
        cb = hs.costs(X, hs.simulate(X, YEARS[name]["boot"]), water)
        trade.append(dict(site=name, rel_target=rel, PV_kWp=x[0], EL_kW=x[1], BATT_kWh=x[2], TANK_kg=x[3],
                          lcoh_boot=float(cb["lcoh"].mean()), rel_boot=float(cb["reliability"].mean()),
                          comply_boot=float((cb["reliability"] >= rel).mean())))
        print(trade[-1])
    for sd in range(5):
        x, fx, _ = run_ga(fitness_for(YEARS[name]["insample"], water, 0.95), LO, HI, seed=300 + sd)
        seeds.append(dict(site=name, seed=300 + sd, obj=fx, PV_kWp=x[0], EL_kW=x[1], BATT_kWh=x[2], TANK_kg=x[3]))
A["tradeoff"] = trade
A["seeds"] = seeds
sd = pd.DataFrame(seeds)
agg = sd.drop(columns="seed").groupby("site").agg(["mean", "std"]).round(3)
agg.columns = [f"{a}_{b}" for a, b in agg.columns]
A["seed_summary"] = agg.reset_index().to_dict(orient="records")
pd.DataFrame(trade).to_csv(os.path.join(RES, "analysis_tradeoff.csv"), index=False)
sd.to_csv(os.path.join(RES, "analysis_seeds.csv"), index=False)
print(sd.groupby("site").agg(["mean", "std"]))

with open(os.path.join(RES, "analysis.json"), "w") as fh:
    json.dump(A, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("done")
