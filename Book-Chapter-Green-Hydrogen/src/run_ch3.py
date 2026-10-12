"""Chapter 3 experiment: three-objective NSGA-II sizing with PV cleaning interval (dust soiling vs water),
TOPSIS decision-making, hypervolume statistics over seeds, soiling-scenario and weight sensitivity.
Objectives (minimise): f1 expected LCOH [USD/kg], f2 expected shortfall [% of demand],
f3 total water [L per kg produced on site] = electrolysis feed + PV cleaning water.
Writes ../results/ch3_*.csv and ch3.json"""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats

import data_prep as dp
import scenarios as sc
import h2system as hs
from nsga2 import run_nsga2, nondominated_sort, hypervolume, topsis
from run_ch2 import block_bootstrap

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LO = [100, 50, 0, 10, 1]
HI = [4000, 2000, 2000, 3000, 90]          # PV kWp, EL kW, battery kWh, tank kg, cleaning interval [days]
M2_PER_KWP = 5.0                             # module area (~20 % efficient modules)
GCR = 0.40                                   # ground coverage ratio -> land = area / GCR
CLEAN_L_M2 = 0.5                             # water per wet cleaning [L/m2]
CLEAN_USD_M2 = 0.015                         # labour + equipment per cleaning [USD/m2]
SOIL = {"A_semiarid_CO": 0.25, "B_humid_NC": 0.05}   # base soiling rate [% PV loss per day]
SOIL_CAP = 25.0                              # maximum soiling loss [%]


def soiling_loss(tau, rate):
    """Mean loss of a saw-tooth that grows `rate` %/day and is reset every tau days (capped)."""
    days = np.arange(int(np.ceil(tau.max())) + 1)
    out = np.empty(len(tau))
    for i, t in enumerate(tau):
        d = days[:max(int(round(t)), 1)]
        out[i] = np.minimum(d * rate, SOIL_CAP).mean() / 100.0
    return out


def make_eval(pv_years, water_price, rate):
    D_year = hs.P["demand_kg_day"] * 365

    def evaluate(X, detail=False):
        loss = soiling_loss(X[:, 4], rate)
        Xe = X[:, :4].copy()
        Xe[:, 0] = X[:, 0] * (1 - loss)                     # soiled PV behaves like a smaller array
        sim = hs.simulate(Xe, pv_years)
        Xc = X[:, :4]
        fixed = hs.annual_fixed_cost(Xc)
        area = X[:, 0] * M2_PER_KWP
        cleanings = 365.0 / X[:, 4]
        clean_cost = cleanings * area * CLEAN_USD_M2
        clean_water_m3 = cleanings * area * CLEAN_L_M2 / 1000.0
        feed_m3 = sim["prod"] * hs.P["water_l_per_kg"] / 1000.0
        cost = fixed[:, None] + clean_cost[:, None] + (feed_m3 + clean_water_m3[:, None]) * water_price \
            + sim["short"] * hs.P["backup_price"]
        lcoh = (cost / D_year).mean(1)
        short = (sim["short"] / D_year).mean(1) * 100
        water = ((feed_m3 + clean_water_m3[:, None]) * 1000 / np.maximum(sim["prod"], 1.0)).mean(1)
        F = np.column_stack([lcoh, short, water])
        if detail:
            return F, dict(loss=loss * 100, land_ha=area / GCR / 1e4, clean_water_m3=clean_water_m3,
                           curtail=(sim["curt"] / (Xe[:, :1] * pv_years.sum(1))).mean(1) * 100)
        return F
    return evaluate


def main():
    df = dp.build()
    years = {}
    for i, name in enumerate(sc.SITES):
        s = sc.SiteData(*dp.daily_arrays(df, name), i)
        geo = hs.geometry(dp.SITES[name])
        rng = np.random.default_rng(300 + i)
        yrs = [(s.ghi, s.ta)] + [block_bootstrap(s, rng) for _ in range(9)]
        years[name] = np.array([hs.pv_per_kwp(g.reshape(-1), t.reshape(-1), geo) for g, t in yrs])
    REF, IDEAL = [12.0, 30.0, 40.0], [5.0, 0.0, 14.0]     # hypervolume box (LCOH, shortfall %, water L/kg)
    fronts, hv_rows, out = [], [], {}
    runs = [(n, SOIL[n], f"base") for n in sc.SITES] + [("A_semiarid_CO", 0.10, "low dust"), ("A_semiarid_CO", 0.50, "high dust")]
    for name, rate, label in runs:
        ev = make_eval(years[name], hs.WATER_PRICE[name], rate)
        seeds = range(5) if label == "base" else range(2)
        best = None
        for sd in seeds:
            X, F, hh = run_nsga2(ev, LO, HI, pop=60, gens=60, seed=sd, hv_ref=REF, hv_scale=IDEAL)
            hv_rows.append(dict(site=name, scenario=label, method="NSGA-II", seed=sd, hv=hh[-1][1], n_front=len(X)))
            out[f"{name}|{label}|hvhist|{sd}"] = hh
            if best is None or hh[-1][1] > best[2]:
                best = (X, F, hh[-1][1])
            if label == "base":   # random-search baseline with the same 3 660 evaluations
                rng = np.random.default_rng(900 + sd)
                XR = np.array(LO) + rng.random((60 * 61, 5)) * (np.array(HI) - np.array(LO))
                FR = np.vstack([ev(XR[k:k + 60]) for k in range(0, len(XR), 60)])
                fr = nondominated_sort(FR) == 0
                hv_rows.append(dict(site=name, scenario=label, method="Random search", seed=sd,
                                    hv=hypervolume(FR[fr], REF, IDEAL), n_front=int(fr.sum())))
            print(hv_rows[-1], flush=True)
        X, F, _ = best
        Fd, det = ev(X, detail=True)
        for k in range(len(X)):
            fronts.append(dict(site=name, scenario=label, PV_kWp=X[k, 0], EL_kW=X[k, 1], BATT_kWh=X[k, 2], TANK_kg=X[k, 3],
                               clean_days=X[k, 4], lcoh=F[k, 0], short_pct=F[k, 1], water_l_kg=F[k, 2],
                               soil_loss_pct=det["loss"][k], land_ha=det["land_ha"][k], curtail_pct=det["curtail"][k]))
    fr = pd.DataFrame(fronts)
    fr.to_csv(os.path.join(RES, "ch3_fronts.csv"), index=False)
    hv = pd.DataFrame(hv_rows)
    hv.to_csv(os.path.join(RES, "ch3_hypervolume.csv"), index=False)
    post(fr, hv, out)


def post(fr, hv, out):

    # TOPSIS: base weights + Monte Carlo weight sensitivity (Dirichlet) on the base fronts
    rng = np.random.default_rng(5)
    tops, sens = [], []
    for name in sc.SITES:
        g = fr[(fr.site == name) & (fr.scenario == "base")].reset_index(drop=True)
        F = g[["lcoh", "short_pct", "water_l_kg"]].values
        for wname, w in [("cost-led (0.5,0.3,0.2)", [0.5, 0.3, 0.2]), ("equal (1/3 each)", [1 / 3] * 3),
                         ("reliability-led (0.3,0.5,0.2)", [0.3, 0.5, 0.2]), ("water-led (0.3,0.2,0.5)", [0.3, 0.2, 0.5])]:
            cl = topsis(F + 1e-6, np.array(w))
            k = int(np.argmax(cl))
            tops.append({**g.iloc[k].to_dict(), "weights": wname, "closeness": float(cl[k]), "idx": k})
        W = rng.dirichlet([1, 1, 1], 10000)
        picks = np.array([np.argmax(topsis(F + 1e-6, w)) for w in W])
        cnt = pd.Series(picks).value_counts()
        base_rank = topsis(F + 1e-6, np.array([0.5, 0.3, 0.2]))
        rhos = [stats.spearmanr(base_rank, topsis(F + 1e-6, w)).statistic for w in W[:1000]]
        sens.append(dict(site=name, n_front=len(F), n_distinct_winners=int(len(cnt)),
                         top1_share=float(cnt.iloc[0] / len(W)), top3_share=float(cnt.iloc[:3].sum() / len(W)),
                         top1_idx=int(cnt.index[0]), spearman_median=float(np.median(rhos)),
                         spearman_p5=float(np.quantile(rhos, 0.05))))
        for idx in cnt.index[:3]:
            r = g.iloc[idx].to_dict(); r.update(site=name, share=float(cnt[idx] / len(W)))
            out.setdefault("weight_winners", []).append(r)
        # correlations among objectives on the front
        out.setdefault("obj_corr", []).append(dict(site=name,
            lcoh_short=float(stats.spearmanr(F[:, 0], F[:, 1]).statistic),
            lcoh_water=float(stats.spearmanr(F[:, 0], F[:, 2]).statistic),
            short_water=float(stats.spearmanr(F[:, 1], F[:, 2]).statistic)))
    pd.DataFrame(tops).to_csv(os.path.join(RES, "ch3_topsis.csv"), index=False)
    out["topsis_sensitivity"] = sens

    # statistics on hypervolume: NSGA-II vs random search (paired by seed) and seed variability
    hvs = []
    for name in sc.SITES:
        a = hv[(hv.site == name) & (hv.method == "NSGA-II") & (hv.scenario == "base")].sort_values("seed").hv.values
        b = hv[(hv.site == name) & (hv.method == "Random search")].sort_values("seed").hv.values
        hvs.append(dict(site=name, nsga_mean=a.mean(), nsga_sd=a.std(ddof=1), rs_mean=b.mean(), rs_sd=b.std(ddof=1),
                        wilcoxon_p=float(stats.wilcoxon(a, b, alternative="greater").pvalue),
                        mannwhitney_p=float(stats.mannwhitneyu(a, b, alternative="greater").pvalue)))
    out["hv_stats"] = hvs
    # extreme-point summary per front
    summ = []
    for (name, label), g in fr.groupby(["site", "scenario"]):
        summ.append(dict(site=name, scenario=label, n=len(g), lcoh_min=g.lcoh.min(), short_min=g.short_pct.min(),
                         water_min=g.water_l_kg.min(), water_max=g.water_l_kg.max(),
                         clean_days_median=g.clean_days.median(), soil_loss_median=g.soil_loss_pct.median(),
                         lcoh_at_short_le5=g[g.short_pct <= 5].lcoh.min() if (g.short_pct <= 5).any() else np.nan,
                         land_ha_median=g.land_ha.median()))
    out["front_summary"] = summ
    pd.DataFrame(summ).to_csv(os.path.join(RES, "ch3_front_summary.csv"), index=False)
    with open(os.path.join(RES, "ch3.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(pd.DataFrame(summ)); print(pd.DataFrame(hvs)); print(pd.DataFrame(sens)); print(pd.DataFrame(tops))
    print("done")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["post"]:
        post(pd.read_csv(os.path.join(RES, "ch3_fronts.csv")), pd.read_csv(os.path.join(RES, "ch3_hypervolume.csv")), {})
    else:
        main()
