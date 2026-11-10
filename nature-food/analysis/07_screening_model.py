"""Brief household risk-screening models (portable logistic) for the paper and the web app.
Outputs: analysis/screening_model.json (coefficients for the app) and analysis/screening_results.json (validation)."""
import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
D = sys.argv[1]; ROOT = "/home/user/CSE366-4--2021-2-60-115/nature-food/analysis/"
d = pd.read_pickle(f"{D}/analysis.pkl"); raw = pd.read_pickle(f"{D}/clean.pkl")
num = lambda c: pd.to_numeric(raw[c], errors="coerce")
X = pd.DataFrame(index=d.index)
assets = {1:"electricity",3:"radio_tv",4:"mobile",5:"fan",7:"wardrobe",8:"fridge",9:"table_chair",10:"clock",11:"bicycle",12:"motorcycle",17:"rickshaw_van",21:"ips_generator"}
for k, n in assets.items(): X[n] = num(f"A2_9_{k}").eq(1).astype(int)
X["finished_floor"] = num("A2_12flor").ge(4).astype(int); X["finished_wall"] = num("A2_10wal").ge(8).astype(int)
X["rooms"] = num("A2_13room").clip(1, 10); X["hh_size"] = d.hh_size.clip(1, 15)
X["clean_fuel"] = num("A2_3_7").isin([1, 2, 3, 10]).astype(int); X["piped_water"] = num("A3_1").isin([1, 2, 3]).astype(int)
X["improved_toilet"] = num("A3_4").isin([1, 2, 3, 5]).astype(int); X["shared_toilet"] = num("A3_5").ne(1).astype(int)
X["internet"] = d.internet; X["urban"] = (d.area == "Urban").astype(int); X["slum"] = (d.area == "Slum").astype(int)
X["cattle"] = num("A2_8_a1").eq(1).astype(int); X["poultry"] = num("A2_8_a3").eq(1).astype(int)
X["log_land_cult"] = np.log1p(num("A2_3_5").clip(lower=0).fillna(0)); X["safety_net"] = d.safety_net.fillna(0)
X["remittance"] = num("A2_17_rmtnce").eq(2).astype(int)
divs = ["Chattogram", "Dhaka", "Khulna", "Mymensingh", "Rajshahi", "Rangpur", "Sylhet"]   # ref: Barishal
for dv in divs: X[f"div_{dv}"] = (d.division == dv).astype(int)
X = X.fillna(X.median())
Y = {"food_insecurity": d.food_insecure_modsev, "no_handwash_facility": 1 - d.hw_station_water_soap, "low_diet_diversity": 1 - d.mdd_cook}
labels = {"food_insecurity": "Moderate or severe household food insecurity (HFIAS)", "no_handwash_facility": "No handwashing place with water and soap (observed)",
          "low_diet_diversity": "Cook ate fewer than 5 of 10 food groups"}
w = d.groupby("sample").w.transform(lambda x: x / x.mean())
model_out = {"features": list(X.columns), "divisions_ref": "Barishal", "models": {}}
val = {}
for t, y in Y.items():
    m = y.notna(); Xm, ym, wm, gm = X[m], y[m].astype(int), w[m], d.psu[m]
    mu, sd = Xm.mean(), Xm.std().replace(0, 1)
    Z = (Xm - mu) / sd
    oof_lr, oof_gb = np.zeros(len(ym)), np.zeros(len(ym))
    for tr, te in GroupKFold(5).split(Z, ym, gm):
        lr = LogisticRegression(C=0.3, max_iter=3000).fit(Z.iloc[tr], ym.iloc[tr], sample_weight=wm.iloc[tr]); oof_lr[te] = lr.predict_proba(Z.iloc[te])[:, 1]
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=1).fit(Xm.iloc[tr], ym.iloc[tr], sample_weight=wm.iloc[tr])
        oof_gb[te] = gb.predict_proba(Xm.iloc[te])[:, 1]
    # bootstrap CI of AUC over PSUs
    rng = np.random.default_rng(7); psus = gm.unique(); aucs = []
    idx_by = {p: np.where(gm.values == p)[0] for p in psus}
    for b in range(300):
        ii = np.concatenate([idx_by[p] for p in rng.choice(psus, len(psus))])
        aucs.append(roc_auc_score(ym.values[ii], oof_lr[ii], sample_weight=wm.values[ii]))
    # capture curve: share of cases among top-k% predicted risk
    order = np.argsort(-oof_lr); yy = ym.values[order]; ww = wm.values[order]
    cumw = np.cumsum(ww) / ww.sum(); cap = np.cumsum(yy * ww) / (yy * ww).sum()
    grid = np.linspace(0, 1, 21); capture = [float(np.interp(g, np.r_[0, cumw], np.r_[0, cap])) for g in grid]
    # calibration by decile
    dec = pd.qcut(oof_lr, 10, labels=False, duplicates="drop")
    calib = [(float(np.average(oof_lr[dec == k], weights=wm.values[dec == k])), float(np.average(ym.values[dec == k], weights=wm.values[dec == k]))) for k in range(int(dec.max()) + 1)]
    final = LogisticRegression(C=0.3, max_iter=3000).fit(Z, ym, sample_weight=wm)
    coef = pd.Series(final.coef_[0] / sd.values, index=X.columns); intercept = float(final.intercept_[0] - (final.coef_[0] * mu.values / sd.values).sum())
    prev = float(np.average(ym, weights=wm))
    top30 = float(np.interp(0.3, np.r_[0, cumw], np.r_[0, cap]))
    # thresholds for the app: risk bands by tertile of predicted probability
    q = np.quantile(oof_lr, [1/3, 2/3])
    model_out["models"][t] = {"label": labels[t], "intercept": round(intercept, 5), "coef": {k: round(float(v), 5) for k, v in coef.items()},
                              "prevalence": round(prev, 4), "bands": [round(float(q[0]), 4), round(float(q[1]), 4)]}
    val[t] = {"label": labels[t], "n": int(m.sum()), "prevalence": round(100 * prev, 1),
              "auc_logit": round(roc_auc_score(ym, oof_lr, sample_weight=wm), 3), "auc_logit_ci": [round(float(np.percentile(aucs, 2.5)), 3), round(float(np.percentile(aucs, 97.5)), 3)],
              "auc_gbm": round(roc_auc_score(ym, oof_gb, sample_weight=wm), 3), "top30_capture": round(100 * top30, 1),
              "lift_top30": round(top30 / 0.3, 2), "capture_curve": capture, "calibration": calib,
              "top_coefs": {k: round(float(v), 3) for k, v in coef.reindex(coef.abs().sort_values(ascending=False).index).head(8).items()}}
    print(f"{t:22s} prev {100*prev:4.1f}%  AUC logit {val[t]['auc_logit']} ({val[t]['auc_logit_ci'][0]}-{val[t]['auc_logit_ci'][1]})  GBM {val[t]['auc_gbm']}  top30 captures {100*top30:.0f}%")
json.dump(model_out, open(ROOT + "screening_model.json", "w"), indent=1)
json.dump(val, open(ROOT + "screening_results.json", "w"), indent=1)
print(json.dumps({t: v["top_coefs"] for t, v in val.items()}, indent=0))
