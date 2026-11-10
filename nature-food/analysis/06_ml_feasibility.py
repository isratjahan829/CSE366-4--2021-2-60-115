"""Feasibility: can a short household questionnaire predict food-security / food-safety risk? (PSU-grouped cross-validation)"""
import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
D = sys.argv[1]; OUT = sys.argv[2]
d = pd.read_pickle(f"{D}/analysis.pkl"); raw = pd.read_pickle(f"{D}/clean.pkl")
num = lambda c: pd.to_numeric(raw[c], errors="coerce")
X = pd.DataFrame(index=d.index)
names = {1:"electricity",2:"solar",3:"radio_tv",4:"mobile",5:"fan",6:"bed",7:"wardrobe",8:"fridge",9:"table_chair",10:"clock",11:"bicycle",
         12:"motorcycle",14:"car",15:"boat",17:"rickshaw_van",19:"water_pump",21:"ips_generator",22:"fishing_net"}
for k, n in names.items(): X[n] = num(f"A2_9_{k}").eq(1).astype(int)
X["finished_floor"] = num("A2_12flor").ge(4).astype(int); X["finished_wall"] = num("A2_10wal").ge(8).astype(int)
X["rooms"] = num("A2_13room").clip(upper=10); X["hh_size"] = d.hh_size
X["clean_fuel"] = num("A2_3_7").isin([1, 2, 3, 10]).astype(int); X["piped_water"] = num("A3_1").isin([1, 2, 3]).astype(int)
X["improved_toilet"] = num("A3_4").isin([1, 2, 3, 5]).astype(int); X["shared_toilet"] = num("A3_5").ne(1).astype(int)
X["internet"] = d.internet; X["urban"] = (d.area == "Urban").astype(int); X["slum"] = (d.area == "Slum").astype(int)
X["cattle"] = num("A2_8_a1").eq(1).astype(int); X["poultry"] = num("A2_8_a3").eq(1).astype(int)
X["land_cult"] = np.log1p(num("A2_3_5").clip(lower=0).fillna(0)); X["safety_net"] = d.safety_net.fillna(0)
X["remittance"] = num("A2_17_rmtnce").eq(2).astype(int)
for dv in d.division.unique(): X[f"div_{dv}"] = (d.division == dv).astype(int)
X = X.fillna(X.median())
targets = {"food_insecure_modsev": "Moderate/severe food insecurity (HFIAS)",
           "no_water_soap": "No handwashing place with water and soap (observed)",
           "low_dds_cook": "Cook ate fewer than 5 food groups",
           "low_fs_know": "Cook food-safety knowledge below 50"}
Y = pd.DataFrame({"food_insecure_modsev": d.food_insecure_modsev, "no_water_soap": 1 - d.hw_station_water_soap,
                  "low_dds_cook": 1 - d.mdd_cook, "low_fs_know": (d.foodsafety_know < 50).astype(float).where(d.foodsafety_know.notna())})
w = d.groupby("sample").w.transform(lambda x: x / x.mean())   # normalise weights within sample
res = {}
for t, lab in targets.items():
    m = Y[t].notna(); Xm, ym, gm, wm = X[m], Y[t][m].astype(int), d.psu[m], w[m]
    oof_gb = np.zeros(len(ym)); oof_lr = np.zeros(len(ym))
    for tr, te in GroupKFold(n_splits=5).split(Xm, ym, gm):
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=1)
        gb.fit(Xm.iloc[tr], ym.iloc[tr], sample_weight=wm.iloc[tr]); oof_gb[te] = gb.predict_proba(Xm.iloc[te])[:, 1]
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
        lr.fit(Xm.iloc[tr], ym.iloc[tr], logisticregression__sample_weight=wm.iloc[tr]); oof_lr[te] = lr.predict_proba(Xm.iloc[te])[:, 1]
    auc_gb = roc_auc_score(ym, oof_gb, sample_weight=wm); auc_lr = roc_auc_score(ym, oof_lr, sample_weight=wm)
    # targeting: share of high-risk households captured if the top 30% by predicted risk are targeted
    cut = np.quantile(oof_gb, 0.70); cap = float((ym[oof_gb >= cut] * wm[oof_gb >= cut]).sum() / (ym * wm).sum())
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=1).fit(Xm, ym, sample_weight=wm)
    pi = permutation_importance(gb, Xm, ym, scoring="roc_auc", n_repeats=5, random_state=1)
    top = pd.Series(pi.importances_mean, index=Xm.columns).sort_values(ascending=False).head(6)
    res[t] = {"label": lab, "n": int(m.sum()), "prevalence": round(float(np.average(ym, weights=wm)) * 100, 1),
              "auc_gbm": round(auc_gb, 3), "auc_logit": round(auc_lr, 3), "captured_top30_pct": round(100 * cap, 1),
              "top_features": {k: round(float(v), 4) for k, v in top.items()}}
    print(f"{lab:55s} prev {res[t]['prevalence']:5.1f}%  AUC GBM {auc_gb:.3f}  logit {auc_lr:.3f}  top-30% captures {100*cap:.0f}%  | {list(top.index[:4])}")
json.dump(res, open(OUT, "w"), indent=1)
