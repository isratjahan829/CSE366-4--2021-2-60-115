"""Main analyses: survey-weighted regressions with PSU-clustered standard errors."""
import sys, json, warnings, numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
D=sys.argv[1]; OUT=sys.argv[2]
d=pd.read_pickle(f"{D}/analysis.pkl")
d["urban"]=(d.area=="Urban").astype(int)
d["hfias_cat"]=pd.Categorical(d.hfias_cat,["Secure","Mild","Moderate","Severe"])
d["wealth_q"]=d.wealth_q.astype(float)
for r in ("cook","purch","adol"):
    d[f"bn10_{r}"]=d[f"bn_{r}"]/10
d["fs10"]=d.foodsafety_know/10; d["ck10"]=d.cooking_know/10; d["iy10"]=d.iycf_know/10
R={}
def wm(x,w): m=x.notna()&w.notna(); return float(np.average(x[m],weights=w[m]))
def fit(formula,data,family=None):
    if data.urban.nunique()==1: formula=formula.replace(" + urban","").replace("urban + ","")
    data=data.dropna(subset=[c for c in data.columns if c in formula.replace("C(","").replace(")"," ").replace("*"," ").replace("+"," ").replace("~"," ").split()]+["w","psu"])
    if family is None:
        m=smf.wls(formula,data=data,weights=data.w).fit(cov_type="cluster",cov_kwds={"groups":data.psu})
    else:
        m=smf.glm(formula,data=data,family=family,var_weights=data.w/data.w.mean()).fit(cov_type="cluster",cov_kwds={"groups":data.psu})
    return m
def coef(m,term,exp=False):
    b=m.params[term]; lo,hi=m.conf_int().loc[term]; p=m.pvalues[term]
    f=(lambda x: float(np.exp(x))) if exp else float
    return {"est":round(f(b),3),"lo":round(f(lo),3),"hi":round(f(hi),3),"p":float(p),"n":int(m.nobs)}
ru=d[d["sample"]=="Rural-urban"].copy(); sl=d[d["sample"]=="Slum"].copy()

# ---- 0. Descriptives ----
desc={}
for s,g in [("RU",ru),("Slum",sl)]:
    x={}
    for r in ("cook","purch","adol"):
        v=g[f"bn_{r}"]; x[f"bn_{r}_mean"]=round(wm(v,g.w),1); x[f"bn_{r}_n"]=int(v.notna().sum())
        x[f"bn_{r}_ge60"]=round(100*wm((v>=60).astype(float).where(v.notna()),g.w),1)
        x[f"dds_{r}_mean"]=round(wm(g[f"dds_{r}"],g.w),2); x[f"mdd_{r}"]=round(100*wm(g[f"mdd_{r}"],g.w),1)
        x[f"anydisc_{r}"]=round(100*wm(g[f"anydisc_{r}"],g.w),1)
        for z in ("snacks","sweets","ssb"): x[f"{z}_{r}"]=round(100*wm(g[f"dz_{r}_{z}"],g.w),1)
    for v in ["foodsafety_know","cooking_know","iycf_know"]: x[v]=round(wm(g[v],g.w),1)
    x["fs_ge60"]=round(100*wm((g.foodsafety_know>=60).astype(float).where(g.foodsafety_know.notna()),g.w),1)
    for c in ["food_insecure_modsev","internet","hw_station_water_soap","food_near_chem_obs","child_mdd","safety_net"]:
        x[c]=round(100*wm(g[c],g.w),1)
    x["hfias"]={k:round(100*float(v),1) for k,v in (g.groupby("hfias_cat").w.sum()/g.w.sum()).items()}
    x["purch_is_cook_share"]=round(100*float(g.bn_purch.isna().mean()),1)
    x["n_hh"]=int(len(g)); x["n_psu"]=int(g.psu.nunique())
    desc[s]=x
R["descriptives"]=desc
# item-level knowledge (RU), % fully correct (score==1) and mean credit
items={}
for r in ("cook","purch","adol"):
    cols=[c for c in d.columns if c.startswith(f"bn_{r}_")]
    items[r]={c.replace(f"bn_{r}_",""):{"mean_credit":round(100*wm(ru[c],ru.w),1),"fully":round(100*wm((ru[c]>=0.999).astype(float).where(ru[c].notna()),ru.w),1),"none":round(100*wm((ru[c]<=0.001).astype(float).where(ru[c].notna()),ru.w),1)} for c in cols}
for pre,lab in [("fs_","foodsafety"),("ck_","cooking"),("iy_","iycf")]:
    cols=[c for c in d.columns if c.startswith(pre)]
    items[lab]={c[len(pre):]:{"mean_credit":round(100*wm(ru[c],ru.w),1),"none":round(100*wm((ru[c]<=0.001).astype(float).where(ru[c].notna()),ru.w),1)} for c in cols}
R["items_RU"]=items

# ---- 1. Correlates of knowledge (M1) ----
cov="C(wealth_q) + urban + C(division) + internet + C(hfias_cat)"
M1={}
for r in ("cook","purch","adol"):
    m=fit(f"bn_{r} ~ {cov} + msg30_{r}",ru)
    M1[r]={t:coef(m,t) for t in m.params.index if t!="Intercept"}
    M1[r]["_r2"]=round(m.rsquared,3)
m=fit(f"foodsafety_know ~ {cov} + msg30_cook",ru); M1["foodsafety"]={t:coef(m,t) for t in m.params.index if t!="Intercept"}
R["M1_knowledge_correlates_RU"]=M1
# wealth gradient: Q5 - Q1 unadjusted means
R["wealth_gradient_bn"]={r:{int(q):round(wm(g[f"bn_{r}"],g.w),1) for q,g in ru.groupby("wealth_q")} for r in ("cook","purch","adol")}

# ---- 2. Knowledge -> diet (M2) and moderation by food insecurity (M3) ----
adj="C(wealth_q) + urban + C(division) + internet + log_food_exp_pc + C(hfias_cat)"
M2={}
for s,g in [("RU",ru),("Slum",sl)]:
    M2[s]={}
    for r in ("cook","purch","adol"):
        x={}
        m=fit(f"dds_{r} ~ bn10_{r}",g); x["dds_crude"]=coef(m,f"bn10_{r}")
        m=fit(f"dds_{r} ~ bn10_{r} + {adj}",g); x["dds_adj"]=coef(m,f"bn10_{r}")
        x["dds_adj_wealthQ5"]=coef(m,"C(wealth_q)[T.5.0]") if "C(wealth_q)[T.5.0]" in m.params else None
        x["dds_adj_foodexp"]=coef(m,"log_food_exp_pc")
        x["dds_adj_severe"]=coef(m,"C(hfias_cat)[T.Severe]")
        m=fit(f"mdd_{r} ~ bn10_{r} + {adj}",g,sm.families.Binomial()); x["mdd_adj_OR"]=coef(m,f"bn10_{r}",exp=True)
        m=fit(f"anydisc_{r} ~ bn10_{r} + {adj}",g,sm.families.Binomial()); x["disc_adj_OR"]=coef(m,f"bn10_{r}",exp=True)
        m=fit(f"disc_{r} ~ bn10_{r} + {adj}",g); x["disc_count_adj"]=coef(m,f"bn10_{r}")
        # M3 interaction
        g2=g.copy(); g2["fi"]=g2.food_insecure_modsev
        m=fit(f"dds_{r} ~ bn10_{r}*fi + C(wealth_q) + urban + C(division) + internet + log_food_exp_pc",g2)
        b=m.params; V=m.cov_params()
        sec=(b[f"bn10_{r}"], np.sqrt(V.loc[f"bn10_{r}",f"bn10_{r}"]))
        ins_b=b[f"bn10_{r}"]+b[f"bn10_{r}:fi"]; ins_se=np.sqrt(V.loc[f"bn10_{r}",f"bn10_{r}"]+V.loc[f"bn10_{r}:fi",f"bn10_{r}:fi"]+2*V.loc[f"bn10_{r}",f"bn10_{r}:fi"])
        x["slope_secure_mild"]={"est":round(sec[0],3),"lo":round(sec[0]-1.96*sec[1],3),"hi":round(sec[0]+1.96*sec[1],3)}
        x["slope_mod_severe"]={"est":round(ins_b,3),"lo":round(ins_b-1.96*ins_se,3),"hi":round(ins_b+1.96*ins_se,3)}
        x["interaction_p"]=float(m.pvalues[f"bn10_{r}:fi"])
        # wealth interaction (poorest two quintiles vs others)
        g2["poor"]=(g2.wealth_q<=2).astype(int)
        m=fit(f"dds_{r} ~ bn10_{r}*poor + urban + C(division) + internet + log_food_exp_pc + C(hfias_cat)",g2)
        b=m.params; V=m.cov_params()
        pb=b[f"bn10_{r}"]+b[f"bn10_{r}:poor"]; pse=np.sqrt(V.loc[f"bn10_{r}",f"bn10_{r}"]+V.loc[f"bn10_{r}:poor",f"bn10_{r}:poor"]+2*V.loc[f"bn10_{r}",f"bn10_{r}:poor"])
        x["slope_nonpoor"]={"est":round(b[f"bn10_{r}"],3)}; x["slope_poor"]={"est":round(pb,3),"lo":round(pb-1.96*pse,3),"hi":round(pb+1.96*pse,3)}
        x["poor_interaction_p"]=float(m.pvalues[f"bn10_{r}:poor"])
        M2[s][r]=x
R["M2_knowledge_diet"]=M2

# ---- 3. Decomposition of rich-poor DDS gap (cooks, RU) ----
dec={}
for r in ("cook","purch"):
    g=ru.dropna(subset=[f"dds_{r}",f"bn_{r}","log_food_exp_pc","hfias_score","internet","wealth_q"])
    m=fit(f"dds_{r} ~ bn10_{r} + hfias_score + log_food_exp_pc + internet + urban + C(division)",g)
    rich=g[g.wealth_q==5]; poor=g[g.wealth_q==1]
    gap=wm(rich[f"dds_{r}"],rich.w)-wm(poor[f"dds_{r}"],poor.w)
    comp={}
    for v,lab in [(f"bn10_{r}","Knowledge"),("hfias_score","Food insecurity"),("log_food_exp_pc","Food expenditure"),("internet","Internet access"),("urban","Urban residence")]:
        comp[lab]=float(m.params[v]*(wm(rich[v],rich.w)-wm(poor[v],poor.w)))
    dec[r]={"gap":round(gap,3),"components":{k:round(v,3) for k,v in comp.items()},
            "share_pct":{k:round(100*v/gap,1) for k,v in comp.items()},"unexplained":round(gap-sum(comp.values()),3)}
R["decomposition_RU"]=dec

# ---- 4. Food safety: knowledge vs observed practice ----
fsx={}
for s,g in [("RU",ru),("Slum",sl)]:
    x={}
    m=fit(f"hw_station_water_soap ~ fs10 + C(wealth_q) + urban + C(division) + improved_water",g,sm.families.Binomial()); x["station_OR_per10"]=coef(m,"fs10",exp=True)
    x["station_wealthQ5_OR"]=coef(m,"C(wealth_q)[T.5.0]",exp=True)
    m=fit(f"food_near_chem_obs ~ fs10 + C(wealth_q) + urban + C(division)",g,sm.families.Binomial()); x["nearchem_OR_per10"]=coef(m,"fs10",exp=True)
    knows=g.fs_no_store_near_chem==1
    x["near_chem_obs_if_knows"]=round(100*wm(g.food_near_chem_obs[knows],g.w[knows]),1)
    x["near_chem_obs_if_not"]=round(100*wm(g.food_near_chem_obs[~knows],g.w[~knows]),1)
    rep=g.hw_rep_before_cooking==1
    x["station_ws_if_reports_before_cooking"]=round(100*wm(g.hw_station_water_soap[rep],g.w[rep]),1)
    x["station_ws_overall"]=round(100*wm(g.hw_station_water_soap,g.w),1)
    x["station_ws_by_wealth"]={int(q):round(100*wm(h.hw_station_water_soap,h.w),1) for q,h in g.groupby("wealth_q")}
    fsx[s]=x
R["foodsafety_practice"]=fsx

# ---- 5. Handwashing knowledge vs reported practice & COVID attribution ----
hw={}
for s,g in [("RU",ru),("Slum",sl)]:
    x={}
    for n in ["before_eating","after_latrine","before_feeding_baby","after_child_faeces","before_cooking","after_returning_home"]:
        rep=g[f"hw_rep_{n}"]; pre=g[f"hw_pre_{n}"]; cov_=g[f"hw_covid_{n}"]; con=g[f"hw_cont_{n}"]
        sub=rep==1
        x[n]={"know":round(100*wm(g[f"hw_know_{n}"],g.w),1),"report":round(100*wm(rep,g.w),1),
              "pre_share":round(100*wm(pre[sub],g.w[sub]),1) if pre[sub].notna().sum()>20 else None,
              "covid_share":round(100*wm(cov_[sub],g.w[sub]),1) if cov_[sub].notna().sum()>20 else None}
    hw[s]=x
R["handwashing"]=hw

# ---- 6. Child diets ----
ch={}
for s,g in [("RU",ru),("Slum",sl)]:
    gc=g[g.child_mdd.notna()]
    x={"n":int(len(gc)),"mdd":round(100*wm(gc.child_mdd,gc.w),1),"iycf_know":round(wm(gc.iycf_know,gc.w),1)}
    x["groups"]={c[4:]:round(100*wm(gc[c],gc.w),1) for c in gc.columns if c.startswith("cfg_")}
    ch[s]=x
gc=ru[ru.child_mdd.notna()]
m=fit("child_mdd ~ iy10 + child_age_m + C(wealth_q) + urban + C(hfias_cat)",gc,sm.families.Binomial())
ch["RU"]["mdd_OR_iycf10"]=coef(m,"iy10",exp=True)
m=fit("child_mdd ~ iy10 + bn10_cook + child_age_m + C(wealth_q) + urban + C(hfias_cat)",gc,sm.families.Binomial())
ch["RU"]["mdd_OR_cookbn10"]=coef(m,"bn10_cook",exp=True)
R["child"]=ch
json.dump(R,open(OUT,"w"),indent=1,default=str)
print("saved")
