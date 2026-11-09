import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
D=sys.argv[1]; d=pd.read_pickle(f"{D}/analysis.pkl")
def wm(x,w): m=x.notna(); return np.nan if m.sum()==0 else 100*np.average(x[m],weights=w[m])
rows=[]
chk={"Cooks: reheat knowledge = once (report 42.3% correct)":"fs_reheat_once",
"Cooks: know not to store near chemicals (report 47.8%)":"fs_no_store_near_chem",
"Cooks: separate raw/cooked (report 81.7%)":"fs_separate_raw_cooked",
"Cooks: wash veg whole (report 24.6%)":"ck_veg_wash_whole",
"Cooks: keep gruel (report 75.1%)":"ck_gruel_keep",
"Purch: vitamin A any correct":"bn_purch_vitA_foods",
"Purch: inadequate DDS (report 57.3% RU)":None,
"HW reported before eating (report 96.5% RU)":"hw_rep_before_eating",
"HW reported before cooking (report 49.7% RU)":"hw_rep_before_cooking",
"HW reported before feeding baby (report 42.0% RU)":"hw_rep_before_feeding_baby",
"Obs station with water+soap":"hw_station_water_soap",
"Mod/severe food insecure":"food_insecure_modsev",
"Internet in HH":"internet",
"Child MDD (report 40.7 RU / 31.0 slum)":"child_mdd"}
d["inad_purch"]=1-d.mdd_purch
for s,g in d.groupby("sample"):
    for k,v in chk.items():
        v = v or "inad_purch"
        rows.append((s,k,round(wm(g[v],g.w),1), int(g[v].notna().sum())))
r=pd.DataFrame(rows,columns=["sample","indicator","weighted %","n"])
print(r.pivot(index="indicator",columns="sample",values="weighted %").to_string())
print()
for s,g in d.groupby("sample"):
    print(s, "BMI cat purch", (g.bmicat_purch.value_counts(normalize=True)*100).round(1).to_dict(), "BMI cat cook", (g.bmicat_cook.value_counts(normalize=True)*100).round(1).to_dict())
    print(s, "HFIAS cat", (g.groupby("hfias_cat").w.sum()/g.w.sum()*100).round(1).to_dict())
    for r_ in ("cook","purch","adol"):
        print(s, r_, "BN mean", round(np.average(g[f"bn_{r_}"].dropna(),weights=g.w[g[f'bn_{r_}'].notna()]),1), "n", g[f"bn_{r_}"].notna().sum(),
              "DDS mean", round(np.average(g[f"dds_{r_}"].dropna(),weights=g.w[g[f'dds_{r_}'].notna()]),2),
              "any disc", round(wm(g[f"anydisc_{r_}"],g.w),1))
    print(s,"FS mean",round(wm(g.foodsafety_know/100,g.w),1),"Cooking mean",round(wm(g.cooking_know/100,g.w),1),"IYCF mean",round(wm(g.iycf_know/100,g.w),1),"n child",g.child_mdd.notna().sum())
print("wealth PC1 share", round(d.wealth_pc1_share.iloc[0],3))
print("BMI missing cook", d.bmi_cook.isna().mean().round(3), "purch", d.bmi_purch.isna().mean().round(3), "adol among adol", d.loc[d.has_adol==1,"bmi_adol"].isna().mean().round(3))
