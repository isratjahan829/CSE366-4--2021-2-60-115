"""Construct analysis variables from the raw FAO/FPMU consumer awareness survey (Stata file).
Usage: python -I 01_construct.py <data_dir>   (expects clean.pkl with identifiers removed and S_weight merged)"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
D = sys.argv[1]
df = pd.read_pickle(f"{D}/clean.pkl")
num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)
def yes(c, code=1): x = num(c); return (x == code).astype(float).where(x.notna())
def frac(prefix, opts, base):
    """fractional credit: share of correct options mentioned; NaN if parent question missing"""
    present = [f"{prefix}{o}" for o in opts if f"{prefix}{o}" in df]
    s = sum(num(c).fillna(0).clip(0, 1) for c in present) / len(opts)
    return s.where(num(base).notna() | sum(num(c).notna() for c in present).gt(0))

# ---------------- Sample identifiers ----------------
out = pd.DataFrame(index=df.index)
out["id"] = df["qidno"]; out["psu"] = df["PSU"]; out["w"] = df["S_weight"]
out["area"] = num("area").map({1: "Rural", 2: "Urban", 3: "Slum"})
divmap = {1:"Barishal",2:"Chattogram",3:"Dhaka",4:"Khulna",5:"Mymensingh",6:"Rajshahi",7:"Rangpur",8:"Sylhet"}
import pickle; meta = pickle.load(open(f"{D}/meta.pkl", "rb"))
out["division"] = df["division"].map(meta["vl"]["division"]).replace({"Chittagong": "Chattogram"})
out["sample"] = np.where(out.area == "Slum", "Slum", "Rural-urban")
out["stratum"] = out.division + "_" + out.area
out["has_child"] = num("module_c").eq(1).astype(int)
out["has_adol"] = num("module_e").eq(1).astype(int)

# ---------------- Basic nutrition knowledge (cook B1, purchaser D1, adolescent E1) ----------------
def basic_nutrition(p):
    it = {}
    it["foodgroups"] = frac(f"{p}1_1", range(1, 9), f"{p}1_1")
    sal = sum(num(f"{p}1_4{k}").fillna(0) for k in (1, 2, 3)).gt(0)
    swe = sum(num(f"{p}1_4{k}").fillna(0) for k in (4, 5, 6, 7)).gt(0)
    ssb = sum(num(f"{p}1_4{k}").fillna(0) for k in (8, 9, 10, 11, 12, 13)).gt(0)
    it["unhealthy_foods"] = ((sal.astype(int) + swe + ssb) / 3).where(num(f"{p}1_4").notna() | num(f"{p}1_41").notna())
    it["harms_ncd"] = frac(f"{p}1_7", (4, 6, 8, 9), f"{p}1_7")
    it["balanced_needed"] = yes(f"{p}1_9")
    it["energy_foods"] = frac(f"{p}1_11", (1, 6), f"{p}1_11_A")
    it["bodybuilding_foods"] = frac(f"{p}1_12", (2, 3), f"{p}1_12_A")
    it["protective_foods"] = frac(f"{p}1_13", (4, 5), f"{p}1_13_A")
    it["vitC_foods"] = num(f"{p}1_141").clip(0, 1).where(num(f"{p}1_14_A").notna() | num(f"{p}1_141").notna())
    it["heard_anaemia"] = yes(f"{p}1_15_A")
    it["vitA_foods"] = frac(f"{p}1_16", (1, 2, 3), f"{p}1_16_A")
    iron = frac(f"{p}1_17", (2, 3, 4), f"{p}1_17_A")
    it["iron_foods"] = (iron - num(f"{p}1_171").fillna(0) / 3).clip(0, 1).where(iron.notna())
    it["arsenic_source"] = num(f"{p}1_181").clip(0, 1).where(num(f"{p}1_18_A").notna() | num(f"{p}1_181").notna())
    it["pregnancy_diet"] = frac(f"{p}1_19", range(1, 7), f"{p}1_19")
    it["lactation_diet"] = frac(f"{p}1_20", range(1, 7), f"{p}1_20")
    if f"{p}1_222" in df:
        tf = (num(f"{p}1_222").fillna(0) + num(f"{p}1_224").fillna(0)).gt(0).astype(float)
    else:
        tf = num(f"{p}1_22").isin([2, 4]).astype(float)
    it["trans_fat"] = tf.where(num(f"{p}1_21").notna())   # not heard -> 0
    it["healthy_diet"] = frac(f"{p}1_23", range(1, 13), f"{p}1_23")
    items = pd.DataFrame(it)
    return items
items = {}
for role, p in [("cook", "B"), ("purch", "D"), ("adol", "E")]:
    it = basic_nutrition(p)
    if role == "adol": it = it.where(out.has_adol.eq(1))
    items[role] = it
    out[f"bn_{role}"] = 100 * it.mean(axis=1, skipna=True).where(it.notna().sum(axis=1) >= 14)
    for c in it: out[f"bn_{role}_{c}"] = it[c]

# ---------------- Cooking practice knowledge (cook) ----------------
ck = pd.DataFrame({
 "gruel_keep": num("B1_24").eq(2), "veg_wash_whole": num("B1_25").eq(2), "boil_water_keep": num("B1_26").eq(2),
 "lid_on": num("B1_27").eq(1), "eat_within_1h": num("B1_28").isin([1, 2]), "cover_food": num("B1_29").eq(1),
 "no_stale_food": num("B1_30").eq(2), "iodised_salt": num("B1_37").isin([2, 3])}).astype(float)
out["cooking_know"] = 100 * ck.mean(axis=1)
for c in ck: out[f"ck_{c}"] = ck[c]

# ---------------- Food safety & hygiene knowledge (cook) ----------------
fs = pd.DataFrame({
 "no_store_near_chem": yes("B1_48_A"), "clean_hands_cooking": yes("B1_50"), "wash_before_eating": yes("B1_52_A"),
 "separate_raw_cooked": yes("B1_54_A"),
 "veg_safety": frac("B1_56", (2, 3, 4), "B1_56_A"), "fruit_safety": frac("B1_57", (2, 3, 4), "B1_57_A"),
 "cooked_food_safety": frac("B1_58", range(1, 6), "B1_58_A"), "clean_utensils": yes("B1_60"),
 "utensil_method": num("B1_62_A").isin([1, 8]).astype(float), "disease_sources": frac("B1_63", range(1, 7), "B1_63_A"),
 "reheat_once": num("B1_66").eq(1).astype(float), "prevent_contamination": frac("B1_67", range(1, 6), "B1_67")})
out["foodsafety_know"] = 100 * fs.mean(axis=1)
for c in fs: out[f"fs_{c}"] = fs[c]

# ---------------- IYCF knowledge (caregiver) ----------------
cg = out.has_child.eq(1)
iy = pd.DataFrame({
 "bf_after_birth": yes("C1_5"), "bf_initiation_1h": num("C1_6").isin([1, 2]).astype(float),
 "colostrum": num("C1_7").eq(2).astype(float), "bf_on_demand": num("C1_8").isin([1, 2]).astype(float),
 "exclusive_bf": num("C1_9").eq(1).astype(float), "insufficient_milk": num("C1_101").clip(0, 1),
 "no_water_lt6m": num("C1_11").eq(2).astype(float), "liquids_at_6m": num("C1_12").between(6, 6.99).astype(float),
 "foods_at_6m": num("C1_13").between(6, 6.99).astype(float), "bf_to_24m": num("C1_14").between(24, 60).astype(float),
 "no_food_3days": num("C1_15").eq(2).astype(float)}).where(cg)
out["iycf_know"] = 100 * iy.mean(axis=1)
for c in iy: out[f"iy_{c}"] = iy[c]

# ---------------- Diets (24 h) ----------------
def dds(p, child=False):
    a = lambda k: num(f"{p}_{k}a").eq(1)
    g = {"grains_roots": a(1) | a(2), "pulses": a(3), "nuts_seeds": a(4), "dairy": a(17),
         "meat_fish": a(13) | a(14) | a(15) | a(16) | a(18), "eggs": a(12), "dglv": a(5),
         "vitA_fv": a(6) | a(7), "other_veg": a(9) | a(10), "other_fruit": a(8) | a(11)}
    G = pd.DataFrame(g).astype(int)
    disc = pd.DataFrame({"snacks": a(20), "sweets": a(21), "ssb": a(22) | a(23)}).astype(int)
    valid = num(f"{p}_1a").notna()
    return G.where(valid), disc.where(valid)
for role, p in [("cook", "B4"), ("purch", "D5"), ("adol", "E5")]:
    G, Dd = dds(p)
    if role == "adol": G, Dd = G.where(out.has_adol.eq(1)), Dd.where(out.has_adol.eq(1))
    out[f"dds_{role}"] = G.sum(axis=1, min_count=1)
    out[f"mdd_{role}"] = (out[f"dds_{role}"] >= 5).astype(float).where(out[f"dds_{role}"].notna())
    out[f"disc_{role}"] = Dd.sum(axis=1, min_count=1)
    out[f"anydisc_{role}"] = (out[f"disc_{role}"] > 0).astype(float).where(out[f"disc_{role}"].notna())
    for c in G: out[f"fg_{role}_{c}"] = G[c]
    for c in Dd: out[f"dz_{role}_{c}"] = Dd[c]
# child (6-23 m): 8 WHO groups
a = lambda k: num(f"C3_{k}a").eq(1)
bm = num("C1_17").eq(1) | num("C1_22_1a").eq(1)
milk = num("C1_22_4a").eq(1) | num("C1_22_5a").eq(1) | num("C1_22_6a").eq(1) | num("C1_22_8a").eq(1) | a(17)
CG = pd.DataFrame({"breastmilk": bm, "grains_roots": a(1) | a(2), "legumes_nuts": a(3) | a(4), "dairy": milk,
                   "flesh": a(13) | a(14) | a(15) | a(16) | a(18), "eggs": a(12), "vitA_fv": a(5) | a(6) | a(7),
                   "other_fv": a(8) | a(9) | a(10) | a(11)}).astype(int).where(cg & num("C3_1a").notna())
out["child_age_m"] = num("C1_3").where(cg)
out["child_dds"] = CG.sum(axis=1, min_count=1)
out["child_mdd"] = (out.child_dds >= 5).astype(float).where(out.child_dds.notna())
for c in CG: out[f"cfg_{c}"] = CG[c]

# ---------------- Household food insecurity (HFIAS) ----------------
H = pd.DataFrame({i: num(f"A4_{i}") for i in range(1, 10)})
out["hfias_score"] = H.sum(axis=1, min_count=9)
q = H
sev = (q[5].eq(3) | q[6].eq(3) | (q[7] > 0) | (q[8] > 0) | (q[9] > 0))
mod = (q[3].isin([2, 3]) | q[4].isin([2, 3]) | q[5].isin([1, 2]) | q[6].isin([1, 2])) & ~sev
mild = (q[1].isin([2, 3]) | q[2].isin([1, 2, 3]) | q[3].eq(1) | q[4].eq(1)) & ~sev & ~mod
cat = np.select([sev, mod, mild], ["Severe", "Moderate", "Mild"], "Secure")
out["hfias_cat"] = pd.Series(cat, index=df.index).where(H.notna().all(axis=1))
out["food_insecure_modsev"] = out.hfias_cat.isin(["Moderate", "Severe"]).astype(float).where(out.hfias_cat.notna())

# ---------------- Wealth index (PCA on assets & housing) ----------------
assets = pd.DataFrame({f"asset{i}": num(f"A2_9_{i}").eq(1).astype(int) for i in range(1, 23)})
assets["wall_finished"] = num("A2_10wal").ge(8).astype(int)
assets["roof_finished"] = num("A2_11roof").ge(6).astype(int)
assets["floor_finished"] = num("A2_12flor").ge(4).astype(int)
assets["rooms"] = num("A2_13room").clip(upper=10).fillna(num("A2_13room").median())
assets["clean_fuel"] = num("A2_3_7").isin([1, 2, 3, 10]).astype(int)
assets["piped_water"] = num("A3_1").isin([1, 2, 3]).astype(int)
assets["improved_toilet"] = num("A3_4").isin([1, 2, 3, 5]).astype(int)
assets["toilet_not_shared"] = num("A3_5").eq(1).astype(int)
assets["land_homestead"] = np.log1p(num("A2_3_4").clip(lower=0).fillna(0))
assets["land_cultivable"] = np.log1p(num("A2_3_5").clip(lower=0).fillna(0))
for k in (1, 2, 3): assets[f"livestock{k}"] = num(f"A2_8_a{k}").eq(1).astype(int)
X = assets.loc[:, assets.std() > 0]; Z = (X - X.mean()) / X.std()
vals, vecs = np.linalg.eigh(np.cov(Z.T)); v = vecs[:, -1]
if np.corrcoef(Z @ v, assets["asset8"])[0, 1] < 0: v = -v
out["wealth_score"] = Z @ v
out["wealth_pc1_share"] = vals[-1] / vals.sum()
def wq(g):
    s = g.wealth_score; w = g.w
    order = s.sort_values().index; cw = w.loc[order].cumsum() / w.sum()
    return pd.Series(np.ceil(cw * 5).clip(1, 5).astype(int), index=order)
out["wealth_q"] = pd.concat([wq(g) for _, g in out.groupby("sample")]).reindex(out.index)

# ---------------- Socio-economic covariates ----------------
out["internet"] = yes("A2_3_1")
out["hh_size"] = num("A21").clip(1, 25)
inc = num("A2_19m_tk").where(lambda x: (x > 0) & (x < 1e6))
fexp = num("A2_21m_tk").where(lambda x: (x > 0) & (x < 5e5)); texp = num("A2_20m_tk").where(lambda x: (x > 0) & (x < 1e6))
out["log_food_exp_pc"] = np.log(fexp / out.hh_size)
out["food_share"] = (fexp / texp).where(lambda x: x <= 1)
out["log_income_pc"] = np.log(inc / out.hh_size)
out["safety_net"] = (1 - num("A2_14gh1").fillna(0).clip(0, 1)).where(num("A2_14gh").notna() | num("A2_14gh1").notna())
for role, p in [("cook", "B2"), ("purch", "D3"), ("adol", "E2")]:
    out[f"msg30_{role}"] = yes(f"{p}_3")

# ---------------- WASH: observed & reported ----------------
out["hw_station"] = num("A3_9").isin([1, 2, 3]).astype(float)
out["hw_station_water_soap"] = (out.hw_station.eq(1) & num("A3_10").eq(1) & num("A3_11").eq(1)).astype(float)
out["soap_in_hh_obs"] = yes("A3_12")
out["toilet_clean_obs"] = yes("A3_6")
out["food_near_chem_obs"] = yes("B1_49_A")
out["improved_water"] = num("A3_1").isin([1, 2, 3, 4, 5, 6, 8, 11, 13]).astype(float)
hwk = {1:"before_eating",2:"after_latrine",3:"before_feeding_baby",4:"after_child_faeces",5:"before_cooking",10:"after_returning_home"}
for k, n in hwk.items():
    out[f"hw_rep_{n}"] = num(f"A3_16{k}").clip(0, 1)
    out[f"hw_know_{n}"] = num(f"A3_13{k}").clip(0, 1)
# COVID matrix (rows 1..12 follow A3_16 option order)
for k, n in hwk.items():
    out[f"hw_pre_{n}"] = yes(f"A3_17c{k}"); out[f"hw_covid_{n}"] = yes(f"A3_17d{k}"); out[f"hw_cont_{n}"] = yes(f"A3_17e{k}")

# ---------------- Anthropometry & health ----------------
def meas(p, k):
    m = pd.concat([num(f"{p}_{k}{x}") for x in "abc"], axis=1)
    return m.median(axis=1)
for role, p, hp in [("cook", "B5", "B6"), ("purch", "D6", "D6"), ("adol", "E6", "E6")]:
    h = meas(p, 7); wt = meas(p, 8)
    h = h.where(h.between(120, 200)); wt = wt.where(wt.between(20, 150))
    out[f"height_{role}"] = h; out[f"weight_{role}"] = wt
    out[f"bmi_{role}"] = (wt / (h / 100) ** 2).where(lambda x: x.between(12, 60))
    hs = num(f"{hp}_22") if role != "cook" else num("B6_22")
    out[f"srh_{role}"] = hs.where(hs.between(1, 5))
    out[f"eqvas_{role}"] = num(f"{p}_21" if role == "cook" else f"{p}_21").where(lambda x: x.between(0, 100))
for role in ("cook", "purch"):
    b = out[f"bmi_{role}"]
    out[f"bmicat_{role}"] = pd.cut(b, [0, 18.5, 25, 30, 100], labels=["Underweight", "Normal", "Overweight", "Obese"], right=False)

out.to_pickle(f"{D}/analysis.pkl")
print(out.shape)
