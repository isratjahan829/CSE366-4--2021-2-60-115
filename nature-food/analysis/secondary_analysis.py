"""Secondary analyses computed from published tables of the FAO/FPMU consumer awareness survey final report (2021).
No microdata are used; all inputs are weighted percentages, odds ratios and SEs printed in the report."""
import math, json
DEFF = 2.5           # design effect assumed in survey sample-size calculation (report §3.1.2)
Z = 1.959964
out = {}

# ---------- 1. Inequality indices (SII / RII) via ridit regression ----------
def ridits(shares):
    s = [x/sum(shares) for x in shares]; r=[]; c=0
    for x in s: r.append(c + x/2); c += x
    return s, r
def sii(prev, shares):
    w, x = ridits(shares)
    xb = sum(wi*xi for wi,xi in zip(w,x)); yb = sum(wi*yi for wi,yi in zip(w,prev))
    b = sum(wi*(xi-xb)*(yi-yb) for wi,xi,yi in zip(w,x,prev))/sum(wi*(xi-xb)**2 for wi,xi in zip(w,x))
    a = yb - b*xb
    return round(b,1), round((a+b)/a,2) if a>0 else None
edu_share = {"purchasers":[28.7,33.6,29.9,7.8],"cooks":[25.4,32.9,35.8,5.9],"adolescents":[1.9,38.6,57.3,2.2]}
edu_high  = {"purchasers":[13.1,27.8,45.0,66.4],"cooks":[13.7,29.1,43.8,69.7],"adolescents":[0.0,20.5,42.7,55.1]}
wq_high   = {"purchasers":[21.0,25.7,29.8,45.0,51.2],"cooks":[21.2,31.8,33.3,44.7,48.8],"adolescents":[28.8,32.2,34.9,38.0,42.4]}
edu_low   = {"purchasers":[55.4,37.7,20.3,6.9],"cooks":[55.5,34.4,22.8,5.5],"adolescents":[80.0,44.3,24.0,20.4]}
wq_low    = {"purchasers":[47.8,38.1,32.9,22.7,19.0],"cooks":[46.7,34.1,32.6,21.1,18.7],"adolescents":[37.9,34.3,32.0,27.0,24.6]}
ineq = {}
for g in edu_share:
    ineq[g] = {"edu_SII_high": sii(edu_high[g], edu_share[g]), "wealth_SII_high": sii(wq_high[g],[20]*5),
               "edu_SII_low": sii(edu_low[g], edu_share[g]), "wealth_SII_low": sii(wq_low[g],[20]*5)}
out["inequality"] = ineq

# ---------- 2. Knowledge–diet: risk differences with design-effect-adjusted CIs ----------
def rd(p1,n1,p0,n0):
    p1/=100; p0/=100
    se = math.sqrt(p1*(1-p1)/n1 + p0*(1-p0)/n0)*math.sqrt(DEFF)
    d = p1-p0
    return round(100*d,1), round(100*(d-Z*se),1), round(100*(d+Z*se),1)
N = {"purchasers":3989,"cooks":3989,"adolescents":2044}
inad = {"purchasers":57.3,"cooks":56.6,"adolescents":63.5}
unf  = {"purchasers":60.8,"cooks":64.0,"adolescents":78.3}
high_dd = {"purchasers":(35.9,30.8),"cooks":(37.9,32.2),"adolescents":(39.4,31.6)}   # adequate, inadequate
low_dd  = {"purchasers":(32.9,33.9),"cooks":(30.9,32.8),"adolescents":(23.1,37.1)}
high_nb = {"purchasers":(32.3,34.1),"cooks":(38.7,27.5),"adolescents":(40.0,36.2)}   # unfavourable(discretionary), favourable
kd = {}
for g in N:
    n_in = round(N[g]*inad[g]/100); n_ad = N[g]-n_in
    n_un = round(N[g]*unf[g]/100);  n_fa = N[g]-n_un
    kd[g] = {"n_adequate":n_ad,"n_inadequate":n_in,
             "RD_high_adequate_vs_inadequate": rd(high_dd[g][0],n_ad,high_dd[g][1],n_in),
             "RD_low_adequate_vs_inadequate":  rd(low_dd[g][0],n_ad,low_dd[g][1],n_in),
             "RD_high_discretionary_vs_not":   rd(high_nb[g][0],n_un,high_nb[g][1],n_fa)}
out["knowledge_diet_RD"] = kd

# ---------- 3. Pooled adjusted OR (high knowledge ~ adequate dietary diversity), inverse-variance ----------
ors = {"purchasers":(0.888,0.102),"cooks":(0.889,0.101),"adolescents":(1.036,0.162)}
w=[];l=[]
for g,(o,se) in ors.items():
    s = se/o; w.append(1/s**2); l.append(math.log(o))
lp = sum(wi*li for wi,li in zip(w,l))/sum(w); sp = 1/math.sqrt(sum(w))
Q = sum(wi*(li-lp)**2 for wi,li in zip(w,l)); I2 = max(0,(Q-2)/Q)*100 if Q>0 else 0
out["pooled_OR_dd"] = {"OR":round(math.exp(lp),2),"lo":round(math.exp(lp-Z*sp),2),"hi":round(math.exp(lp+Z*sp),2),
                       "Q":round(Q,2),"I2":round(I2,1)}
# minimum detectable OR (80% power, two-sided alpha .05) given each model's SE on log scale
out["MDE_OR"] = {g: (round(math.exp(-(Z+0.8416)*o[1]/o[0]),2), round(math.exp((Z+0.8416)*o[1]/o[0]),2)) for g,o in ors.items()}
out["MDE_pooled"] = (round(math.exp(-(Z+0.8416)*sp),2), round(math.exp((Z+0.8416)*sp),2))
# education OR for comparison (higher secondary, high vs low)
out["edu_vs_dd_ratio"] = {g: None for g in ors}

# ---------- 4. Handwashing: pre-pandemic, current, projected post-pandemic ----------
hw = {  # current %, usual-before %, adopted-due-to-COVID %, persist % (among adopters)
 "rural_urban": {"After defecation":(97.6,92.6,7.1,99.6),"Before eating":(96.5,80.9,18.8,99.7),
   "After cleaning child's faeces":(50.5,86.8,13.7,93.1),"Before cooking":(49.7,81.1,18.7,98.1),
   "After returning home":(45.7,44.3,54.3,98.6),"Before feeding a child":(42.0,76.7,29.7,100.0)},
 "slum": {"After defecation":(97.2,90.4,9.6,97.9),"Before eating":(96.6,73.5,26.1,100.0),
   "After cleaning child's faeces":(65.0,82.9,20.7,100.0),"Before cooking":(35.3,79.0,20.2,100.0),
   "After returning home":(46.0,28.2,70.1,99.4),"Before feeding a child":(65.0,74.3,30.8,100.0)}}
hwo = {}
for s,d in hw.items():
    hwo[s]={}
    for k,(c,u,n,p) in d.items():
        un = u+n
        u_, n_ = (u/un*100, n/un*100) if un>100 else (u,n)   # rescale where shares sum >100
        pre = c*u_/100; post = c*(u_+n_*p/100)/100
        hwo[s][k] = {"pre":round(pre,1),"current":c,"post":round(min(post,c),1),"gain_pp":round(c-pre,1),"shares_sum":round(un,1)}
out["handwashing"] = hwo

# ---------- 5. Slum vs rural–urban differences (design-effect-adjusted) ----------
def rd2(p_s,n_s,p_r,n_r): return rd(p_s,n_s,p_r,n_r)
sl = {
 "Inadequate dietary diversity, purchasers": (65.9,1011,57.3,3989),
 "Inadequate dietary diversity, cooks": (65.7,1011,56.6,3989),
 "Inadequate dietary diversity, adolescents": (70.2,429,63.5,2044),
 "Discretionary foods, purchasers": (66.4,1011,60.8,3989),
 "Discretionary foods, cooks": (76.2,1011,64.0,3989),
 "Discretionary foods, adolescents": (85.6,429,78.3,2044),
 "Child minimum dietary diversity (6–23 mo)": (31.0,100,40.7,481),
 "Child consumed dairy (6–23 mo)": (11.0,100,23.1,481),
 "Child consumed eggs (6–23 mo)": (28.0,100,40.9,481),
 "Handwashing with soap before cooking": (35.3,1011,49.7,3989),
 "Handwashing with soap before feeding a child": (65.0,1011,42.0,3989),
 "Handwashing with soap after cleaning child's faeces": (65.0,1011,50.5,3989),
}
out["slum_vs_ru"] = {k: {"slum":v[0],"ru":v[2],"RD":rd2(*v)} for k,v in sl.items()}

# ---------- 6. Division profile (share in lowest national tertile; 33.3% = national expectation) ----------
out["division_low"] = {
 "divisions":["Barisal","Chattogram","Dhaka","Khulna","Mymensingh","Rajshahi","Rangpur","Sylhet"],
 "Basic nutrition — purchasers":[37.5,32.6,30.9,31.3,33.3,36.4,29.2,46.1],
 "Basic nutrition — cooks":[34.4,33.3,28.9,30.4,28.3,41.4,23.5,41.1],
 "Basic nutrition — adolescents":[36.1,34.4,33.8,34.0,26.2,33.3,12.7,47.3],
 "Food safety & hygiene — cooks":[33.9,16.4,27.8,51.9,34.6,34.2,55.2,31.4],
 "Cooking practices — cooks":[54.4,49.4,40.5,56.3,36.8,19.2,50.2,41.1]}
json.dump(out, open("/home/user/CSE366-4--2021-2-60-115/nature-food/analysis/secondary_results.json","w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
