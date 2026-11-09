"""Figures from microdata analysis (Nature Food draft)."""
import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D=sys.argv[1]; OUT="/home/user/CSE366-4--2021-2-60-115/nature-food/figures/"
R=json.load(open("/home/user/CSE366-4--2021-2-60-115/nature-food/analysis/microdata_results.json"))
d=pd.read_pickle(f"{D}/analysis.pkl")
B,O,A="#2a78d6","#eb6834","#1baf7a"; INK,INK2,GRID="#0b0b0b","#52514e","#e4e3df"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.edgecolor":INK2,"axes.labelcolor":INK,"xtick.color":INK2,"ytick.color":INK2,
 "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6,"legend.frameon":False,"savefig.dpi":300})
def lab(ax,s,x=-0.14): ax.text(x,1.03,s,transform=ax.transAxes,fontsize=9,fontweight="bold",va="bottom")
def wm(x,w): m=x.notna(); return 100*np.average(x[m],weights=w[m])
ru=d[d["sample"]=="Rural-urban"]; sl=d[d["sample"]=="Slum"]
roles=[("purch","Food purchasers",B),("cook","Cooks",O),("adol","Adolescents",A)]

# ---------- Fig 1: knowledge gaps ----------
fig,axs=plt.subplots(1,2,figsize=(7.2,3.3),gridspec_kw={"width_ratios":[1.05,1]})
it=[("harms_ncd","Diet-related NCD risks of\nunhealthy foods"),("energy_foods","Energy-giving foods"),("trans_fat","Trans-fatty acids"),
    ("vitA_foods","Foods rich in vitamin A"),("arsenic_source","Source of arsenic"),("iron_foods","Foods rich in iron"),("heard_anaemia","Iron-deficiency anaemia")]
ax=axs[0]; y=np.arange(len(it)); h=0.26
for k,(r,n,c) in enumerate(roles):
    v=[R["items_RU"][r][i]["none"] for i,_ in it]
    ax.barh(y+(k-1)*h,v,height=h-0.04,color=c,label=n)
ax.set_yticks(y); ax.set_yticklabels([l for _,l in it]); ax.invert_yaxis(); ax.set_xlim(0,100)
ax.set_xlabel("Respondents giving no correct answer (%)"); ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=6.2,loc="lower center",bbox_to_anchor=(0.45,1.0),ncol=3); lab(ax,"a",-0.42)
ax=axs[1]
fs=[("ck_veg_wash_whole","Wash vegetables whole\nbefore cutting"),("fs_utensil_method","Clean utensils with soap/\nhot water after each use"),
    ("fs_reheat_once","Reheat cooked food\nonly once"),("fs_no_store_near_chem","Do not store food near\nchemicals"),
    ("fs_separate_raw_cooked","Store raw and cooked\nfood separately"),("fs_clean_hands_cooking","Clean hands\nbefore cooking")]
v1=[100-wm(ru[c],ru.w) for c,_ in fs]; v2=[100-wm(sl[c],sl.w) for c,_ in fs]; y=np.arange(len(fs))
ax.barh(y-0.18,v1,height=0.34,color=O,label="Rural–urban"); ax.barh(y+0.18,v2,height=0.34,color="white",edgecolor=O,hatch="////",lw=0.8,label="Slum")
for i,v in enumerate(v1): ax.text(v+1,i-0.18,f"{v:.0f}",va="center",fontsize=6)
ax.set_yticks(y); ax.set_yticklabels([l for _,l in fs]); ax.invert_yaxis(); ax.set_xlim(0,100)
ax.set_xlabel("Cooks not knowing the practice (%)"); ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); ax.legend(fontsize=6.5,loc="lower right"); lab(ax,"b",-0.4)
fig.tight_layout(); fig.savefig(OUT+"Fig1_knowledge_gaps.png"); plt.close()

# ---------- Fig 2: correlates of knowledge (coefficient plot) ----------
M=R["M1_knowledge_correlates_RU"]
terms=[("C(wealth_q)[T.5.0]","Richest vs poorest quintile"),("C(wealth_q)[T.3.0]","Middle vs poorest quintile"),("internet","Household internet access"),
       ("msg30_{r}","Saw nutrition/food-safety message (30 d)"),("urban","Urban vs rural"),("C(hfias_cat)[T.Moderate]","Moderate food insecurity"),("C(hfias_cat)[T.Severe]","Severe food insecurity")]
fig,ax=plt.subplots(figsize=(7.2,3.0)); y=np.arange(len(terms))
for k,(r,n,c) in enumerate(roles):
    for i,(t,_) in enumerate(terms):
        e=M[r][t.format(r=r)]; yy=i+(k-1)*0.24
        ax.plot([e["lo"],e["hi"]],[yy,yy],color=c,lw=1.6); ax.plot(e["est"],yy,"o",color=c,ms=5,label=n if i==0 else None)
ax.axvline(0,color=INK2,lw=0.7); ax.set_yticks(y); ax.set_yticklabels([l for _,l in terms]); ax.invert_yaxis()
ax.set_xlabel("Adjusted difference in basic nutrition knowledge score (points, 0–100; 95% CI)"); ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=6.5,loc="lower right")
fig.tight_layout(); fig.savefig(OUT+"Fig2_social_gradient.png"); plt.close()

# ---------- Fig 3: diets ----------
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),gridspec_kw={"width_ratios":[1.15,1]})
fg=[("grains_roots","Grains, roots, tubers"),("meat_fish","Meat, poultry, fish"),("other_veg","Other vegetables"),("pulses","Pulses"),("dglv","Dark green leafy veg."),
    ("eggs","Eggs"),("other_fruit","Other fruits"),("vitA_fv","Vitamin A-rich F&V"),("dairy","Dairy"),("nuts_seeds","Nuts and seeds")]
ax=axs[0]; y=np.arange(len(fg))
for k,(r,n,c) in enumerate(roles):
    v=[wm(ru[f"fg_{r}_{g}"],ru.w) for g,_ in fg]; ax.barh(y+(k-1)*0.26,v,height=0.22,color=c,label=n)
ax.set_yticks(y); ax.set_yticklabels([l for _,l in fg]); ax.invert_yaxis(); ax.set_xlim(0,105); ax.set_xlabel("Consumed in previous 24 h (%), rural–urban")
ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); ax.legend(fontsize=6.2,loc="lower right"); lab(ax,"a",-0.34)
ax=axs[1]; x=np.arange(3); w=0.24
for k,(z,n,c) in enumerate([("ssb","Sugar-sweetened beverages",B),("sweets","Sweets",O),("snacks","Savoury/fried snacks",A)]):
    v=[wm(ru[f"dz_{r}_{z}"],ru.w) for r,_,_ in roles]; vs=[wm(sl[f"dz_{r}_{z}"],sl.w) for r,_,_ in roles]
    ax.bar(x+(k-1)*w,v,width=w-0.03,color=c,label=n)
    ax.scatter(x+(k-1)*w,vs,marker="D",s=12,color="white",edgecolor=INK,lw=0.8,zorder=3,label="Slum" if k==0 else None)
ax.set_xticks(x); ax.set_xticklabels([n for _,n,_ in roles]); ax.set_ylim(0,100); ax.set_ylabel("Consumed in previous 24 h (%)")
ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); ax.legend(fontsize=6,loc="lower center",bbox_to_anchor=(0.5,1.0),ncol=2); lab(ax,"b",-0.18)
fig.tight_layout(); fig.savefig(OUT+"Fig3_diets.png"); plt.close()

# ---------- Fig 4: knowledge vs diet ----------
M2=R["M2_knowledge_diet"]["RU"]
fig,axs=plt.subplots(1,3,figsize=(7.2,3.1),gridspec_kw={"width_ratios":[1,1,1.05]})
ax=axs[0]
for k,(r,n,c) in enumerate(roles):
    for j,(key,ls) in enumerate([("dds_crude","o"),("dds_adj","s")]):
        e=M2[r][key]; yy=k+(j-0.5)*0.3
        ax.plot([e["lo"],e["hi"]],[yy,yy],color=c,lw=1.6); ax.plot(e["est"],yy,ls,color=c,ms=5,mfc="white" if j==0 else c)
ax.axvline(0,color=INK2,lw=0.7); ax.set_yticks(range(3)); ax.set_yticklabels([n for _,n,_ in roles]); ax.invert_yaxis()
ax.set_xlabel("Change in dietary diversity (food groups)\nper 10-point higher knowledge"); ax.set_xlim(-0.2,0.2)
ax.plot([],[],"o",color=INK2,mfc="white",label="Crude"); ax.plot([],[],"s",color=INK2,label="Adjusted"); ax.legend(fontsize=6,loc="lower center",bbox_to_anchor=(0.5,1.0),ncol=2); lab(ax,"a",-0.5)
ax=axs[1]
for k,(r,n,c) in enumerate(roles):
    e=M2[r]["disc_adj_OR"]; ax.plot([e["lo"],e["hi"]],[k,k],color=c,lw=1.6); ax.plot(e["est"],k,"s",color=c,ms=5)
ax.axvline(1,color=INK2,lw=0.7); ax.set_yticks(range(3)); ax.set_yticklabels(["","",""]); ax.invert_yaxis(); ax.set_xlim(0.8,1.45)
ax.set_xlabel("Adjusted odds ratio of discretionary\nfoods per 10-point higher knowledge"); lab(ax,"b",-0.08)
ax=axs[2]
for k,(r,n,c) in enumerate(roles):
    for j,(key,mk) in enumerate([("slope_secure_mild","o"),("slope_mod_severe","D")]):
        e=M2[r][key]; yy=k+(j-0.5)*0.3
        ax.plot([e["lo"],e["hi"]],[yy,yy],color=c,lw=1.6); ax.plot(e["est"],yy,mk,color=c,ms=5,mfc="white" if j==0 else c)
    p=M2[r]["interaction_p"]; ax.text(0.27,k,("P < 0.001" if p<0.001 else f"P = {p:.2f}"),fontsize=6,va="center",color=INK2)
ax.axvline(0,color=INK2,lw=0.7); ax.set_yticks(range(3)); ax.set_yticklabels(["","",""]); ax.invert_yaxis(); ax.set_xlim(-0.3,0.4)
ax.plot([],[],"o",color=INK2,mfc="white",label="Food-secure/mildly insecure"); ax.plot([],[],"D",color=INK2,label="Moderately/severely insecure")
ax.set_xlabel("Change in dietary diversity per 10-point\nhigher knowledge, by food security"); ax.legend(fontsize=5.8,loc="lower center",bbox_to_anchor=(0.5,1.0),ncol=1); lab(ax,"c",-0.08)
fig.tight_layout(); fig.savefig(OUT+"Fig4_knowledge_vs_diet.png"); plt.close()

# ---------- Fig 5: hygiene ----------
fig,axs=plt.subplots(1,2,figsize=(7.2,2.9))
ax=axs[0]; q=np.arange(1,6)
for s,c,mk,n in [("RU",B,"o","Rural–urban"),("Slum",A,"s","Slum")]:
    v=[R["foodsafety_practice"][s]["station_ws_by_wealth"][str(i)] for i in q]; ax.plot(q,v,color=c,lw=2,marker=mk,ms=5,label=n)
ax.set_xticks(q); ax.set_xticklabels(["Poorest","Poorer","Middle","Richer","Richest"]); ax.set_ylim(0,100)
ax.set_ylabel("Handwashing place with water and soap,\nobserved by enumerator (%)"); ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); ax.legend(fontsize=6.5,loc="upper left"); lab(ax,"a")
ax=axs[1]
ct=[("after_latrine","After defecation"),("before_eating","Before eating"),("before_cooking","Before cooking"),("after_returning_home","After returning home")]
y=np.arange(len(ct))
for off,s,c in [(-0.15,"RU",B),(0.15,"Slum",A)]:
    for i,(k,_) in enumerate(ct):
        h=R["handwashing"][s][k]; cur=h["report"]; pre=cur*h["pre_share"]/100
        ax.plot([pre,cur],[i+off,i+off],color=c,lw=2,alpha=0.55); ax.plot(pre,i+off,"o",ms=5,mfc="white",mec=c,mew=1.4); ax.plot(cur,i+off,"o",ms=5,color=c)
ax.plot([],[],"o",ms=5,mfc="white",mec=INK2,ls="",label="Before pandemic (recalled)"); ax.plot([],[],"o",ms=5,color=INK2,ls="",label="Current (Oct–Nov 2020)")
ax.plot([],[],color=B,lw=2,label="Rural–urban"); ax.plot([],[],color=A,lw=2,label="Slum")
ax.set_yticks(y); ax.set_yticklabels([l for _,l in ct]); ax.invert_yaxis(); ax.set_xlim(0,105)
ax.set_xlabel("Report washing hands with soap (%)"); ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=6,loc="lower center",bbox_to_anchor=(0.5,1.02),ncol=2); ax.text(-0.36,1.03,"b",transform=ax.transAxes,fontsize=9,fontweight="bold",va="bottom")
fig.tight_layout(); fig.savefig(OUT+"Fig5_handwashing.png"); plt.close()
print("figs ok")
