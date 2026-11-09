import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
OUT="/home/user/CSE366-4--2021-2-60-115/nature-food/figures/"
B,O,A="#2a78d6","#eb6834","#1baf7a"
INK,INK2,GRID="#0b0b0b","#52514e","#e4e3df"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.edgecolor":INK2,"axes.labelcolor":INK,
 "xtick.color":INK2,"ytick.color":INK2,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6,
 "xtick.major.width":0.6,"ytick.major.width":0.6,"legend.frameon":False,"savefig.dpi":300})
def lab(ax,s): ax.text(-0.14,1.04,s,transform=ax.transAxes,fontsize=9,fontweight="bold",va="bottom")

# ---------- Fig 1: knowledge gaps (item-level) ----------
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0),gridspec_kw={"width_ratios":[1.1,1]})
items=["Foods rich in vitamin C","Body-building foods","Body-protective foods","Energy-giving foods",
       "Foods rich in iron","Foods rich in vitamin A","Trans-fatty acids"]
inc=[36.6,5.3,14.6,59.8,59.6,61.2,64.1]   # food purchasers, rural-urban, incorrect %
inc_c=[35.3,6.0,14.7,60.6,59.2,60.7,66.7]
inc_a=[36.0,17.2,26.6,66.2,67.2,66.3,70.8]
y=np.arange(len(items)); h=0.26
ax=axs[0]
for k,(v,c,n) in enumerate([(inc,B,"Food purchasers"),(inc_c,O,"Cooks"),(inc_a,A,"Adolescents")]):
    ax.barh(y+(1-k)*h,v,height=h-0.04,color=c,label=n)
ax.set_yticks(y); ax.set_yticklabels(items); ax.set_xlim(0,80); ax.set_xlabel("Respondents answering incorrectly (%)")
ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
ax.legend(loc="lower right",fontsize=6.5); lab(ax,"a")
ax=axs[1]
fs=["Clean vegetables whole\nbefore cutting","Safe number of times\nto reheat cooked food","Do not store food near\nchemicals/cleaners",
    "Keep food covered\nafter cooking","Store raw and cooked\nfood separately","Wash hands before eating","Clean hands before cooking"]
fsv=[74.5,57.7,52.2,21.9,18.3,4.5,3.9]
ax.barh(np.arange(len(fs)),fsv,height=0.6,color=O)
for i,v in enumerate(fsv): ax.text(v+1,i,f"{v:.1f}",va="center",fontsize=6.5,color=INK)
ax.set_yticks(np.arange(len(fs))); ax.set_yticklabels(fs); ax.set_xlim(0,85); ax.invert_yaxis()
ax.set_xlabel("Cooks answering incorrectly (%)"); ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); lab(ax,"b")
fig.tight_layout(); fig.savefig(OUT+"Fig1_knowledge_gaps.png"); plt.close()

# ---------- Fig 2: social gradient (ORs, high vs low knowledge) ----------
fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
ax=axs[0]
edu=["Primary","Secondary","Higher\nsecondary+"]
orp=[3.93,9.75,41.38]; orc=[3.04,5.74,23.22]; ora=[None,5.05,28.57]
x=np.arange(3); w=0.26
for k,(v,c,n) in enumerate([(orp,B,"Food purchasers"),(orc,O,"Cooks"),(ora,A,"Adolescents")]):
    xs=[x[i]+(k-1)*w for i in range(3) if v[i] is not None]; vs=[t for t in v if t is not None]
    ax.bar(xs,vs,width=w-0.04,color=c,label=n)
    for xx,vv in zip(xs,vs): ax.text(xx,vv*1.08,f"{vv:.1f}",ha="center",fontsize=6,color=INK)
ax.set_yscale("log"); ax.set_ylim(1,80); ax.set_yticks([1,2,5,10,20,50]); ax.set_yticklabels(["1","2","5","10","20","50"]); ax.minorticks_off(); ax.axhline(1,color=INK2,lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(edu); ax.set_ylabel("Adjusted OR, high vs low knowledge\n(ref. no education; log scale)")
ax.legend(fontsize=6.5,loc="upper left"); lab(ax,"a")
ax=axs[1]
wq=["Poorest","Poorer","Middle","Richer","Richest"]
hp=[21.0,25.7,29.8,45.0,51.2]; hc=[21.2,31.8,33.3,44.7,48.8]; ha_=[28.8,32.2,34.9,38.0,42.4]
for v,c,n,m in [(hp,B,"Food purchasers","o"),(hc,O,"Cooks","s"),(ha_,A,"Adolescents","^")]:
    ax.plot(range(5),v,color=c,lw=2,marker=m,ms=5,label=n)
ax.set_xticks(range(5)); ax.set_xticklabels(wq); ax.set_ylim(0,60); ax.set_ylabel("High basic nutrition knowledge (%)")
ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); ax.legend(fontsize=6.5,loc="upper left"); lab(ax,"b")
fig.tight_layout(); fig.savefig(OUT+"Fig2_social_gradient.png"); plt.close()

# ---------- Fig 3: diets ----------
fig,axs=plt.subplots(1,2,figsize=(7.2,2.9),gridspec_kw={"width_ratios":[1.15,1]})
ax=axs[0]
fg=["Grains, roots, tubers","Meat, poultry, fish","Other vegetables","Pulses","Dark green leafy veg.",
    "Eggs","Other vitamin A-rich F&V","Other fruits","Dairy","Nuts and seeds"]
pv=[99.9,79.3,71.9,53.4,45.2,30.3,19.5,17.2,14.7,5.8]; av=[68.4,71.5,59.4,46.5,38.6,29.8,18.4,18.6,13.5,6.8]
yy=np.arange(len(fg))
ax.barh(yy-0.18,pv,height=0.34,color=B,label="Food purchasers")
ax.barh(yy+0.18,av,height=0.34,color=A,label="Adolescents")
ax.set_yticks(yy); ax.set_yticklabels(fg); ax.invert_yaxis(); ax.set_xlim(0,105)
ax.set_xlabel("Consumed in previous 24 h (%)"); ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=6.5,loc="lower right"); lab(ax,"a")
ax=axs[1]
grp=["Purchasers","Cooks","Adolescents"]
inad_ru=[57.3,56.6,63.5]; inad_sl=[65.9,65.7,70.2]; unf_ru=[60.8,64.0,78.3]; unf_sl=[66.4,76.2,85.6]
x=np.arange(3); w=0.19
for k,(v,c,n,hatch) in enumerate([(inad_ru,B,"Inadequate diversity, rural–urban",None),(inad_sl,B,"Inadequate diversity, slum","////"),
                                  (unf_ru,O,"Discretionary foods, rural–urban",None),(unf_sl,O,"Discretionary foods, slum","////")]):
    ax.bar(x+(k-1.5)*w,v,width=w-0.03,color=c if hatch is None else "white",edgecolor=c,hatch=hatch,lw=0.8,label=n)
ax.set_xticks(x); ax.set_xticklabels(grp); ax.set_ylim(0,100); ax.set_ylabel("Respondents (%)")
ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); ax.legend(fontsize=5.8,loc="upper left",ncol=1); lab(ax,"b")
fig.tight_layout(); fig.savefig(OUT+"Fig3_diets.png"); plt.close()

# ---------- Fig 4: knowledge vs diet ----------
fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
ax=axs[0]
x=np.arange(3); w=0.36
ina=[30.8,32.2,31.6]; ade=[35.9,37.9,39.4]
ax.bar(x-w/2,ina,width=w-0.04,color="#9fc3ee",label="Inadequate diversity (<5 groups)")
ax.bar(x+w/2,ade,width=w-0.04,color=B,label="Adequate diversity (≥5 groups)")
for i in range(3):
    ax.text(x[i]-w/2,ina[i]+1,f"{ina[i]:.1f}",ha="center",fontsize=6); ax.text(x[i]+w/2,ade[i]+1,f"{ade[i]:.1f}",ha="center",fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(grp); ax.set_ylim(0,60); ax.set_ylabel("High basic nutrition knowledge (%)")
ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); ax.legend(fontsize=6.2,loc="upper left"); lab(ax,"a")
ax=axs[1]
fav=[34.1,27.5,36.2]; unf=[32.3,38.7,40.0]
ax.bar(x-w/2,fav,width=w-0.04,color="#f6b597",label="No discretionary foods")
ax.bar(x+w/2,unf,width=w-0.04,color=O,label="Discretionary foods in previous 24 h")
pl=["P = 0.3","P < 0.001","P = 0.4"]
for i in range(3):
    ax.text(x[i]-w/2,fav[i]+1,f"{fav[i]:.1f}",ha="center",fontsize=6); ax.text(x[i]+w/2,unf[i]+1,f"{unf[i]:.1f}",ha="center",fontsize=6)
    ax.text(x[i],max(fav[i],unf[i])+6,pl[i],ha="center",fontsize=6,color=INK2)
ax.set_xticks(x); ax.set_xticklabels(grp); ax.set_ylim(0,60); ax.set_ylabel("High basic nutrition knowledge (%)")
ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); ax.legend(fontsize=6.2,loc="upper left"); lab(ax,"b")
fig.tight_layout(); fig.savefig(OUT+"Fig4_knowledge_vs_diet.png"); plt.close()

# ---------- Fig 5: handwashing ----------
fig,axs=plt.subplots(1,2,figsize=(7.2,2.9))
ct=["After defecation","Before eating","After cleaning\nchild's faeces","Before cooking","After returning\nhome","Before feeding\na child"]
ru=[97.6,96.5,50.5,49.7,45.7,42.0]; sl=[97.2,96.6,65.0,35.3,46.0,65.0]
cov=[7.1,18.8,13.7,18.7,54.3,29.7]; covs=[9.6,26.1,20.7,20.2,70.1,30.8]
yy=np.arange(len(ct))
for ax,(a,b,xl,cols,t) in zip(axs,[(ru,sl,"Report washing hands with soap (%)",(B,A),"a"),
                                  (cov,covs,"Practice attributed to COVID-19 messaging\n(% of those practising)",(B,A),"b")]):
    ax.barh(yy-0.18,a,height=0.34,color=cols[0],label="Rural–urban")
    ax.barh(yy+0.18,b,height=0.34,color=cols[1],label="Slum")
    ax.set_yticks(yy); ax.set_yticklabels(ct if t=="a" else [""]*len(ct)); ax.invert_yaxis(); ax.set_xlim(0,105)
    ax.set_xlabel(xl); ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True); ax.legend(fontsize=6.5,loc="lower right"); lab(ax,t)
fig.tight_layout(); fig.savefig(OUT+"Fig5_handwashing.png"); plt.close()
print("ok")
