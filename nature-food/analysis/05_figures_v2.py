"""Editorial-style figures for the Nature Food manuscript (v2).
Usage: python -I 05_figures_v2.py <data_dir>
Reads <data_dir>/analysis.pkl and analysis/microdata_results.json; writes figures/Fig1..Fig5 PNG (and PDF)."""
import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import statsmodels.formula.api as smf, statsmodels.api as sm

D = sys.argv[1]
ROOT = "/home/user/CSE366-4--2021-2-60-115/nature-food/"
OUT = ROOT + "figures/"
R = json.load(open(ROOT + "analysis/microdata_results.json"))
d = pd.read_pickle(f"{D}/analysis.pkl")
d["urban"] = (d.area == "Urban").astype(int)
d["hfias_cat"] = pd.Categorical(d.hfias_cat, ["Secure", "Mild", "Moderate", "Severe"])
d["wealth_q"] = d.wealth_q.astype(float)
for r in ("cook", "purch", "adol"): d[f"bn10_{r}"] = d[f"bn_{r}"] / 10
ru = d[d["sample"] == "Rural-urban"].copy(); sl = d[d["sample"] == "Slum"].copy()

# ---------------- design system ----------------
INK, INK2, MUTED, TRACK, RULE = "#1c2230", "#4a5060", "#8a8f9c", "#efede8", "#d9d6cf"
PU, CO, AD = "#3366b0", "#dd5a2f", "#1f9e89"
ROLE = {"purch": ("Food purchasers", PU), "cook": ("Cooks", CO), "adol": ("Adolescents", AD)}
plt.rcParams.update({
    "font.family": "Inter", "font.size": 6.8, "axes.titlesize": 7.5, "axes.labelsize": 6.8,
    "axes.edgecolor": RULE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "xtick.labelsize": 6.3, "ytick.labelsize": 6.6, "axes.linewidth": 0.6, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "legend.frameon": False, "savefig.dpi": 400, "figure.dpi": 100, "lines.solid_capstyle": "round",
})
def panel(ax, s, x=-0.02, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=INK, va="bottom", ha="right")
def kicker(ax, txt, x=0.0, y=1.06):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=7.2, fontweight="semibold", color=INK, va="bottom")
def head(ax, letter, title, x=0.0, y=1.06, sub=None):
    fig_w = ax.figure.get_size_inches()[0]; aw = ax.get_position().width * fig_w
    off = 0.15 / aw
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=INK, va="bottom", ha="left")
    ax.text(x + off, y, title, transform=ax.transAxes, fontsize=7.2, fontweight="semibold", color=INK, va="bottom", ha="left")
    if sub: ax.text(x + off, y - 0.06, sub, transform=ax.transAxes, fontsize=5.8, color=MUTED, va="bottom", ha="left")
def vgrid(ax, ticks):
    ax.set_xticks(ticks)
    for t in ticks: ax.axvline(t, color=TRACK, lw=0.8, zorder=0)
def wm(x, w): m = x.notna(); return 100 * np.average(x[m], weights=w[m])
def save(fig, name):
    fig.savefig(OUT + name + ".png", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(OUT + name + ".pdf", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)

# ======================= Fig 1: knowledge gaps =======================
def dotrow(ax, rows, series, xmax=100, label_first=True, fmt="{:.0f}"):
    """rows: list of labels; series: list of (name, color, values, filled)"""
    n = len(rows)
    for i in range(n):
        ax.add_patch(Rectangle((0, i - 0.17), xmax, 0.34, color=TRACK, lw=0, zorder=0))
        vals = [s[2][i] for s in series]
        ax.plot([min(vals), max(vals)], [i, i], color=INK2, lw=0.9, zorder=1, alpha=0.55)
        for name, c, v, filled in series:
            ax.scatter(v[i], i, s=34, color=c if filled else "white", edgecolor=c, linewidth=1.3, zorder=3)
    ax.set_yticks(range(n)); ax.set_yticklabels(rows); ax.set_ylim(n - 0.5, -0.9); ax.set_xlim(0, xmax)
    ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x", pad=1)
    if label_first:
        placed = sorted([(s[2][0], s[0], s[1]) for s in series])
        # stagger labels above first row to avoid collisions
        for k, (v, name, c) in enumerate(placed):
            ax.annotate(name, (v, 0), xytext=(v, -0.62 - 0.0 * k), ha="center", fontsize=6, color=c, fontweight="semibold",
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.5, shrinkA=0, shrinkB=4))

fig = plt.figure(figsize=(7.2, 3.45))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.78)
ax = fig.add_subplot(gs[0])
items = [("harms_ncd", "NCD risks of unhealthy foods"), ("trans_fat", "Trans-fatty acids are harmful"), ("energy_foods", "Energy-giving foods"),
         ("vitA_foods", "Vitamin A-rich foods"), ("arsenic_source", "Shallow tubewells carry arsenic"), ("iron_foods", "Iron-rich foods"),
         ("heard_anaemia", "Iron-deficiency anaemia (heard of)"), ("protective_foods", "Protective foods")]
series = [(ROLE[r][0], ROLE[r][1], [R["items_RU"][r][k]["none"] for k, _ in items], True) for r in ("purch", "cook", "adol")]
# spread first-row labels manually
dotrow(ax, [l for _, l in items], series, label_first=False)
for k, r in enumerate(("purch", "cook", "adol")):
    ax.scatter([2, 46, 68][k], -0.75, s=20, color=ROLE[r][1], clip_on=False)
    ax.text([4.5, 48.5, 70.5][k], -0.75, ROLE[r][0], va="center", fontsize=6, color=ROLE[r][1], fontweight="semibold")
vgrid(ax, [0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100%"])
ax.set_xlabel("Respondents with no correct answer", labelpad=3)
head(ax, "a", "Most cannot name a single nutrient source", x=-0.66, y=1.07)

ax = fig.add_subplot(gs[1])
fsi = [("ck_veg_wash_whole", "Wash vegetables whole, then cut"), ("fs_utensil_method", "Soap or hot water for utensils"),
       ("fs_reheat_once", "Reheat cooked food only once"), ("fs_no_store_near_chem", "Keep food away from chemicals"),
       ("fs_separate_raw_cooked", "Store raw and cooked apart"), ("fs_clean_hands_cooking", "Clean hands before cooking")]
v_ru = [100 - wm(ru[c], ru.w) for c, _ in fsi]; v_sl = [100 - wm(sl[c], sl.w) for c, _ in fsi]
dotrow(ax, [l for _, l in fsi], [("Rural–urban", CO, v_ru, True), ("Slum", CO, v_sl, False)], label_first=False)
for i, (a_, b_) in enumerate(zip(v_ru, v_sl)):
    ax.text(max(a_, b_) + 3.5, i, f"{a_:.0f}%", va="center", fontsize=6, color=INK, fontweight="semibold")
ax.scatter([], [], s=30, color=CO, label="Rural–urban"); ax.scatter([], [], s=30, color="white", edgecolor=CO, linewidth=1.3, label="Slum")
ax.legend(loc="lower right", fontsize=6, handletextpad=0.2, borderaxespad=0.1, bbox_to_anchor=(1.02, -0.02))
vgrid(ax, [0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100%"])
ax.set_xlabel("Cooks not knowing the practice", labelpad=3)
head(ax, "b", "Hygiene in principle, not in practice", x=-0.66, y=1.07)
save(fig, "Fig1_knowledge_gaps")

# ======================= Fig 2: correlates + division hex map =======================
fig = plt.figure(figsize=(7.2, 3.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.08)
ax = fig.add_subplot(gs[0])
M1 = R["M1_knowledge_correlates_RU"]
groups = [("RESOURCES", [("C(wealth_q)[T.5.0]", "Richest vs poorest fifth"), ("C(wealth_q)[T.3.0]", "Middle vs poorest fifth"), ("urban", "Urban vs rural")]),
          ("INFORMATION", [("internet", "Internet in household"), ("msg30_{r}", "Saw a nutrition message (30 d)")]),
          ("HARDSHIP", [("C(hfias_cat)[T.Moderate]", "Moderate food insecurity"), ("C(hfias_cat)[T.Severe]", "Severe food insecurity")])]
y = 0; ylabels = []; ypos = []
for gi, (gname, terms) in enumerate(groups):
    ax.text(-11.9, y - 0.15, gname, fontsize=5.6, color=MUTED, fontweight="semibold", va="bottom")
    y += 0.55
    for t, lab in terms:
        ax.add_patch(Rectangle((-11.9, y - 0.38), 26, 0.76, color=TRACK if len(ylabels) % 2 == 0 else "white", lw=0, zorder=0))
        for k, r in enumerate(("purch", "cook", "adol")):
            e = M1[r][t.format(r=r)]; yy = y + (k - 1) * 0.24
            ax.plot([e["lo"], e["hi"]], [yy, yy], color=ROLE[r][1], lw=1.4, alpha=0.9, zorder=2)
            ax.scatter(e["est"], yy, s=18, color=ROLE[r][1], zorder=3, edgecolor="white", linewidth=0.5)
        ylabels.append(lab); ypos.append(y); y += 1
    y += 0.25
ax.axvline(0, color=INK2, lw=0.7, zorder=1)
ax.set_yticks(ypos); ax.set_yticklabels(ylabels); ax.set_ylim(y - 0.4, -0.6); ax.set_xlim(-11.9, 14.1)
ax.set_xticks([-10, -5, 0, 5, 10]); ax.set_xticklabels(["−10", "−5", "0", "+5", "+10"])
ax.set_xlabel("Adjusted difference in knowledge score (points of 100)", labelpad=3)
for k, r in enumerate(("purch", "cook", "adol")):
    ax.text(1.0, 1.015 - k * 0.0, "", transform=ax.transAxes)
lx = 0.98
for k, r in enumerate(("adol", "cook", "purch")):
    ax.text(lx, 0.04 + k * 0.055, ROLE[r][0], transform=ax.transAxes, ha="right", fontsize=6, color=ROLE[r][1], fontweight="semibold")
head(ax, "a", "Knowledge follows money, connectivity and messaging", x=-0.47, y=1.03)

# hex tile map of divisions (schematic geography)
ax = fig.add_subplot(gs[1]); ax.set_aspect("equal"); ax.axis("off")
fsd = M1["foodsafety"]
vals = {"Barishal": 0.0}
for k, v in fsd.items():
    if k.startswith("C(division)[T."): vals[k[14:-1]] = v["est"]
pos = {"Rangpur": (0, 3), "Mymensingh": (1, 3), "Sylhet": (2, 3), "Rajshahi": (0.5, 2), "Dhaka": (1.5, 2), "Khulna": (0, 1), "Barishal": (1, 1), "Chattogram": (2, 1)}
cmap = LinearSegmentedColormap.from_list("div", ["#c2461f", "#ef9a76", "#f3efe8", "#8fb2df", "#2c5aa0"])
norm = TwoSlopeNorm(vmin=-9, vcenter=0, vmax=4)
rh = 0.56
for name, (cx, cy) in pos.items():
    x0, y0 = cx * 1.0, cy * 0.87
    v = vals[name]; col = cmap(norm(v))
    ax.add_patch(RegularPolygon((x0, y0), numVertices=6, radius=rh, orientation=0, facecolor=col, edgecolor="white", lw=2))
    dark = abs(v) > 5
    ax.text(x0, y0 + 0.08, name, ha="center", va="center", fontsize=5.6, color="white" if dark else INK, fontweight="semibold")
    ax.text(x0, y0 - 0.15, "ref." if name == "Barishal" else f"{v:+.1f}".replace("-", "−"), ha="center", va="center", fontsize=6.4,
            color="white" if dark else INK, fontweight="bold")
ax.set_xlim(-0.7, 2.7); ax.set_ylim(-0.05, 3.75)
sm_ = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cax = ax.inset_axes([0.14, 0.07, 0.72, 0.028]); cb = fig.colorbar(sm_, cax=cax, orientation="horizontal")
cb.outline.set_visible(False); cb.set_ticks([-8, -4, 0, 4]); cb.ax.set_xticklabels(["−8", "−4", "0", "+4"], fontsize=5.6); cb.ax.tick_params(length=0, pad=1)
ax.text(0.5, -0.02, "Food-safety knowledge vs Barishal (adjusted points)", transform=ax.transAxes, ha="center", fontsize=5.8, color=INK2)
head(ax, "b", "Food safety lags in the west", x=0.02, y=1.03)
ax.text(0.08, 0.975, "Schematic tile map; positions approximate geography", transform=ax.transAxes, fontsize=5.4, color=MUTED, va="bottom")
save(fig, "Fig2_social_gradient")

# ======================= Fig 3: diets =======================
fig = plt.figure(figsize=(7.2, 3.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.5)
ax = fig.add_subplot(gs[0])
fg = [("grains_roots", "Grains, roots, tubers"), ("other_veg", "Other vegetables"), ("meat_fish", "Meat, poultry, fish"), ("pulses", "Pulses"),
      ("dglv", "Dark green leafy veg."), ("eggs", "Eggs"), ("vitA_fv", "Vitamin A-rich fruit & veg."), ("other_fruit", "Other fruits"),
      ("dairy", "Dairy"), ("nuts_seeds", "Nuts and seeds")]
series = [(ROLE[r][0], ROLE[r][1], [wm(ru[f"fg_{r}_{g}"], ru.w) for g, _ in fg], True) for r in ("purch", "cook", "adol")]
dotrow(ax, [l for _, l in fg], series, label_first=False)
for (name, c, v, _), dy in zip(series, (0, 0, 0)):
    pass
ax.text(101, -0.72, "", fontsize=5)
for k, r in enumerate(("purch", "cook", "adol")):
    ax.scatter([2, 46, 68][k], -0.95, s=20, color=ROLE[r][1], clip_on=False)
    ax.text([4.5, 48.5, 70.5][k], -0.95, ROLE[r][0], va="center", fontsize=6, color=ROLE[r][1], fontweight="semibold")
ax.axvspan(0, 33.3, color="#fbe9e2", alpha=0.0)
vgrid(ax, [0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100%"])
ax.set_xlabel("Ate the food group yesterday (rural–urban)", labelpad=3)
head(ax, "a", "Rice and fish, little else", x=-0.55, y=1.075)

# waffles: discretionary food consumption
ax = fig.add_subplot(gs[1]); ax.axis("off"); ax.set_xlim(0, 3.3); ax.set_ylim(-0.25, 2.55)
rows_ = [("Rural–urban", ru), ("Slum", sl)]
for ci_, r in enumerate(("purch", "cook", "adol")):
    for ri, (sname, g) in enumerate(rows_):
        p = wm(g[f"anydisc_{r}"], g.w); k = int(round(p))
        x0, y0 = ci_ * 1.1, 1.35 - ri * 1.3
        for j in range(100):
            cx, cy = x0 + (j % 10) * 0.085, y0 + (j // 10) * 0.085
            ax.add_patch(Rectangle((cx, cy), 0.07, 0.07, color=ROLE[r][1] if j < k else TRACK, lw=0))
        ax.text(x0 + 0.42, y0 - 0.12, f"{p:.0f}%", ha="center", va="top", fontsize=8, fontweight="bold", color=ROLE[r][1])
    ax.text(ci_ * 1.1 + 0.42, 2.38, ROLE[r][0], ha="center", fontsize=6.2, color=ROLE[r][1], fontweight="semibold")
for ri, (sname, _) in enumerate(rows_):
    ax.text(-0.08, 1.35 - ri * 1.3 + 0.42, sname, rotation=90, ha="right", va="center", fontsize=6.2, color=INK2)
head(ax, "b", "Discretionary foods reach most people daily", x=-0.06, y=1.075)
ax.text(0.0, -0.06, "Each square = 1% of respondents eating sugar-sweetened drinks, sweets or fried snacks yesterday",
        transform=ax.transAxes, fontsize=5.6, color=MUTED, va="top")
save(fig, "Fig3_diets")

# ======================= Fig 4: knowledge vs diet =======================
rng = np.random.default_rng(2027)
ADJ = "C(wealth_q) + urban + C(division) + internet + log_food_exp_pc"
import pickle, hashlib, os
def standardise(g, formula, var, grid, family=None, group=None, B=150):
    key = hashlib.md5(f"{formula}|{var}|{list(grid)}|{group}|{B}|{len(g)}".encode()).hexdigest()[:12]
    cf = f"{D}/boot_{key}.pkl"
    if os.path.exists(cf): return pickle.load(open(cf, "rb"))
    res = _standardise(g, formula, var, grid, family, group, B)
    pickle.dump(res, open(cf, "wb")); return res
def _standardise(g, formula, var, grid, family=None, group=None, B=150):
    cols = ["dds_cook", "anydisc_cook", "anydisc_purch", var, "w", "psu", "wealth_q", "division", "internet", "log_food_exp_pc", "urban", "hfias_cat", "food_insecure_modsev"]
    g = g.dropna(subset=[c for c in cols if c in formula or c in ("w", "psu", var)]).copy()
    def fitpred(data):
        if family is None: m = smf.wls(formula, data=data, weights=data.w).fit()
        else: m = smf.glm(formula, data=data, family=family, var_weights=data.w / data.w.mean()).fit()
        out = {}
        for key, sub in ([("all", data)] if group is None else list(data.groupby(group))):
            preds = []
            for x in grid:
                s2 = sub.copy(); s2[var] = x
                preds.append(np.average(m.predict(s2), weights=s2.w))
            out[key] = np.array(preds)
        return out
    est = fitpred(g)
    psus = g.psu.unique(); boots = {k: [] for k in est}
    for b in range(B):
        samp = rng.choice(psus, size=len(psus), replace=True)
        gb = pd.concat([g[g.psu == p] for p in samp], ignore_index=True)
        try:
            pb = fitpred(gb)
            for k in boots: boots[k].append(pb[k])
        except Exception:
            pass
    return {k: (est[k], np.percentile(boots[k], 2.5, axis=0), np.percentile(boots[k], 97.5, axis=0)) for k in est}

fig = plt.figure(figsize=(7.4, 3.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1, 1], wspace=0.55)
# a: ridgeline of dietary diversity by knowledge quintile (cooks, RU)
ax = fig.add_subplot(gs[0])
g = ru.dropna(subset=["dds_cook", "bn_cook"]).copy()
g["kq"] = pd.qcut(g.bn_cook.rank(method="first"), 5, labels=False)
xs = np.arange(0, 11)
for qi in range(5):
    h = g[g.kq == qi]
    dens = np.array([h.w[h.dds_cook == x].sum() for x in xs]) / h.w.sum()
    base = (4 - qi) * 0.36
    xx = np.repeat(xs, 2)[1:]; xx = np.concatenate([[xs[0] - 0.5], (xs[:-1] + 0.5).repeat(2), [xs[-1] + 0.5]])
    yy = np.repeat(dens, 2)
    col = plt.cm.Blues(0.35 + 0.13 * qi)
    ax.fill_between(xx, base, base + yy * 0.78, color=col, lw=0, zorder=10 - qi, alpha=0.95)
    ax.plot(xx, base + yy * 0.78, color="white", lw=0.8, zorder=10 - qi)
    mu = np.average(h.dds_cook, weights=h.w)
    ax.plot([mu, mu], [base, base + 0.27], color=INK, lw=1.0, zorder=20)
    lo_, hi_ = h.bn_cook.min(), h.bn_cook.max()
    ax.text(-0.9, base + 0.05, f"Q{qi+1}", ha="right", fontsize=6, color=INK2, fontweight="semibold")
    ax.text(10.6, base + 0.05, f"{mu:.2f}", ha="left", fontsize=6, color=INK, fontweight="semibold")
ax.text(-0.9, 1.72, "Knowledge\nfifth", ha="right", fontsize=5.6, color=MUTED, va="bottom")
ax.text(10.6, 1.72, "Mean", ha="left", fontsize=5.6, color=MUTED, va="bottom")
ax.set_yticks([]); ax.set_xlim(-0.6, 10.5); ax.set_ylim(-0.02, 1.95); ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_xlabel("Food groups eaten yesterday (cooks)", labelpad=3)
head(ax, "a", "Same diets at every knowledge level", x=-0.12, y=1.06)

# b: predicted DDS vs knowledge by food security (cooks)
ax = fig.add_subplot(gs[1])
grid = np.linspace(2.5, 6.5, 21)
g = ru.copy(); g["fi"] = g.food_insecure_modsev
pr = standardise(g, f"dds_cook ~ bn10_cook*fi + {ADJ}", "bn10_cook", grid, group="fi")
for key, col, lab, yo in [(0.0, PU, "Food-secure or mild", 0.09), (1.0, CO, "Moderate/severe insecurity", -0.1)]:
    e, lo, hi = pr[key]
    ax.fill_between(grid * 10, lo, hi, color=col, alpha=0.14, lw=0)
    ax.plot(grid * 10, e, color=col, lw=2)
    ax.text(grid[-1] * 10 - 0.5, e[-1] + yo, lab, color=col, fontsize=5.9, fontweight="semibold", va="center", ha="right")
ax.set_xlim(25, 66); ax.set_xticks([30, 40, 50, 60]); ax.yaxis.grid(True, color=TRACK, lw=0.8); ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Cook's knowledge score", labelpad=3); ax.set_ylabel("Predicted food groups", labelpad=2)
head(ax, "b", "Food insecurity flips the slope", x=-0.2, y=1.06)
p_int = R["M2_knowledge_diet"]["RU"]["cook"]["interaction_p"]
ax.text(0.03, 0.04, "P for interaction < 0.001" if p_int < 0.001 else f"P for interaction = {p_int:.3f}", transform=ax.transAxes, fontsize=5.8, color=MUTED)

# c: predicted probability of discretionary foods vs knowledge
ax = fig.add_subplot(gs[2])
for r in ("cook", "purch"):
    g = ru.copy()
    pr = standardise(g, f"anydisc_{r} ~ bn10_{r} + {ADJ} + C(hfias_cat)", f"bn10_{r}", grid, family=sm.families.Binomial())
    e, lo, hi = pr["all"]
    col = ROLE[r][1]
    ax.fill_between(grid * 10, 100 * lo, 100 * hi, color=col, alpha=0.16, lw=0)
    ax.plot(grid * 10, 100 * e, color=col, lw=2)
    orr = R["M2_knowledge_diet"]["RU"][r]["disc_adj_OR"]
    ax.text(26, 93.5 if r == "purch" else 51, f"{ROLE[r][0]} · OR {orr['est']:.2f}", color=col, fontsize=5.9, fontweight="semibold", va="center")
ax.set_xlim(25, 66); ax.set_xticks([30, 40, 50, 60]); ax.yaxis.grid(True, color=TRACK, lw=0.8); ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Knowledge score", labelpad=3); ax.set_ylabel("Predicted % eating discretionary foods", labelpad=2); ax.set_ylim(48, 96)
head(ax, "c", "More knowledge, more sweets", x=-0.2, y=1.06)
save(fig, "Fig4_knowledge_vs_diet")

# ======================= Fig 5: hygiene =======================
fig = plt.figure(figsize=(7.2, 3.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.72)
ax = fig.add_subplot(gs[0]); q = np.arange(1, 6)
FS = R["foodsafety_practice"]
for s, col, lab in [("RU", PU, "Rural–urban"), ("Slum", AD, "Slum")]:
    v = np.array([FS[s]["station_ws_by_wealth"][str(i)] for i in q])
    ax.fill_between(q, 0, v, color=col, alpha=0.08, lw=0)
    ax.plot(q, v, color=col, lw=2.2, marker="o", ms=4.5, mec="white", mew=0.8)
    ax.text(5.15, v[-1], f"{lab}  {v[-1]:.0f}%", color=col, fontsize=6.2, fontweight="semibold", va="center")
    ax.text(0.85, v[0], f"{v[0]:.0f}%", color=col, fontsize=6.2, fontweight="semibold", va="center", ha="right")
ax.set_xlim(0.55, 6.4); ax.set_ylim(0, 106); ax.set_xticks(q); ax.set_xticklabels(["Poorest", "2", "3", "4", "Richest"])
ax.yaxis.grid(True, color=TRACK, lw=0.8); ax.set_axisbelow(True); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25", "50", "75", "100%"])
ax.set_xlabel("Household wealth fifth", labelpad=3)
head(ax, "a", "Water and soap where hands are washed", x=-0.14, y=1.1, sub="observed by enumerator, by wealth")

ax = fig.add_subplot(gs[1])
HW = R["handwashing"]
ct = [("after_latrine", "After defecation"), ("before_eating", "Before eating"), ("after_returning_home", "After returning home"), ("before_cooking", "Before cooking")]
for i, (k, lab) in enumerate(ct):
    ax.add_patch(Rectangle((0, i - 0.36), 100, 0.72, color=TRACK if i % 2 == 0 else "white", lw=0, zorder=0))
    for off, s, col in [(-0.14, "RU", PU), (0.14, "Slum", AD)]:
        cur = HW[s][k]["report"]; pre = cur * HW[s][k]["pre_share"] / 100
        ax.add_patch(FancyArrowPatch((pre, i + off), (cur, i + off), arrowstyle="-|>,head_length=3,head_width=2", color=col, lw=1.6, mutation_scale=1, zorder=3))
        ax.scatter(pre, i + off, s=12, color="white", edgecolor=col, lw=1.1, zorder=4)
        if s == "RU": ax.text(cur + 2.2, i + off, f"+{cur - pre:.0f}", va="center", fontsize=5.8, color=col, fontweight="semibold")
ax.set_yticks(range(len(ct))); ax.set_yticklabels([l for _, l in ct]); ax.set_ylim(len(ct) - 0.5, -0.6)
vgrid(ax, [0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100%"]); ax.set_xlim(0, 108)
ax.set_xlabel("Report washing hands with soap", labelpad=3)
head(ax, "b", "Pandemic gains skipped the kitchen", x=-0.5, y=1.1, sub="open circle = before pandemic (recalled); arrow head = Oct–Nov 2020")
ax.text(0.99, 0.02, "Rural–urban", transform=ax.transAxes, ha="right", fontsize=6, color=PU, fontweight="semibold")
ax.text(0.99, 0.08, "Slum", transform=ax.transAxes, ha="right", fontsize=6, color=AD, fontweight="semibold")
save(fig, "Fig5_handwashing")
print("v2 figures ok")
