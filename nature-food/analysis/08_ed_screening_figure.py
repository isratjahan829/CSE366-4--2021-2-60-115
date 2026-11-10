"""Extended Data Fig. 1: performance of brief household screening models."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = "/home/user/CSE366-4--2021-2-60-115/nature-food/"
V = json.load(open(ROOT + "analysis/screening_results.json"))
INK, INK2, MUTED, TRACK = "#1c2230", "#4a5060", "#8a8f9c", "#efede8"
C = {"food_insecurity": "#dd5a2f", "no_handwash_facility": "#3366b0", "low_diet_diversity": "#1f9e89"}
short = {"food_insecurity": "Food insecurity", "no_handwash_facility": "No water and soap", "low_diet_diversity": "Low diet diversity"}
plt.rcParams.update({"font.family": "Inter", "font.size": 6.8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#d9d6cf",
                     "xtick.major.size": 0, "ytick.major.size": 0, "xtick.color": INK2, "ytick.color": INK2, "savefig.dpi": 400})
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.38})
ax = axs[0]; g = np.linspace(0, 100, 21)
ax.plot([0, 100], [0, 100], color=MUTED, lw=0.9, ls=(0, (3, 3)))
ax.text(88, 75, "random", color=MUTED, fontsize=5.8, rotation=36, ha="center")
for k, (t, v) in enumerate(V.items()):
    cap = 100 * np.array(v["capture_curve"]); ax.plot(g, cap, color=C[t], lw=2)
    ax.text(98, 26 - 7 * k, f"{short[t]} · AUC {v['auc_logit']:.2f}", color=C[t], fontsize=5.6, fontweight="semibold", ha="right")
ax.axvline(30, color=INK2, lw=0.6); ax.text(30.8, 3, "top 30%", fontsize=5.6, color=INK2)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.grid(color=TRACK, lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("Households screened, ranked by predicted risk (%)"); ax.set_ylabel("Affected households identified (%)")
ax.text(-0.02, 1.05, "a", transform=ax.transAxes, fontsize=9.5, fontweight="bold", ha="right", va="bottom")
ax.text(0.03, 1.05, "Ranking by a 2-minute questionnaire finds risk early", transform=ax.transAxes, fontsize=7.2, fontweight="semibold", va="bottom")
ax = axs[1]
ax.plot([0, 100], [0, 100], color=MUTED, lw=0.9, ls=(0, (3, 3)))
for t, v in V.items():
    p, o = zip(*v["calibration"]); ax.plot(100 * np.array(p), 100 * np.array(o), color=C[t], lw=1.4, marker="o", ms=3.5, mec="white", mew=0.6)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.grid(color=TRACK, lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("Predicted risk, decile mean (%)"); ax.set_ylabel("Observed prevalence (%)")
ax.text(-0.02, 1.05, "b", transform=ax.transAxes, fontsize=9.5, fontweight="bold", ha="right", va="bottom")
ax.text(0.03, 1.05, "Predicted and observed risk agree", transform=ax.transAxes, fontsize=7.2, fontweight="semibold", va="bottom")
fig.savefig(ROOT + "figures/EDFig1_screening.png", bbox_inches="tight", pad_inches=0.04, facecolor="white")
fig.savefig(ROOT + "figures/EDFig1_screening.pdf", bbox_inches="tight", pad_inches=0.04, facecolor="white")
print("ok")
