"""Publication figures (600 dpi TIFF/PNG + PDF) for case studies A-D."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "../figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
                     "axes.labelsize": 8.5, "legend.fontsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 600})
BLUE, GREY, ORANGE, TEAL, RED = "#1f5fa8", "#8a8a8a", "#e76f51", "#2a9d8f", "#b23a48"
MCOL = {"textbook": GREY, "blackbox": ORANGE, "hybrid": BLUE}
MLAB = {"textbook": "Textbook mechanistic", "blackbox": "Black-box NN", "hybrid": "Hybrid (NN + mass balances)"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.tif", bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def tag(ax, s):
    ax.text(-0.16, 1.03, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


# ---------------- Figure 6: case A ----------------
A = json.load(open("results/case_a.json"))
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.5, 1, 1]})
ax = axs[0]; sets = ["interp", "extrap_feed", "extrap_temp"]
names = ["Interp.", "Extrap.\nfeed", "Extrap.\ntemp."]
for i, m in enumerate(MCOL):
    data = [A["raw"][m][s] for s in sets]
    pos = np.arange(3) + (i - 1) * 0.26
    bp = ax.boxplot(data, positions=pos, widths=0.22, patch_artist=True, showfliers=True,
                    medianprops={"color": "k", "lw": 1}, flierprops={"ms": 2.5})
    for b in bp["boxes"]:
        b.set(facecolor=MCOL[m], alpha=0.75, lw=0.6)
ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels(names)
ax.set_ylabel("NRMSE of X, S, P (%)"); ax.set_title("Prediction error, 10 replicates")
ax.legend([plt.Rectangle((0, 0), 1, 1, fc=MCOL[m], alpha=0.75) for m in MCOL], [MLAB[m] for m in MCOL],
          loc="upper left", frameon=False, fontsize=6.5)
ax.set_ylim(2, 6000)
tag(ax, "a")
ex = A["example"]; t = np.array(ex["t"])
for j, (ax, lab, st) in enumerate(zip(axs[1:], ["Biomass X (g/L)", "Product P (g/L)"], [0, 2])):
    r = 1  # run 1: T = 34 C, feed exponent 0.31 1/h (outside training feed range)
    ax.plot(t, np.array(ex["truth"])[:, r, st], "k-", lw=2.2, alpha=0.35, label="Virtual plant (truth)")
    for m in MCOL:
        ax.plot(t, np.array(ex[m])[:, r, st], color=MCOL[m], lw=1.2, label=MLAB[m])
    ax.set_xlabel("Time (h)"); ax.set_ylabel(lab)
    ax.set_title("Unseen feed rate, k = 0.31" if j == 0 else "Same run: product")
    tag(ax, "bc"[j])
axs[1].legend(frameon=False, loc="upper left", fontsize=6.5)
fig.tight_layout(); save(fig, "Figure_6")

# ---------------- Figure 7: case B ----------------
B = json.load(open("results/case_b.json")); p = B["plot"]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0))
ax = axs[0, 0]; wn = np.array(p["wn"])
ax.plot(wn, p["spec_examples"][0], color=GREY, lw=0.8, label="Low glucose sample")
ax.plot(wn, p["spec_examples"][1], color=BLUE, lw=0.8, label="High glucose sample")
for c in (520, 911, 1065, 1125):
    ax.axvline(c, color=TEAL, lw=0.5, ls=":")
ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_ylabel("Intensity (a.u.)")
ax.set_title("Simulated in-line Raman spectra"); ax.legend(frameon=False); tag(ax, "a")
ax = axs[0, 1]; y, mu, sd = map(np.array, (p["yt_glc"], p["p_gp"], p["sd_gp"]))
ax.errorbar(y, mu, yerr=1.96 * sd, fmt="o", ms=2.5, color=BLUE, ecolor="#9db8d9", elinewidth=0.7, capsize=0)
lim = [0, max(y.max(), mu.max()) * 1.05]; ax.plot(lim, lim, "k--", lw=0.7)
r = B["results"]["models"]["glucose"]
ax.text(0.04, 0.96, f"PLS ({r['n_components']} LV): RMSEP = {r['PLS']['RMSEP']:.2f} g/L, R$^2$ = {r['PLS']['R2']:.3f}\n"
        f"GP 95 % interval coverage = {r['GP_95PI_coverage'] * 100:.0f} %", transform=ax.transAxes, va="top", fontsize=7)
ax.set_xlabel("Reference glucose (g/L)"); ax.set_ylabel("Predicted glucose (g/L)")
ax.set_title("Test set (n = 60), 95 % intervals"); tag(ax, "b")
ax = axs[1, 0]; q = np.array(p["q"]); idx = np.arange(len(q))
ax.plot(idx, q, "o-", ms=2.5, lw=0.7, color=BLUE)
ax.axhline(p["q_lim"], color=RED, lw=1, ls="--", label="99 % control limit")
ax.axvline(p["onset"] - 0.5, color=GREY, lw=1, ls=":", label="New raw-material lot")
ax.set_yscale("log"); ax.set_xlabel("Sample number in monitoring run"); ax.set_ylabel("Q statistic (SPE)")
ax.set_title("Drift detection with multivariate SPC"); ax.legend(frameon=False, loc="upper left"); tag(ax, "c")
ax = axs[1, 1]; m = B["results"]["monitoring"]
vals = [m["RMSEP_glucose_before_drift"], m["RMSEP_glucose_after_drift"], m["RMSEP_glucose_after_drift_recalibrated"]]
bars = ax.bar(["Before drift", "After drift,\noriginal model", "After drift,\nrecalibrated"], vals,
              color=[GREY, RED, BLUE], alpha=0.85, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5)
ax.set_ylabel("Glucose RMSEP (g/L)"); ax.set_title("Accuracy loss and recovery"); tag(ax, "d")
fig.tight_layout(); save(fig, "Figure_7")

# ---------------- Figure 8: case C ----------------
import sys; sys.path.insert(0, ".")
from case_c_scaleup import (scale_down_pulse, steady_profile, lifelines, q_up, NC, REGIMES)
C = json.load(open("results/case_c.json"))["results"]
fig = plt.figure(figsize=(7.2, 5.2)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
ax = fig.add_subplot(gs[0, 0]); S = np.array(C["large_scale"]["S_profile_g_L"])
ax.barh(np.arange(NC)[::-1] + 0.5, S, height=0.8, color=[ORANGE if q_up(s) * 3600 > 0.7 else (GREY if q_up(s) * 3600 < 0.12 else TEAL) for s in S])
ax.set_yticks([0.5, NC - 0.5]); ax.set_yticklabels(["bottom", "top (feed)"]); ax.set_xscale("log")
ax.set_xlabel("Glucose (g/L)"); ax.set_title("50 m$^3$ tank profile"); tag(ax, "a")
ax = fig.add_subplot(gs[0, 1:])
traj_S = None
Qe = 2080.0; Sc = steady_profile(Qe)
rng = np.random.default_rng(3); pos = 0; dt = 0.1; Sline = []
for k in range(3000):
    u = rng.random(); pr = Qe / (50_000 / NC) * dt
    if u < pr and pos > 0: pos -= 1
    elif pr <= u < 2 * pr and pos < NC - 1: pos += 1
    Sline.append(Sc[pos])
tt = np.arange(3000) * dt
ax.plot(tt, Sline, color=BLUE, lw=0.8, label="One cell lifeline, 50 m$^3$ tank")
pd = C["sd_regime_matched_pulse"]; Tc, du = pd["period_s"], pd["duty"]
Fon = 0.45 / 3600 * 30 / du; s_ = 0.03; Sp = []
for k in range(int(900 / dt)):
    tk = k * dt; on = (tk % Tc) < du * Tc
    s_ = max(s_ + dt * (Fon * on - q_up(s_) * 30), 0); Sp.append(s_)
ax.plot(tt, Sp[-3000:], color=ORANGE, lw=0.8, alpha=0.9, label=f"Regime-matched pulse feed ({Tc:.0f} s, {du * 100:.0f} % duty)")
ax.axhline(0.03, color="k", lw=0.8, ls="--", label="Homogeneous lab vessel")
ax.set_yscale("log"); ax.set_ylim(1e-6, 2); ax.set_xlabel("Time (s)"); ax.set_ylabel("Glucose seen by cell (g/L)")
ax.set_title("What a cell experiences"); ax.legend(frameon=False, fontsize=6.5, loc="lower right", ncol=1); tag(ax, "b")
ax = fig.add_subplot(gs[1, :2])
systems = [("large_scale", "50 m$^3$ production"), ("lab_homogeneous", "Lab, homogeneous"),
           ("sd_conventional", "STR+PFR, heuristic"), ("sd_conventional_pulse", "Pulse, heuristic"),
           ("sd_regime_matched_pulse", "Pulse, regime-matched")]
cols = ["#5b5b5b", "#9db8d9", TEAL, "#f4a261", RED]
w = 0.16
for i, (key, lab) in enumerate(systems):
    f = np.array(C[key]["regime_fractions"]) * 100
    ax.bar(np.arange(5) + (i - 2) * w, f, width=w, color=["#2b2b2b", "#7f9fc6", TEAL, "#f4a261", RED][i] if False else
           ["#3a3a3a", "#a9c1e0", "#7cc4b9", "#f6b38f", BLUE][i], label=lab, edgecolor="white", lw=0.3)
ax.set_xticks(range(5)); ax.set_xticklabels(["Starvation", "Limited", "Near-\ncritical", "Overflow", "Strong\noverflow"]); ax.set_ylim(0, 135)
ax.set_ylabel("Time in regime (%)"); ax.set_title("Regime distribution")
ax.legend(frameon=False, ncol=3, fontsize=6.3, loc="upper center"); tag(ax, "c")
ax = fig.add_subplot(gs[1, 2])
W1 = [C[k].get("W1_g_g_h", 0.0) for k, _ in systems[1:]]
ac = [C[k]["acetate_mg_g_h"] for k, _ in systems]
ax.bar(range(4), W1, color=["#a9c1e0", "#7cc4b9", "#f6b38f", BLUE])
ax.set_xticks(range(4)); ax.set_xticklabels(["Lab", "STR+\nPFR", "Pulse\nheur.", "Pulse\nmatch."], fontsize=6.8)
ax.set_ylabel("Wasserstein distance to\nproduction (g g$^{-1}$ h$^{-1}$)"); ax.set_title("Environment mismatch")
for i, v in enumerate(W1):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=6.8)
tag(ax, "d")
fig.tight_layout(); save(fig, "Figure_8")

# ---------------- Figure 9: case D ----------------
D = json.load(open("results/case_d.json")); fstar = D["f_star"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axs[0]
for (name, res), c in zip(D["results"].items(), [BLUE, GREY, ORANGE]):
    cur = np.array(res["curves"]) / fstar * 100; n = np.arange(1, cur.shape[1] + 1)
    med, q1, q3 = np.median(cur, 0), np.percentile(cur, 25, 0), np.percentile(cur, 75, 0)
    ax.plot(n, med, color=c, lw=1.4, label=name); ax.fill_between(n, q1, q3, color=c, alpha=0.15, lw=0)
ax.axhline(100, color="k", lw=0.6, ls=":")
ax.set_xlabel("Number of bioreactor runs"); ax.set_ylabel("Best observed product mass (% of optimum)")
ax.set_title("20 campaigns: median and IQR"); ax.legend(frameon=False, loc="lower right"); tag(ax, "a")
ax = axs[1]; u = np.array(D["landscape"]["u"]); f = np.array(D["landscape"]["f"])
LO, HI = np.array([30.0, 0.08]), np.array([38.0, 0.32]); xy = LO + u * (HI - LO)
n = int(np.sqrt(len(f)))
cs = ax.contourf(xy[:, 0].reshape(n, n), xy[:, 1].reshape(n, n), f.reshape(n, n), levels=14, cmap="Blues")
recs = np.array(D["results"]["Bayesian optimisation"]["recommendations"])
ax.plot(recs[:, 0], recs[:, 1], "o", ms=3.5, mfc=ORANGE, mec="k", mew=0.4, label="BO recommendations")
ax.plot(*D["x_star"], "*", ms=10, color=RED, mec="k", mew=0.5, label="True optimum")
ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Feed exponent k (h$^{-1}$)")
ax.set_title("Product mass landscape (g)"); ax.legend(frameon=False, fontsize=6.5, loc="lower left")
fig.colorbar(cs, ax=ax, shrink=0.85); tag(ax, "b")
fig.tight_layout(); save(fig, "Figure_9")
print("figures written")
