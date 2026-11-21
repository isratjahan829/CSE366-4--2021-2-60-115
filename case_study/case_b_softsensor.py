"""Case study B: Raman soft sensor for glucose and lactate with uncertainty and drift detection.

Synthetic Raman spectra (400-1800 cm-1) are built from Gaussian bands of glucose, lactate,
cells/protein and medium, plus fluorescence baseline and noise. Models: PLS (Savitzky-Golay
1st derivative + mean centring, components chosen by 10-fold CV), Gaussian-process regression
on PLS scores (gives prediction intervals) and a multilayer perceptron. Multivariate statistical
process control (Hotelling T2 and Q/SPE) is used to detect a raw-material drift.
"""
import json
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import f as fdist, chi2
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

WN = np.arange(400, 1802, 2.0)


def band(c, w, a):
    return a * np.exp(-0.5 * ((WN - c) / w) ** 2)


PURE = {
    "glucose": band(1125, 12, 1.0) + band(1065, 14, 0.7) + band(911, 10, 0.5) + band(520, 12, 0.6) + band(1365, 15, 0.3),
    "lactate": band(853, 10, 1.0) + band(1040, 12, 0.8) + band(1420, 16, 0.5) + band(1310, 12, 0.25),
    "cells":   band(1655, 25, 0.9) + band(1450, 20, 0.8) + band(1003, 6, 0.5) + band(1250, 30, 0.4),
    "medium":  band(1080, 40, 0.5) + band(760, 20, 0.3) + band(1590, 30, 0.25),
}
INTERFERENT = band(1110, 16, 1.0) + band(1060, 14, 0.6) + band(1290, 18, 0.5)   # sugar-rich hydrolysate lot (drift)


def make_spectra(n, rng, drift=0.0):
    t = rng.uniform(0, 1, n)                               # culture progress
    glc = np.clip(7.5 * (1 - t) + rng.normal(0, 1.0, n), 0.2, 9)
    lac = np.clip(0.3 + 3.5 * t + rng.normal(0, 0.6, n), 0.1, 5)
    vcd = np.clip(2 + 18 * t + rng.normal(0, 2, n), 1, 25)  # 1e6 cells/mL
    med = rng.normal(1, 0.08, n)
    shift = rng.normal(0, 1.0, n)                          # wavenumber calibration jitter (cm-1)
    def comp(name, w):
        return np.stack([np.interp(WN, WN + s_, PURE[name]) for s_ in shift]) * w[:, None]
    x = (comp("glucose", glc / 8) * 0.30 + comp("lactate", lac / 4) * 0.30
         + comp("cells", vcd / 20) * 0.40 + comp("medium", med))
    x = x * rng.normal(1, 0.03, n)[:, None]                # laser power / focus variation
    # fluorescence baseline (random smooth polynomial) + detector noise
    u = (WN - WN.mean()) / np.ptp(WN)
    base = (rng.normal(2, 0.3, n)[:, None] + rng.normal(0, 0.4, n)[:, None] * u
            + rng.normal(0, 0.3, n)[:, None] * u ** 2)
    x = x + base + rng.normal(0, 0.02, x.shape)
    if np.any(drift):
        x = x + np.outer(np.broadcast_to(drift, (n,)), INTERFERENT) * 0.25
    # reference analyser error (e.g. enzymatic analyser: 3% CV + 0.05 g/L)
    ref = np.c_[glc * rng.normal(1, 0.03, n) + rng.normal(0, 0.05, n), lac * rng.normal(1, 0.03, n) + rng.normal(0, 0.05, n)]
    return x, ref


def prep(x):
    return savgol_filter(x, 15, 2, deriv=1, axis=1)


def choose_ncomp(X, y, max_a=12):
    cv = KFold(10, shuffle=True, random_state=0); rmse = []
    for a in range(1, max_a + 1):
        p = cross_val_predict(PLSRegression(a, scale=False), X, y, cv=cv)
        rmse.append(np.sqrt(np.mean((p - y) ** 2)))
    # smallest model within 2% of the minimum (parsimony)
    best = int(np.argmin(rmse)); a = int(np.argmax(np.array(rmse) <= rmse[best] * 1.02)) + 1
    return a, rmse


def metrics(p, y):
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    r2 = float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))
    return {"RMSEP": rmse, "R2": r2, "bias": float(np.mean(p - y)), "RPD": float(np.std(y, ddof=1) / rmse)}


class MSPC:
    """Hotelling T2 and Q (SPE) statistics with 99% control limits from a PLS model."""
    def __init__(self, pls, X):
        self.pls = pls; self.mu = X.mean(0)
        T = pls.transform(X); self.s2 = T.var(0, ddof=1); n, A = T.shape
        self.t2_lim = A * (n - 1) * (n + 1) / (n * (n - A)) * fdist.ppf(0.99, A, n - A)
        Q = self.q(X); m, v = Q.mean(), Q.var(ddof=1)
        g, h = v / (2 * m), 2 * m ** 2 / v
        self.q_lim = g * chi2.ppf(0.99, h)

    def t2(self, X):
        T = self.pls.transform(X); return np.sum(T ** 2 / self.s2, 1)

    def q(self, X):
        T = self.pls.transform(X)
        Xhat = T @ self.pls.x_loadings_.T + self.mu
        return np.sum((X - Xhat) ** 2, 1)


def run():
    rng = np.random.default_rng(7)
    Xc_raw, Yc = make_spectra(120, rng); Xt_raw, Yt = make_spectra(60, rng)
    Xc, Xt = prep(Xc_raw), prep(Xt_raw)
    out = {"n_cal": 120, "n_test": 60, "models": {}}
    preds = {}
    for j, name in enumerate(("glucose", "lactate")):
        a, cvcurve = choose_ncomp(Xc, Yc[:, j])
        pls = PLSRegression(a, scale=False).fit(Xc, Yc[:, j])
        p_pls = pls.predict(Xt).ravel()
        # GP on PLS scores -> predictive standard deviation
        Tc, Tt = pls.transform(Xc), pls.transform(Xt)
        ss = StandardScaler().fit(Tc)
        kern = ConstantKernel(1.0) * RBF(np.ones(a)) + WhiteKernel(0.05)
        gp = GaussianProcessRegressor(kern, normalize_y=True, random_state=0).fit(ss.transform(Tc), Yc[:, j])
        p_gp, sd_gp = gp.predict(ss.transform(Tt), return_std=True)
        cover = float(np.mean(np.abs(p_gp - Yt[:, j]) <= 1.96 * sd_gp))
        # MLP on preprocessed full spectrum
        sx = StandardScaler().fit(Xc)
        mlp = MLPRegressor(hidden_layer_sizes=(64, 16), alpha=1e-1, max_iter=5000, random_state=0,
                           early_stopping=True, validation_fraction=0.15).fit(sx.transform(Xc), Yc[:, j])
        p_mlp = mlp.predict(sx.transform(Xt))
        out["models"][name] = {"n_components": a, "cv_rmse_curve": cvcurve,
                               "PLS": metrics(p_pls, Yt[:, j]), "GP_on_PLS_scores": metrics(p_gp, Yt[:, j]),
                               "MLP": metrics(p_mlp, Yt[:, j]), "GP_95PI_coverage": cover,
                               "GP_mean_PI_halfwidth": float(np.mean(1.96 * sd_gp))}
        preds[name] = (pls, gp, ss, p_gp, sd_gp)

    # ---------- monitoring run with raw-material drift after sample 40 ----------
    pls_g, gp_g, ss_g, p_gp, sd_gp = preds["glucose"]
    mspc = MSPC(pls_g, Xc)
    n_mon, onset = 80, 40
    drift = np.r_[np.zeros(onset), np.linspace(0.3, 1.0, n_mon - onset)]
    Xm_raw, Ym = make_spectra(n_mon, rng, drift=drift); Xm = prep(Xm_raw)
    t2, q = mspc.t2(Xm), mspc.q(Xm)
    pm = pls_g.predict(Xm).ravel()
    alarm = (t2 > mspc.t2_lim) | (q > mspc.q_lim)
    first = int(np.argmax(alarm[onset:])) + onset if alarm[onset:].any() else None
    # recalibration: add 12 samples measured under the new raw-material lot
    Xr_raw, Yr = make_spectra(12, rng, drift=rng.uniform(0.3, 1.0, 12))
    Xaug, yaug = np.vstack([Xc, prep(Xr_raw)]), np.r_[Yc[:, 0], Yr[:, 0]]
    a2, _ = choose_ncomp(Xaug, yaug)
    pls2 = PLSRegression(a2, scale=False).fit(Xaug, yaug)
    pm2 = pls2.predict(Xm).ravel()
    out["monitoring"] = {
        "T2_limit_99": float(mspc.t2_lim), "Q_limit_99": float(mspc.q_lim),
        "false_alarm_rate_before_drift": float(alarm[:onset].mean()),
        "detection_rate_after_drift": float(alarm[onset:].mean()),
        "first_alarm_index": first, "drift_onset_index": onset,
        "RMSEP_glucose_before_drift": float(np.sqrt(np.mean((pm[:onset] - Ym[:onset, 0]) ** 2))),
        "RMSEP_glucose_after_drift": float(np.sqrt(np.mean((pm[onset:] - Ym[onset:, 0]) ** 2))),
        "RMSEP_glucose_after_drift_recalibrated": float(np.sqrt(np.mean((pm2[onset:] - Ym[onset:, 0]) ** 2))),
        "n_components_recalibrated": a2,
    }
    plot = {"wn": WN.tolist(), "spec_examples": Xc_raw[[np.argmin(Yc[:, 0]), np.argmax(Yc[:, 0])]].tolist(),
            "yt_glc": Yt[:, 0].tolist(), "p_gp": p_gp.tolist(), "sd_gp": sd_gp.tolist(),
            "q": q.tolist(), "t2": t2.tolist(), "q_lim": float(mspc.q_lim), "t2_lim": float(mspc.t2_lim),
            "onset": onset, "err_before": np.abs(pm - Ym[:, 0]).tolist(), "err_recal": np.abs(pm2 - Ym[:, 0]).tolist()}
    json.dump({"results": out, "plot": plot}, open("results/case_b.json", "w"), indent=1)
    print(json.dumps(out, indent=1, default=lambda o: round(o, 4)))


if __name__ == "__main__":
    run()
