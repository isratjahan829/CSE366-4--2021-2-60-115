"""Case study C: regime-matched scale-down design for a 50 m3 glucose-limited fed-batch.

1. Classical scale-up criteria (geometric similarity, Rushton turbine, Grenville mixing time).
2. Large tank: 10-compartment axial model, feed at the top, quasi-steady glucose field.
3. Cell lifelines: continuous-time Markov chain between compartments -> regime statistics.
4. Scale-down simulator: stirred tank (STR) + plug-flow loop (PFR) fed at the PFR inlet.
   Grid search for the PFR volume fraction and residence time that minimise the Jensen-Shannon
   divergence between scale-down and large-scale regime distributions (+ excursion duration).
5. Consequence: overflow (acetate) and starvation exposure in each system.
All parameter values are illustrative, chosen in ranges typical for E. coli fed-batch processes.
"""
import json
import numpy as np
from scipy.optimize import brentq

# ---------------- kinetics (per second) ----------------
QMAX, KS = 1.2 / 3600, 0.05          # g glucose / g cells / s ; g/L
QCRIT = 0.70 / 3600                  # overflow threshold (acetate formation above this uptake rate)
QSTARV = 0.12 / 3600                 # below this, cells are in starvation regime
X = 30.0                             # g cells / L
Q_TARGET = 0.45 / 3600               # mean uptake demanded by the feed (mu = 0.2 1/h)


def q_up(S):
    return QMAX * S / (KS + S)


EDGES = np.array([0.12, 0.40, 0.70, 0.95]) / 3600   # uptake-rate regime boundaries (g/g/s)
REGIMES = ["starvation", "limited", "near-critical", "overflow", "strong overflow"]
NR = len(REGIMES)


def regime(S):
    """0 starvation, 1 limited, 2 near-critical, 3 overflow, 4 strong overflow."""
    return np.digitize(q_up(S), EDGES)


def acetate(S):
    return 0.5 * np.maximum(q_up(S) - QCRIT, 0)                 # g acetate / g cells / s


# ---------------- 1. classical scale-up criteria ----------------
def classical_table():
    Np, DT_ratio = 5.0, 1 / 3
    Vs, Vl = 2e-3, 50.0
    Ts = (4 * Vs / np.pi) ** (1 / 3); Tl = (4 * Vl / np.pi) ** (1 / 3)      # H = T
    Ds, Dl = Ts * DT_ratio, Tl * DT_ratio; Ns = 800 / 60
    def props(N, D, T, V):
        P = Np * 1000 * N ** 3 * D ** 5
        return {"N_rpm": N * 60, "P_per_V_kW_m3": P / V / 1000, "tip_speed_m_s": np.pi * N * D,
                "Re": 1000 * N * D ** 2 / 1e-3, "t_mix95_s": 5.20 / (Np ** (1 / 3) * N) * (T / D) ** 2,
                "kLa_rel": (P / V) ** 0.4 * (T / Ts) ** 0.5}    # van't Riet, constant vvm -> vs ~ T
    small = props(Ns, Ds, Ts, Vs)
    crit = {"constant P/V": Ns * (Ds / Dl) ** (2 / 3), "constant tip speed": Ns * Ds / Dl,
            "constant Re": Ns * (Ds / Dl) ** 2, "constant mixing time": Ns}
    rows = {"2 L laboratory": small}
    for name, N in crit.items():
        rows["50 m3, " + name] = props(N, Dl, Tl, Vl)
    k0 = small["kLa_rel"]
    for r in rows.values():
        r["kLa_rel"] = r["kLa_rel"] / k0
    return {"T_small_m": Ts, "T_large_m": Tl, "rows": rows}


# ---------------- 2. large-tank compartment model ----------------
NC, VL = 10, 50_000.0                # compartments, litres
Vi = VL / NC


def steady_profile(Qe, feed_comp=0, n_iter=400):
    """Quasi-steady glucose in an axial chain with exchange flow Qe (L/s) and point feed."""
    F = Q_TARGET * X * VL                                   # g/s, feed equals total demand
    S = np.full(NC, 0.03)
    for _ in range(n_iter):                                 # Gauss-Seidel with scalar root finding
        for i in range(NC):
            nb = [j for j in (i - 1, i + 1) if 0 <= j < NC]
            inflow = sum(Qe * S[j] for j in nb) + (F if i == feed_comp else 0.0)
            g = lambda s: inflow - len(nb) * Qe * s - q_up(s) * X * Vi
            S[i] = brentq(g, 0.0, 1e3)
    return S


def mixing_time(Qe):
    """95% homogenisation time after a tracer pulse in the top compartment (no reaction)."""
    c = np.zeros(NC); c[0] = 1.0; dt, t = 0.05, 0.0
    A = np.zeros((NC, NC))
    for i in range(NC):
        for j in (i - 1, i + 1):
            if 0 <= j < NC:
                A[i, j] += Qe / Vi; A[i, i] -= Qe / Vi
    mean = c.mean()
    while np.max(np.abs(c - mean)) > 0.05 * mean:
        c = c + dt * A @ c; t += dt
    return t


def lifelines(S_comp, Qe, n_cells=3000, t_end=1800.0, dt=0.1, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.integers(0, NC, n_cells); p = Qe / Vi * dt
    reg_comp = regime(S_comp)
    traj_reg = np.empty((int(t_end / dt), n_cells), np.int8)
    for k in range(traj_reg.shape[0]):
        u = rng.random(n_cells)
        up = (u < p) & (pos > 0); down = (u >= p) & (u < 2 * p) & (pos < NC - 1)
        pos = pos - up + down
        traj_reg[k] = reg_comp[pos]
    return traj_reg, dt


def regime_stats(traj_reg, dt):
    frac = np.array([(traj_reg == r).mean() for r in range(NR)])
    # mean duration of overflow excursions (s) and their frequency (per cell per hour)
    durs, n_exc = [], 0
    for c in range(min(traj_reg.shape[1], 600)):
        x = (traj_reg[:, c] >= 3).astype(int)
        d = np.diff(np.r_[0, x, 0]); st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        durs += list((en - st) * dt); n_exc += len(st)
    hours = traj_reg.shape[0] * dt / 3600 * min(traj_reg.shape[1], 600)
    return frac, (float(np.mean(durs)) if durs else 0.0), n_exc / hours


# ---------------- 3. STR + PFR scale-down simulator ----------------
def scale_down(phi, tau_p, n_pfr=40):
    """Return regime time-fractions, mean overflow-excursion duration and exposure averages."""
    V = 2.0; Vp = phi * V; Vs = V - Vp; Qr = Vp / tau_p
    F = Q_TARGET * X * V
    def pfr_profile(S_str):
        S_in = (Qr * S_str + F) / Qr                        # feed enters at the PFR inlet
        S = [S_in]; dtau = tau_p / n_pfr
        for _ in range(n_pfr - 1):                          # integrate dS/dt = -q X along the loop
            s = S[-1]
            for _ in range(5):
                s = max(s - dtau / 5 * q_up(s) * X, 0.0)
            S.append(s)
        return np.array(S)
    def balance(S_str):                                     # glucose balance over the STR
        return Qr * pfr_profile(S_str)[-1] - Qr * S_str - q_up(S_str) * X * Vs
    S_str = brentq(balance, 0.0, 50.0)
    Sp = pfr_profile(S_str)
    w = np.r_[np.full(n_pfr, Vp / n_pfr), Vs]               # volume weights = time weights
    Sall = np.r_[Sp, S_str]; reg = regime(Sall)
    frac = np.array([w[reg == r].sum() / V for r in range(NR)])
    ov = (regime(Sp) >= 3).sum() * tau_p / n_pfr if regime(S_str) < 3 else np.inf
    return frac, float(ov), Sall, w


def scale_down_pulse(Tc, duty, horizon=3000.0, warm=600.0, dt=0.1):
    """Single STR with cyclic (on/off) feeding. Vectorised over design arrays Tc, duty.
    Returns regime fractions (n, NR), mean overflow-excursion duration (s) and mean acetate rate."""
    Tc, duty = np.asarray(Tc, float), np.asarray(duty, float)
    Fon = Q_TARGET * X / duty                    # g/L/s during the on-phase (same mean feed)
    S = np.full(Tc.shape, 0.03); counts = np.zeros(Tc.shape + (NR,)); H = np.zeros(Tc.shape + (60,))
    ac = np.zeros(Tc.shape); entries = np.zeros(Tc.shape); prev = np.zeros(Tc.shape, bool)
    n = 0
    for k in range(int(horizon / dt)):
        t = k * dt
        on = (t % Tc) < duty * Tc
        S = np.maximum(S + dt * (Fon * on - q_up(S) * X), 0.0)
        if t >= warm:
            r = regime(S); counts[np.arange(S.size), r] += 1
            H[np.arange(S.size), np.clip(np.digitize(q_up(S), QBINS) - 1, 0, 59)] += 1
            ov = r >= 3; entries += ov & ~prev; prev = ov
            ac += acetate(S); n += 1
    frac = counts / n
    dur = np.where(entries > 0, frac[:, 3:].sum(1) * (horizon - warm) / np.maximum(entries, 1), 0.0)
    return frac, dur, ac / n, H / H.sum(1, keepdims=True)


QBINS = np.linspace(0, QMAX, 61)


def qhist(q, w=None):
    h, _ = np.histogram(q, QBINS, weights=w); return h / h.sum()


def wasserstein(h1, h2):
    """1-D Wasserstein (earth mover's) distance between two uptake-rate histograms, in g/g/h."""
    return float(np.sum(np.abs(np.cumsum(h1) - np.cumsum(h2))) * (QBINS[1] - QBINS[0]) * 3600)


def jsd(p, q, eps=1e-12):
    p, q = np.asarray(p) + eps, np.asarray(q) + eps; p, q = p / p.sum(), q / q.sum(); m = (p + q) / 2
    kl = lambda a, b: np.sum(a * np.log2(a / b))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def run():
    out = {"classical": classical_table(), "regimes": REGIMES, "edges_g_g_h": (EDGES * 3600).tolist()}
    Qe = 2080.0                                             # L/s exchange flow (tuned: 95 % mixing time = 90 s)
    tm = mixing_time(Qe)
    S_comp = steady_profile(Qe)
    traj, dt = lifelines(S_comp, Qe)
    frac_L, dur_L, freq_L = regime_stats(traj, dt)
    w = np.full(NC, Vi)
    out["large_scale"] = {"mixing_time_95_s": tm, "S_profile_g_L": S_comp.tolist(),
                          "regime_fractions": frac_L.tolist(), "overflow_excursion_mean_s": dur_L,
                          "overflow_excursions_per_h": freq_L,
                          "acetate_mg_g_h": float(np.sum(acetate(S_comp) * w) / VL * 3.6e6)}
    S_h = brentq(lambda s: q_up(s) - Q_TARGET, 0, 10)
    out["lab_homogeneous"] = {"S_g_L": S_h, "W1_g_g_h": wasserstein(qhist(np.array([q_up(S_h)])), qhist(q_up(S_comp))), "regime_fractions": np.eye(NR)[int(regime(S_h))].tolist(),
                              "acetate_mg_g_h": float(acetate(S_h) * 3.6e6)}
    # conventional heuristic: PFR residence time = large-scale mixing time, 10 % volume
    f_c, d_c, S_c, w_c = scale_down(0.10, tm)
    out["sd_conventional"] = {"phi": 0.10, "tau_p_s": tm, "regime_fractions": f_c.tolist(),
                              "overflow_excursion_s": d_c, "JSD": jsd(f_c, frac_L), "W1_g_g_h": wasserstein(qhist(q_up(S_c), w_c), qhist(q_up(S_comp))),
                              "acetate_mg_g_h": float(np.sum(acetate(S_c) * w_c) / 2.0 * 3.6e6)}
    # conventional heuristic 2: pulse feeding with period = large-scale mixing time, 50 % duty
    f_p, d_p, a_p, h_p = scale_down_pulse(np.array([tm]), np.array([0.5]))
    out["sd_conventional_pulse"] = {"period_s": tm, "duty": 0.5, "regime_fractions": f_p[0].tolist(),
                                    "overflow_excursion_s": float(d_p[0]), "JSD": jsd(f_p[0], frac_L), "W1_g_g_h": wasserstein(h_p[0], qhist(q_up(S_comp))),
                                    "acetate_mg_g_h": float(a_p[0] * 3.6e6)}
    hL = qhist(q_up(S_comp))
    def objective(h, d):
        return wasserstein(h, hL) + 0.02 * np.log(max(d, 1e-3) / dur_L) ** 2
    out["large_scale"]["q_hist"] = hL.tolist()
    # regime-matched design, topology A: STR + PFR (PFR <= 30 % of volume)
    bestA, grid = None, []
    for phi in np.arange(0.04, 0.31, 0.01):
        for tau in np.arange(5, 301, 2.5):
            f, d, S, ww = scale_down(phi, tau)
            if not np.isfinite(d):
                continue
            J = objective(qhist(q_up(S), ww), d); grid.append(("STR+PFR", float(phi), float(tau), J))
            if bestA is None or J < bestA[0]:
                bestA = (J, phi, tau, f, d, float(np.sum(acetate(S) * ww) / 2.0 * 3.6e6), wasserstein(qhist(q_up(S), ww), hL))
    # topology B: cyclic pulse feeding
    TC, DU = np.meshgrid(np.arange(10, 241, 2.5), np.r_[np.arange(0.02, 0.2, 0.01), np.arange(0.2, 1.0, 0.1)])
    fB, dB, aB, hB = scale_down_pulse(TC.ravel(), DU.ravel())
    JB = np.array([objective(hB[i], dB[i]) for i in range(len(fB))])
    i = int(np.argmin(JB))
    out["sd_regime_matched_STR_PFR"] = {"phi": float(bestA[1]), "tau_p_s": float(bestA[2]),
                                        "regime_fractions": bestA[3].tolist(), "overflow_excursion_s": bestA[4],
                                        "JSD": jsd(bestA[3], frac_L), "W1_g_g_h": bestA[6], "objective": bestA[0], "acetate_mg_g_h": bestA[5]}
    out["sd_regime_matched_pulse"] = {"period_s": float(TC.ravel()[i]), "duty": float(DU.ravel()[i]),
                                      "regime_fractions": fB[i].tolist(), "overflow_excursion_s": float(dB[i]),
                                      "JSD": jsd(fB[i], frac_L), "W1_g_g_h": wasserstein(hB[i], hL), "objective": float(JB[i]),
                                      "acetate_mg_g_h": float(aB[i] * 3.6e6)}
    grid_pulse = [(float(a), float(b), float(c)) for a, b, c in zip(TC.ravel(), DU.ravel(), JB)]
    json.dump({"results": out, "grid": grid, "grid_pulse": grid_pulse}, open("results/case_c.json", "w"), indent=1)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    run()
