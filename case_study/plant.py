"""Shared fed-batch process model ("virtual plant") used by the in-silico case studies.

The virtual plant is a fed-batch E. coli-type cultivation with substrate inhibition,
a Gaussian temperature optimum and Luedeking-Piret product formation. It is the
"ground truth" that the AI models must learn from noisy, sparse samples.
All numbers are illustrative, not calibrated to a specific strain.
"""
import numpy as np

# ---- true kinetic parameters (unknown to the data-driven models) ----
TRUE = dict(mu_ref=0.60, T_opt=34.0, T_w=3.0, Ks=0.15, Ki=15.0, Kr=1.0,
            Yxs=0.48, ms=0.03, alpha=0.08, beta=0.006, Sf=400.0)
T_END, DT, SAMPLE_EVERY = 16.0, 0.05, 1.0  # h


def true_rates(S, T, p=TRUE):
    """Specific rates mu, qS, qP (1/h, g/g/h) of the virtual plant."""
    S = np.maximum(S, 0.0)
    mu_max = p["mu_ref"] * np.exp(-((T - p["T_opt"]) / p["T_w"]) ** 2)
    mu = mu_max * S / (p["Ks"] + S) * p["Ki"] / (p["Ki"] + S)
    qS = mu / p["Yxs"] + p["ms"] * S / (p["Ks"] + S)
    # product formation: growth-associated, repressed by excess glucose, heat-labile pathway
    qP = (p["alpha"] * mu + p["beta"]) * p["Kr"] / (p["Kr"] + S) * np.exp(-0.15 * np.maximum(T - 33.0, 0.0) ** 2)
    return mu, qS, qP


def feed(t, k, F0=0.0015):
    """Exponential feed (L/h) for a 1 L start volume."""
    return F0 * np.exp(k * t)


def simulate(T, k, rate_fn=true_rates, x0=(0.5, 2.0, 0.0, 1.0), Sf=TRUE["Sf"]):
    """Integrate mass balances with RK4. T, k: arrays (n_runs,). Returns t, states (n_t, n_runs, 4)."""
    T = np.atleast_1d(np.asarray(T, float)); k = np.atleast_1d(np.asarray(k, float))
    n = T.size
    y = np.tile(np.asarray(x0, float), (n, 1))
    nsteps = int(round(T_END / DT)); every = int(round(SAMPLE_EVERY / DT))
    out = [y.copy()]

    def f(t, y):
        X, S, P, V = y.T
        F = feed(t, k)
        mu, qS, qP = rate_fn(S, T)
        D = F / V
        return np.stack([mu * X - D * X, -qS * X + D * (Sf - S), qP * X - D * P, F], axis=1)

    t = 0.0
    for i in range(nsteps):
        k1 = f(t, y); k2 = f(t + DT / 2, y + DT / 2 * k1)
        k3 = f(t + DT / 2, y + DT / 2 * k2); k4 = f(t + DT, y + DT * k3)
        y = y + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        y[:, 1] = np.maximum(y[:, 1], 0.0)
        t += DT
        if (i + 1) % every == 0:
            out.append(y.copy())
    tt = np.arange(len(out)) * SAMPLE_EVERY
    return tt, np.stack(out)


def measure(states, rng, rel=0.05, abs_=(0.05, 0.05, 0.01)):
    """Add heteroscedastic measurement noise to X, S, P (volume assumed known)."""
    y = states.copy()
    for j in range(3):
        sd = rel * np.abs(y[..., j]) + abs_[j]
        y[..., j] = np.maximum(y[..., j] + rng.normal(0, 1, y[..., j].shape) * sd, 0.0)
    return y
