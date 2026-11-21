"""Case study A: hybrid vs black-box vs textbook-mechanistic models of a fed-batch bioreactor.

Training data: 12 noisy runs inside a narrow operating window (Latin hypercube).
Test data: 8 runs inside the window (interpolation) and 8 outside it (extrapolation).
Repeated for 10 random replicates (new noise, new design, new NN initialisation).
"""
import json, warnings
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares
from scipy.stats import wilcoxon
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from plant import simulate, measure, feed, TRUE, DT
warnings.filterwarnings("ignore")

IN_T, IN_K = (32.0, 36.0), (0.12, 0.26)       # training window
OUT_T, OUT_K = (30.5, 37.5), (0.08, 0.30)     # wider test window


def lhs(n, rng, lo, hi):
    u = (rng.permutation(n) + rng.random(n)) / n
    return lo + u * (hi - lo)


def outside_feed(n, rng):
    """Temperature inside the window, feed exponent outside it (extrapolation via known physics)."""
    k = np.where(rng.random(n) < 0.5, rng.uniform(OUT_K[0], IN_K[0], n), rng.uniform(IN_K[1], OUT_K[1], n))
    return lhs(n, rng, *IN_T), k


def outside_temp(n, rng):
    """Feed inside the window, temperature outside it (extrapolation via learned kinetics)."""
    T = np.where(rng.random(n) < 0.5, rng.uniform(OUT_T[0], IN_T[0], n), rng.uniform(IN_T[1], OUT_T[1], n))
    return T, lhs(n, rng, *IN_K)


def nrmse(pred, true):
    """Mean over X,S,P of RMSE normalised by the range of the true trajectory set (%)."""
    out = []
    for j in range(3):
        rng_ = true[..., j].max() - true[..., j].min() + 1e-9
        out.append(np.sqrt(np.mean((pred[..., j] - true[..., j]) ** 2)) / rng_ * 100)
    return float(np.mean(out)), out


# ---------------- hybrid model: mass balances + NN for specific rates ----------------
def estimate_rates(t, Y, T, k):
    """Two-step (incremental) hybrid training: smooth data, then back-calculate specific rates."""
    rows, targets = [], []
    tf = np.linspace(t[0], t[-1], 161)
    for r in range(Y.shape[1]):
        X, S, P, V = (Y[:, r, j] for j in range(4))
        sX = UnivariateSpline(t, np.log(np.maximum(X, 1e-3)), s=len(t) * 0.003)
        sP = UnivariateSpline(t, P, s=len(t) * (0.05 * P.max() + 0.01) ** 2)
        sS = UnivariateSpline(t, S, s=len(t) * (0.05 * S.max() + 0.05) ** 2)
        sV = UnivariateSpline(t, V, s=0)
        Xf, Pf, Sf_, Vf = np.exp(sX(tf)), sP(tf), np.maximum(sS(tf), 0), sV(tf)
        D = feed(tf, k[r]) / Vf
        mu = sX.derivative()(tf) + D
        qP = (sP.derivative()(tf) + D * Pf) / Xf
        qS = (-sS.derivative()(tf) + D * (TRUE["Sf"] - Sf_)) / Xf
        rows.append(np.c_[Sf_, np.full_like(tf, T[r])]); targets.append(np.c_[mu, qS, qP])
    return np.vstack(rows), np.vstack(targets)


class Hybrid:
    """Serial hybrid model: known mass balances + small neural network (2-10-3) for mu, qS, qP.

    Training: (1) incremental step - fit the network to rates back-calculated from smoothed data;
    (2) end-to-end step - refine all weights by minimising the trajectory error through the ODE
    solver (L-BFGS, gradients by batched finite differences over the whole parameter vector).
    """
    H, SCALE = 10, np.array([0.6, 1.5, 0.06])

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        nH = self.H; self.n_par = 2 * nH + nH + nH * 3 + 3
        self.theta = self.rng.normal(0, 0.3, self.n_par)

    def _unpack(self, th):
        nH = self.H; th = np.atleast_2d(th)
        W1 = th[:, :2 * nH].reshape(-1, 2, nH); b1 = th[:, 2 * nH:3 * nH]
        W2 = th[:, 3 * nH:6 * nH].reshape(-1, nH, 3); b2 = th[:, 6 * nH:]
        return W1, b1, W2, b2

    def _nn(self, th, S, T):
        """th: (B, n_par); S, T: (B, n) -> rates (B, n, 3)."""
        z = np.stack([(np.log1p(np.maximum(S, 0)) - 1.0) / 1.5, (T - 34.0) / 2.0], -1)
        W1, b1, W2, b2 = self._unpack(th)
        h = np.tanh(np.einsum('bni,bih->bnh', z, W1) + b1[:, None, :])
        o = np.einsum('bnh,bho->bno', h, W2) + b2[:, None, :]
        return np.logaddexp(0, o) * self.SCALE           # softplus: rates >= 0 by construction

    def _sim(self, thetas, T, k):
        B, n = thetas.shape[0], T.size
        TT, KK = np.tile(T, B), np.tile(k, B)
        def rate_fn(S, Tv):
            r = self._nn(thetas, S.reshape(B, n), Tv.reshape(B, n)).reshape(B * n, 3)
            return r[:, 0], r[:, 1], r[:, 2]
        t, Y = simulate(TT, KK, rate_fn=rate_fn)
        return Y.reshape(len(t), B, n, 4)

    def fit(self, t, Y, T, k):
        # step 1: incremental fit to back-calculated rates
        Zs, R = estimate_rates(t, Y, T, k)
        def rate_loss(th):
            r = self._nn(th[None], Zs[None, :, 0], Zs[None, :, 1])[0]
            return np.mean(((r - np.maximum(R, 0)) / self.SCALE) ** 2) + 1e-4 * th @ th
        from scipy.optimize import minimize
        self.theta = minimize(rate_loss, self.theta, method="L-BFGS-B", options={"maxiter": 400}).x
        # step 2: end-to-end refinement on the measured trajectories
        sc = Y[..., :3].reshape(-1, 3).std(0) + 1e-6
        eps = 1e-4; I = np.eye(self.n_par) * eps
        def loss_grad(th):
            thetas = np.vstack([th, th + I])
            P = self._sim(thetas, T, k)[..., :3]
            L = np.array([np.mean(((P[:, b] - Y[..., :3]) / sc) ** 2) for b in range(thetas.shape[0])])
            L = L + 1e-4 * np.sum(thetas ** 2, 1)
            return L[0], (L[1:] - L[0]) / eps
        self.theta = minimize(loss_grad, self.theta, jac=True, method="L-BFGS-B",
                              options={"maxiter": 60}).x
        return self

    def rates(self, S, T):
        r = self._nn(self.theta[None], S[None], T[None])[0]
        return r[:, 0], r[:, 1], r[:, 2]

    def predict(self, T, k):
        return simulate(T, k, rate_fn=self.rates)[1]


# ---------------- black-box model: NN mapping (t, T, k) -> (X, S, P) ----------------
class BlackBox:
    def __init__(self, seed):
        self.sx, self.sy = StandardScaler(), StandardScaler()
        self.nn = MLPRegressor(hidden_layer_sizes=(32, 32), activation="tanh", alpha=1e-3, max_iter=4000,
                               learning_rate_init=3e-3, random_state=seed)

    def fit(self, t, Y, T, k):
        tt, TT = np.meshgrid(t, T, indexing="ij")
        _, KK = np.meshgrid(t, k, indexing="ij")
        Xin = np.c_[tt.ravel(), TT.ravel(), KK.ravel()]
        self.nn.fit(self.sx.fit_transform(Xin), self.sy.fit_transform(Y[..., :3].reshape(-1, 3)))
        self.t = t
        return self

    def predict(self, T, k):
        tt, TT = np.meshgrid(self.t, T, indexing="ij")
        _, KK = np.meshgrid(self.t, k, indexing="ij")
        Xin = np.c_[tt.ravel(), TT.ravel(), KK.ravel()]
        out = self.sy.inverse_transform(self.nn.predict(self.sx.transform(Xin)))
        return np.maximum(out.reshape(len(self.t), len(T), 3), 0)


# ---------------- textbook mechanistic model: Monod + constant yields ----------------
class Textbook:
    def _rates(self, th):
        mu_max, Ks, Yxs, alpha = np.exp(th)
        def fn(S, T):
            S = np.maximum(S, 0); mu = mu_max * S / (Ks + S)
            return mu, mu / Yxs, alpha * mu
        return fn

    def fit(self, t, Y, T, k):
        sc = Y[..., :3].reshape(-1, 3).std(0) + 1e-6
        def res(th):
            pred = simulate(T, k, rate_fn=self._rates(th))[1]
            return ((pred[..., :3] - Y[..., :3]) / sc).ravel()
        self.th = least_squares(res, np.log([0.5, 0.2, 0.5, 0.07]), max_nfev=60).x
        return self

    def predict(self, T, k):
        return simulate(T, k, rate_fn=self._rates(self.th))[1]


def run(n_rep=10, n_train=12, n_test=8):
    res = {m: {"interp": [], "extrap_feed": [], "extrap_temp": []} for m in ("textbook", "blackbox", "hybrid")}
    per_state = {m: {"interp": [], "extrap_feed": [], "extrap_temp": []} for m in res}
    example = None
    for rep in range(n_rep):
        rng = np.random.default_rng(100 + rep)
        Ttr, ktr = lhs(n_train, rng, *IN_T), lhs(n_train, rng, *IN_K)
        t, Ytrue = simulate(Ttr, ktr)
        Y = measure(Ytrue, rng)
        sets = {"interp": (lhs(n_test, rng, *IN_T), lhs(n_test, rng, *IN_K)),
                "extrap_feed": outside_feed(n_test, rng), "extrap_temp": outside_temp(n_test, rng)}
        models = {"textbook": Textbook().fit(t, Y, Ttr, ktr),
                  "blackbox": BlackBox(rep).fit(t, Y, Ttr, ktr),
                  "hybrid": Hybrid(rep).fit(t, Y, Ttr, ktr)}
        for name, (Ts, ks) in sets.items():
            truth = simulate(Ts, ks)[1][..., :3]
            for m, mod in models.items():
                pred = mod.predict(Ts, ks)[..., :3]
                e, ej = nrmse(pred, truth)
                res[m][name].append(e); per_state[m][name].append(ej)
        if rep == 0:
            Ts, ks = np.array([34.0, 34.0, 38.0]), np.array([0.18, 0.31, 0.18])
            example = {"t": t.tolist(), "T": Ts.tolist(), "k": ks.tolist(),
                       "truth": simulate(Ts, ks)[1][..., :3].tolist(),
                       **{m: models[m].predict(Ts, ks)[..., :3].tolist() for m in models}}
        print(rep, {m: [round(res[m][s][-1], 1) for s in ("interp", "extrap_feed", "extrap_temp")] for m in res}, flush=True)

    summary = {}
    for m in res:
        summary[m] = {s: {"median": float(np.median(v)), "q1": float(np.percentile(v, 25)),
                          "q3": float(np.percentile(v, 75)), "mean": float(np.mean(v)),
                          "sd": float(np.std(v, ddof=1)),
                          "per_state_mean_XSP": np.mean(per_state[m][s], 0).tolist()}
                      for s, v in res[m].items()}
    tests = {}
    for s in ("interp", "extrap_feed", "extrap_temp"):
        for a, b in (("hybrid", "blackbox"), ("hybrid", "textbook")):
            x, y = np.array(res[a][s]), np.array(res[b][s])
            stat, p = wilcoxon(x, y)
            d = x - y
            n = len(d); rank_biserial = (np.sum(d < 0) - np.sum(d > 0)) / n
            tests[f"{a}_vs_{b}_{s}"] = {"W": float(stat), "p": float(p),
                                        "median_diff": float(np.median(d)),
                                        "wins_for_" + a: int(np.sum(d < 0)), "n": n,
                                        "sign_effect": float(rank_biserial)}
    json.dump({"raw": res, "summary": summary, "tests": tests, "example": example},
              open("results/case_a.json", "w"), indent=1)
    print(json.dumps(summary, indent=1)); print(json.dumps(tests, indent=1))


if __name__ == "__main__":
    import os; os.makedirs("results", exist_ok=True); run()
