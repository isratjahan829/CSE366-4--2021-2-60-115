"""Case study D: Bayesian optimisation of temperature and feed exponent vs random search vs DoE/RSM.

Objective: final product mass P*V (g) after 16 h in the virtual plant, measured with 3 % noise.
Budget: 15 bioreactor runs per campaign; 20 independent campaigns per strategy.
"""
import json, warnings
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from plant import simulate
warnings.filterwarnings("ignore")

LO, HI = np.array([30.0, 0.08]), np.array([38.0, 0.32])
BUDGET, N_INIT, N_CAMPAIGNS = 15, 5, 20


def true_obj(x):
    x = np.atleast_2d(x); _, Y = simulate(x[:, 0], x[:, 1])
    return Y[-1, :, 2] * Y[-1, :, 3]


def noisy(x, rng):
    v = true_obj(x); return v * (1 + 0.03 * rng.standard_normal(v.shape))


def to_unit(x): return (x - LO) / (HI - LO)
def from_unit(u): return LO + u * (HI - LO)


def bayes_opt(rng):
    U = rng.random((N_INIT, 2)); y = list(noisy(from_unit(U), rng)); U = list(U)
    cand = rng.random((4000, 2))
    while len(y) < BUDGET:
        kern = ConstantKernel(1.0) * Matern(length_scale=[0.3, 0.3], nu=2.5) + WhiteKernel(1e-3)
        gp = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=3,
                                      random_state=int(rng.integers(1e9))).fit(np.array(U), np.array(y))
        mu, sd = gp.predict(cand, return_std=True)
        best = np.max(gp.predict(np.array(U)))
        z = (mu - best - 0.01) / np.maximum(sd, 1e-9)
        ei = (mu - best - 0.01) * norm.cdf(z) + sd * norm.pdf(z)          # expected improvement
        u = cand[np.argmax(ei)]
        U.append(u); y.append(noisy(from_unit(u[None]), rng)[0])
    # recommend the point with best GP posterior mean among evaluated points
    rec = from_unit(np.array(U)[np.argmax(gp.predict(np.array(U)))])
    return np.array(y), rec


def random_search(rng):
    U = rng.random((BUDGET, 2)); y = noisy(from_unit(U), rng)
    return y, from_unit(U[np.argmax(y)])


def doe_rsm(rng):
    """Face-centred central composite design (9 runs, 2 centre replicates = 10) + quadratic RSM,
    then sequential confirmation runs at the predicted optimum region (5 runs)."""
    pts = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1], [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0], [0, 0]])
    U = (pts + 1) / 2; y = noisy(from_unit(U), rng)
    def quad(U):
        a, b = U[:, 0], U[:, 1]; return np.c_[np.ones(len(U)), a, b, a * b, a ** 2, b ** 2]
    beta = np.linalg.lstsq(quad(U), y, rcond=None)[0]
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101)), -1).reshape(-1, 2)
    u_opt = grid[np.argmax(quad(grid) @ beta)]
    conf = np.clip(u_opt + rng.normal(0, 0.05, (5, 2)), 0, 1)
    yc = noisy(from_unit(conf), rng)
    allU, ally = np.vstack([U, conf]), np.r_[y, yc]
    return ally, from_unit(allU[np.argmax(ally)])


def run():
    g = np.stack(np.meshgrid(np.linspace(0, 1, 81), np.linspace(0, 1, 81)), -1).reshape(-1, 2)
    vals = true_obj(from_unit(g)); x_star = from_unit(g[np.argmax(vals)]); f_star = float(vals.max())
    res = {}
    for name, fn in (("Bayesian optimisation", bayes_opt), ("Random search", random_search), ("DoE + RSM (CCD)", doe_rsm)):
        curves, finals, recs = [], [], []
        for c in range(N_CAMPAIGNS):
            rng = np.random.default_rng(1000 + c)
            y, rec = fn(rng)
            curves.append(np.maximum.accumulate(y).tolist())
            f_rec = float(true_obj(rec[None])[0]); finals.append(f_rec); recs.append(rec.tolist())
        finals = np.array(finals)
        res[name] = {"curves": curves, "true_value_of_recommendation": finals.tolist(),
                     "mean_pct_of_optimum": float(np.mean(finals) / f_star * 100), "median_pct_of_optimum": float(np.median(finals) / f_star * 100),
                     "sd_pct_of_optimum": float(np.std(finals, ddof=1) / f_star * 100),
                     "worst_pct_of_optimum": float(np.min(finals) / f_star * 100),
                     "campaigns_within_5pct": int(np.sum(finals >= 0.95 * f_star)),
                     "recommendations": recs}
        print(name, round(res[name]["mean_pct_of_optimum"], 1), round(res[name]["sd_pct_of_optimum"], 1),
              res[name]["campaigns_within_5pct"], flush=True)
    from scipy.stats import mannwhitneyu
    bo = np.array(res["Bayesian optimisation"]["true_value_of_recommendation"])
    tests = {k: float(mannwhitneyu(bo, np.array(res[k]["true_value_of_recommendation"]), alternative="greater").pvalue)
             for k in ("Random search", "DoE + RSM (CCD)")}
    json.dump({"x_star": x_star.tolist(), "f_star": f_star, "results": res, "mannwhitney_p_BO_greater": tests,
               "landscape": {"u": g.tolist(), "f": vals.tolist()}}, open("results/case_d.json", "w"))
    print("optimum", x_star, f_star, tests)


if __name__ == "__main__":
    run()
