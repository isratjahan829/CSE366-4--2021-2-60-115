"""Real-coded genetic algorithm (tournament selection, BLX-alpha crossover, Gaussian mutation,
elitism) - the same operator family as the CSE366 GA assignment, extended to continuous sizing."""
import numpy as np


def run_ga(fitness, lo, hi, pop=40, gens=60, k=3, pc=0.9, alpha=0.5, pm=0.2, sigma=0.1,
           elite=2, seed=0, log=None):
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    X = lo + rng.random((pop, len(lo))) * (hi - lo)
    f = fitness(X)
    best_hist = []
    for g in range(gens):
        order = np.argsort(f)
        best_hist.append(float(f[order[0]]))
        if log and g % 10 == 0:
            log(f"gen {g:3d}  best {f[order[0]]:.4f}  x={np.round(X[order[0]], 1)}")
        # tournament selection (minimisation)
        cand = rng.integers(0, pop, (pop, k))
        parents = X[cand[np.arange(pop), np.argmin(f[cand], axis=1)]]
        # BLX-alpha crossover
        kids = parents.copy()
        for i in range(0, pop - 1, 2):
            if rng.random() < pc:
                a, b = parents[i], parents[i + 1]
                mn, mx = np.minimum(a, b), np.maximum(a, b)
                span = mx - mn
                kids[i] = rng.uniform(mn - alpha * span, mx + alpha * span)
                kids[i + 1] = rng.uniform(mn - alpha * span, mx + alpha * span)
        # Gaussian mutation
        mask = rng.random(kids.shape) < pm
        kids = kids + mask * rng.normal(0, sigma, kids.shape) * (hi - lo)
        kids = np.clip(kids, lo, hi)
        # elitism
        kids[:elite] = X[order[:elite]]
        fk = np.empty(pop)
        fk[:elite] = f[order[:elite]]
        fk[elite:] = fitness(kids[elite:])
        X, f = kids, fk
    order = np.argsort(f)
    best_hist.append(float(f[order[0]]))
    return X[order[0]], float(f[order[0]]), np.array(best_hist)


def random_search(fitness, lo, hi, n_eval, batch=40, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    best_x, best_f, hist = None, np.inf, []
    for _ in range(n_eval // batch):
        X = lo + rng.random((batch, len(lo))) * (hi - lo)
        f = fitness(X)
        i = int(np.argmin(f))
        if f[i] < best_f:
            best_x, best_f = X[i], float(f[i])
        hist.append(best_f)
    return best_x, best_f, np.array(hist)
