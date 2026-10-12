"""NSGA-II (Deb et al. 2002) written from scratch: fast non-dominated sorting, crowding distance,
binary tournament on (rank, crowding), simulated binary crossover (SBX) and polynomial mutation.
Plus Monte Carlo hypervolume and TOPSIS."""
import numpy as np


def nondominated_sort(F):
    n = len(F)
    dom = (F[:, None, :] <= F[None, :, :]).all(-1) & (F[:, None, :] < F[None, :, :]).any(-1)  # i dominates j
    n_dom = dom.sum(0)
    ranks = np.full(n, -1)
    front, r = np.where(n_dom == 0)[0], 0
    while len(front):
        ranks[front] = r
        n_dom = n_dom - dom[front].sum(0)
        n_dom[ranks >= 0] = -1
        front = np.where(n_dom == 0)[0]
        r += 1
    return ranks


def crowding(F, ranks):
    cd = np.zeros(len(F))
    for r in np.unique(ranks):
        idx = np.where(ranks == r)[0]
        if len(idx) <= 2:
            cd[idx] = np.inf
            continue
        for k in range(F.shape[1]):
            o = idx[np.argsort(F[idx, k])]
            span = F[o[-1], k] - F[o[0], k] or 1.0
            cd[o[0]] = cd[o[-1]] = np.inf
            cd[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / span
    return cd


def sbx(a, b, lo, hi, rng, eta=15):
    u = rng.random(len(a))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * a + (1 - beta) * b)
    c2 = 0.5 * ((1 - beta) * a + (1 + beta) * b)
    swap = rng.random(len(a)) < 0.5
    c1[swap], c2[swap] = c2[swap].copy(), c1[swap].copy()
    return np.clip(c1, lo, hi), np.clip(c2, lo, hi)


def poly_mut(x, lo, hi, rng, pm, eta=20):
    y = x.copy()
    for i in np.where(rng.random(len(x)) < pm)[0]:
        u = rng.random()
        d = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        y[i] = np.clip(y[i] + d * (hi[i] - lo[i]), lo[i], hi[i])
    return y


def run_nsga2(evaluate, lo, hi, pop=60, gens=60, seed=0, hv_ref=None, hv_scale=None):
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    X = lo + rng.random((pop, len(lo))) * (hi - lo)
    F = evaluate(X)
    hv_hist = []
    for g in range(gens):
        ranks = nondominated_sort(F)
        cd = crowding(F, ranks)
        # binary tournament
        a, b = rng.integers(0, pop, (2, pop))
        better = (ranks[a] < ranks[b]) | ((ranks[a] == ranks[b]) & (cd[a] > cd[b]))
        par = X[np.where(better, a, b)]
        kids = []
        for i in range(0, pop, 2):
            c1, c2 = sbx(par[i], par[(i + 1) % pop], lo, hi, rng) if rng.random() < 0.9 else (par[i].copy(), par[(i + 1) % pop].copy())
            kids += [poly_mut(c1, lo, hi, rng, 1 / len(lo)), poly_mut(c2, lo, hi, rng, 1 / len(lo))]
        K = np.array(kids[:pop])
        FK = evaluate(K)
        XA, FA = np.vstack([X, K]), np.vstack([F, FK])
        ranks = nondominated_sort(FA)
        cd = crowding(FA, ranks)
        order = np.lexsort((-cd, ranks))[:pop]
        X, F = XA[order], FA[order]
        if hv_ref is not None and (g % 5 == 0 or g == gens - 1):
            hv_hist.append((g, hypervolume(F[nondominated_sort(F) == 0], hv_ref, hv_scale)))
    first = nondominated_sort(F) == 0
    return X[first], F[first], hv_hist


def hypervolume(F, ref, scale, n=200000, seed=1):
    """Monte Carlo hypervolume (minimisation) of the box [ideal, ref], normalised to [0,1]."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(scale)
    ref = np.asarray(ref)
    Z = lo + rng.random((n, len(ref))) * (ref - lo)
    Fn = F[(F < ref).all(1)]
    if len(Fn) == 0:
        return 0.0
    dominated = np.zeros(n, bool)
    for f in Fn:
        dominated |= (Z >= f).all(1)
    return float(dominated.mean())


def topsis(F, w, benefit=None):
    """Rank alternatives F (n x k, all costs unless benefit[k]) with weights w. Returns closeness (higher=better)."""
    N = F / np.sqrt((F ** 2).sum(0))
    V = N * w
    benefit = np.zeros(F.shape[1], bool) if benefit is None else np.asarray(benefit)
    ideal = np.where(benefit, V.max(0), V.min(0))
    anti = np.where(benefit, V.min(0), V.max(0))
    dp = np.sqrt(((V - ideal) ** 2).sum(1))
    dn = np.sqrt(((V - anti) ** 2).sum(1))
    return dn / (dp + dn)
