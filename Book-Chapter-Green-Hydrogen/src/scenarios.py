"""Scenario generation (CVAE + latent AR(1)) and baselines, plus probabilistic scores."""
import numpy as np
from cvae import CVAE

SITES = ["A_semiarid_CO", "B_humid_NC"]
T_SCALE = 0.5  # temperature z-scores are down-weighted so kt dominates the reconstruction loss


def cond(doy, site_idx):
    a = 2 * np.pi * np.asarray(doy) / 365.0
    oh = np.zeros((len(a), len(SITES)))
    oh[np.arange(len(a)), site_idx] = 1.0
    return np.column_stack([np.sin(a), np.cos(a), oh])


class SiteData:
    def __init__(self, kt, ta, cs, ghi, site_idx):
        self.kt, self.ta, self.cs, self.ghi, self.idx = kt, ta, cs, ghi, site_idx
        self.t_mu, self.t_sd = ta.mean(), ta.std()

    def x(self, days):
        return np.hstack([self.kt[days], T_SCALE * (self.ta[days] - self.t_mu) / self.t_sd])

    def unpack(self, x):
        kt = np.clip(x[:, :24], 0.0, 1.3)
        ta = x[:, 24:] / T_SCALE * self.t_sd + self.t_mu
        return kt, ta


def split_days(seed=1):
    days = np.arange(365)
    test = days[days % 5 == 2]  # every 5th day held out (73 days), spread across seasons
    train = np.setdiff1d(days, test)
    return train, test


def train_cvae(sites, train, epochs=3000, seed=0):
    X = np.vstack([s.x(train) for s in sites])
    C = np.vstack([cond(train, s.idx) for s in sites])
    model = CVAE(X.shape[1], C.shape[1], z_dim=8, hidden=96, seed=seed)
    hist = model.fit(X, C, epochs=epochs)
    return model, hist


# ------------------------------------------------------------------ samplers
def sample_cvae(model, site, doys, n, rng, z=None):
    """n samples for each doy -> kt, ta arrays shape (len(doys), n, 24)."""
    d = np.repeat(np.asarray(doys), n)
    zz = rng.standard_normal((len(d), model.z_dim)) if z is None else z
    kt, ta = site.unpack(model.decode(zz, cond(d, site.idx)))
    return kt.reshape(len(doys), n, 24), ta.reshape(len(doys), n, 24)


def sample_bootstrap(site, train, doys, n, rng, window=15):
    kts, tas = [], []
    for d in doys:
        dist = np.minimum(np.abs(train - d), 365 - np.abs(train - d))
        pool = train[dist <= window]
        pick = rng.choice(pool, n)
        kts.append(site.kt[pick])
        tas.append(site.ta[pick])
    return np.array(kts), np.array(tas)


def fit_gaussian(site, train):
    """Month-wise multivariate normal on the 48-dim daily vector (shrunk covariance)."""
    month = (train / 30.42).astype(int).clip(0, 11)
    par = {}
    for m in range(12):
        X = site.x(train[month == m])
        mu = X.mean(0)
        S = np.cov(X.T)
        S = 0.8 * S + 0.2 * np.diag(np.diag(S)) + 1e-6 * np.eye(len(mu))
        par[m] = (mu, np.linalg.cholesky(S))
    return par


def sample_gaussian(site, par, doys, n, rng):
    kts, tas = [], []
    for d in doys:
        mu, L = par[min(int(d / 30.42), 11)]
        X = mu + rng.standard_normal((n, len(mu))) @ L.T
        X[:, :24] *= site.kt[:, :].max(0) > 0  # night hours stay zero
        kt, ta = site.unpack(X)
        kts.append(kt)
        tas.append(ta)
    return np.array(kts), np.array(tas)


# ------------------------------------------------------------------ scores
def energy_score(samples, obs):
    """samples (n, k), obs (k,) -> energy score (lower is better)."""
    t1 = np.mean(np.linalg.norm(samples - obs, axis=1))
    diff = samples[:, None, :] - samples[None, :, :]
    t2 = 0.5 * np.mean(np.linalg.norm(diff, axis=2))
    return t1 - t2


def crps(samples, obs):
    s = np.sort(samples)
    return np.mean(np.abs(s - obs)) - 0.5 * np.mean(np.abs(s[:, None] - s[None, :]))


def evaluate(site, test, kt_s, ta_s):
    """kt_s,ta_s: (len(test), n, 24). Scores on GHI profiles [kW/m2] and daily insolation [kWh/m2]."""
    es, cr, cov = [], [], []
    for i, d in enumerate(test):
        ghi_s = kt_s[i] * site.cs[d] / 1000.0
        ghi_o = site.ghi[d] / 1000.0
        es.append(energy_score(ghi_s, ghi_o))
        e_s, e_o = ghi_s.sum(1), ghi_o.sum()
        cr.append(crps(e_s, e_o))
        lo, hi = np.quantile(e_s, [0.05, 0.95])
        cov.append(lo <= e_o <= hi)
        _ = ta_s
    return dict(ES=float(np.mean(es)), CRPS=float(np.mean(cr)), PICP90=float(np.mean(cov)))


def latent_rho(model, site):
    mu, _ = model.encode(site.x(np.arange(365)), cond(np.arange(365), site.idx))
    r = [np.corrcoef(mu[:-1, j], mu[1:, j])[0, 1] for j in range(mu.shape[1])]
    return float(np.clip(np.nanmean(r), 0.0, 0.95))


def year_scenarios(model, site, S, rng, rho):
    """S synthetic years (S, 8760) of GHI [W/m2] and Tair [C] with day-to-day latent persistence."""
    z = np.zeros((S, 365, model.z_dim))
    z[:, 0] = rng.standard_normal((S, model.z_dim))
    for d in range(1, 365):
        z[:, d] = rho * z[:, d - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal((S, model.z_dim))
    doys = np.tile(np.arange(365), S)
    kt, ta = site.unpack(model.decode(z.reshape(-1, model.z_dim), cond(doys, site.idx)))
    ghi = kt.reshape(S, 365, 24) * site.cs[None]
    return ghi.reshape(S, 8760), ta.reshape(S, 8760)


def bootstrap_years(site, S, rng, window=15):
    days = np.arange(365)
    ghi, ta = np.zeros((S, 365, 24)), np.zeros((S, 365, 24))
    for d in days:
        dist = np.minimum(np.abs(days - d), 365 - np.abs(days - d))
        pick = rng.choice(days[dist <= window], S)
        ghi[:, d] = site.kt[pick] * site.cs[d]
        ta[:, d] = site.ta[pick]
    return ghi.reshape(S, 8760), ta.reshape(S, 8760)
