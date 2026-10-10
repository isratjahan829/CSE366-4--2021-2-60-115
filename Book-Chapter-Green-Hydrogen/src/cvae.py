"""Conditional variational autoencoder (CVAE) written from scratch in NumPy.

x : daily profile (24 clearness-index values + 24 standardised temperatures) = 48 dims
c : condition vector [sin(2*pi*doy/365), cos(2*pi*doy/365), site one-hot (2)]
Encoder  q(z|x,c): [x,c] -> tanh(H) -> (mu, logvar)
Decoder  p(x|z,c): [z,c] -> tanh(H) -> tanh(H) -> x_hat   (Gaussian likelihood, fixed sigma)
Trained with Adam on the negative ELBO with KL warm-up.
"""
import numpy as np


def _init(rng, n_in, n_out):
    return rng.normal(0, np.sqrt(1.0 / n_in), (n_in, n_out)), np.zeros(n_out)


class CVAE:
    def __init__(self, x_dim, c_dim, z_dim=8, hidden=96, seed=0):
        rng = np.random.default_rng(seed)
        self.z_dim = z_dim
        self.p = {}
        self.p["eW1"], self.p["eb1"] = _init(rng, x_dim + c_dim, hidden)
        self.p["eWm"], self.p["ebm"] = _init(rng, hidden, z_dim)
        self.p["eWv"], self.p["ebv"] = _init(rng, hidden, z_dim)
        self.p["dW1"], self.p["db1"] = _init(rng, z_dim + c_dim, hidden)
        self.p["dW2"], self.p["db2"] = _init(rng, hidden, hidden)
        self.p["dW3"], self.p["db3"] = _init(rng, hidden, x_dim)
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0
        self.rng = rng

    # ---------------- forward pieces ----------------
    def encode(self, x, c):
        h = np.tanh(np.hstack([x, c]) @ self.p["eW1"] + self.p["eb1"])
        return h @ self.p["eWm"] + self.p["ebm"], h @ self.p["eWv"] + self.p["ebv"]

    def decode(self, z, c):
        a1 = np.tanh(np.hstack([z, c]) @ self.p["dW1"] + self.p["db1"])
        a2 = np.tanh(a1 @ self.p["dW2"] + self.p["db2"])
        return a2 @ self.p["dW3"] + self.p["db3"]

    # ---------------- one training step ----------------
    def step(self, x, c, beta, lr=1e-3, sigma2=0.01):
        p, n = self.p, len(x)
        xc = np.hstack([x, c])
        h = np.tanh(xc @ p["eW1"] + p["eb1"])
        mu = h @ p["eWm"] + p["ebm"]
        lv = np.clip(h @ p["eWv"] + p["ebv"], -10, 5)
        eps = self.rng.standard_normal(mu.shape)
        z = mu + np.exp(0.5 * lv) * eps
        zc = np.hstack([z, c])
        a1 = np.tanh(zc @ p["dW1"] + p["db1"])
        a2 = np.tanh(a1 @ p["dW2"] + p["db2"])
        xh = a2 @ p["dW3"] + p["db3"]

        rec = 0.5 * np.sum((xh - x) ** 2) / sigma2 / n
        kl = -0.5 * np.sum(1 + lv - mu ** 2 - np.exp(lv)) / n
        g = {}
        d_xh = (xh - x) / sigma2 / n
        g["dW3"], g["db3"] = a2.T @ d_xh, d_xh.sum(0)
        d_a2 = d_xh @ p["dW3"].T * (1 - a2 ** 2)
        g["dW2"], g["db2"] = a1.T @ d_a2, d_a2.sum(0)
        d_a1 = d_a2 @ p["dW2"].T * (1 - a1 ** 2)
        g["dW1"], g["db1"] = zc.T @ d_a1, d_a1.sum(0)
        d_z = (d_a1 @ p["dW1"].T)[:, : self.z_dim]
        d_mu = d_z + beta * mu / n
        d_lv = d_z * eps * 0.5 * np.exp(0.5 * lv) + beta * 0.5 * (np.exp(lv) - 1) / n
        g["eWm"], g["ebm"] = h.T @ d_mu, d_mu.sum(0)
        g["eWv"], g["ebv"] = h.T @ d_lv, d_lv.sum(0)
        d_h = (d_mu @ p["eWm"].T + d_lv @ p["eWv"].T) * (1 - h ** 2)
        g["eW1"], g["eb1"] = xc.T @ d_h, d_h.sum(0)

        self.last_grad = g
        self.t += 1
        b1, b2 = 0.9, 0.999
        for k in p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        return rec, kl

    def fit(self, x, c, epochs=3000, batch=64, lr=1e-3, warmup=600, log_every=250):
        hist = []
        n = len(x)
        for ep in range(epochs):
            beta = min(1.0, (ep + 1) / warmup)
            perm = self.rng.permutation(n)
            r_tot = k_tot = 0.0
            for i in range(0, n, batch):
                b = perm[i: i + batch]
                r, k = self.step(x[b], c[b], beta, lr)
                r_tot += r * len(b)
                k_tot += k * len(b)
            hist.append((ep, r_tot / n, k_tot / n))
            if log_every and ep % log_every == 0:
                print(f"epoch {ep:5d}  rec {r_tot / n:9.3f}  KL {k_tot / n:7.3f}  beta {beta:.2f}")
        return np.array(hist)

    def sample(self, c, z=None):
        if z is None:
            z = self.rng.standard_normal((len(c), self.z_dim))
        return self.decode(z, c)
