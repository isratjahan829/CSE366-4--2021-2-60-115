"""Chapter 2: degradation-aware model predictive control (MPC) of an off-grid PV-battery-PEM-H2 plant.

Two-layer energy management:
  * day-ahead layer: a mixed-integer linear programme (MILP, HiGHS via scipy.optimize.milp) solved every
    24 h over a 48 h horizon with a PV forecast; decides electrolyser on/off commitment and power set-points;
  * real-time layer: follows the commitment with the actual PV, using the battery to hold the set-point.
Degradation (assumption, see chapter): wear cost per operating hour grows with load, plus a cost per cold start.
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

import h2system as hs

P = hs.P
SEC_FULL = P["sec_a"] + P["sec_b"] + P["comp"]          # 56.5 kWh/kg at rated load


def deg_params(el_kw, start_hours=10.0, stack_share=0.35, life_h=60000.0):
    """Stack replacement cost spread over its life; cost per hour at load l is c_h*(0.5+0.5 l);
    a cold start costs `start_hours` equivalent full-load hours."""
    stack = stack_share * P["el_capex"] * el_kw
    c_h = stack / life_h
    return dict(c_h=c_h, c_start=start_hours * c_h)


def h2_rate(p, el_kw):
    l = np.where(el_kw > 0, p / el_kw, 0.0)
    return np.where(p > 0, p / (P["sec_a"] + P["sec_b"] * l + P["comp"]), 0.0)


def linear_h2(el_kw):
    """Secant through (0.1 EL, m) and (EL, m): m ~ a*P + b*u."""
    p1, p2 = 0.1 * el_kw, el_kw
    m1, m2 = h2_rate(np.array(p1), el_kw), h2_rate(np.array(p2), el_kw)
    a = (m2 - m1) / (p2 - p1)
    return float(a), float(m1 - a * p1)


def solve_day(pv_fc, x, e0, h0, u0, deg, degrade=True, backup=P["backup_price"]):
    """MILP over horizon H. Variables per hour: p, u, s, c, d, cu, sh  and states e, h (H+1)."""
    PV_, EL, B, T = x
    H = len(pv_fc)
    D = P["demand_kg_day"] / 24.0
    a, b = linear_h2(EL)
    eta, pb = P["b_eta"], B * P["b_crate"]
    nv = 7 * H + 2 * (H + 1)
    ip = lambda t: t; iu = lambda t: H + t; is_ = lambda t: 2 * H + t; ic = lambda t: 3 * H + t
    id_ = lambda t: 4 * H + t; icu = lambda t: 5 * H + t; ish = lambda t: 6 * H + t
    ie = lambda t: 7 * H + t; ih = lambda t: 7 * H + (H + 1) + t
    cost = np.zeros(nv)
    for t in range(H):
        cost[ish(t)] = backup
        if degrade:
            cost[iu(t)] += 0.5 * deg["c_h"]
            cost[ip(t)] += 0.5 * deg["c_h"] / EL
            cost[is_(t)] += deg["c_start"]
        cost[icu(t)] = 1e-4
    cost[ih(H)] = -0.9 * backup            # terminal value of stored hydrogen
    cost[ie(H)] = -0.9 * backup / SEC_FULL * eta   # terminal value of stored electricity
    lo = np.zeros(nv); hi = np.full(nv, np.inf)
    for t in range(H):
        hi[ip(t)] = EL; hi[iu(t)] = 1; hi[is_(t)] = 1; hi[ic(t)] = pb; hi[id_(t)] = pb; hi[ish(t)] = D
    for t in range(H + 1):
        lo[ie(t)] = B * P["b_soc_min"]; hi[ie(t)] = B; hi[ih(t)] = T
    lo[ie(0)] = hi[ie(0)] = e0
    lo[ih(0)] = hi[ih(0)] = h0
    integ = np.zeros(nv); integ[[iu(t) for t in range(H)]] = 1; integ[[is_(t) for t in range(H)]] = 1
    nc = 6 * H
    A = lil_matrix((nc, nv)); lb = np.zeros(nc); ub = np.zeros(nc); r = 0
    for t in range(H):
        # power balance: pv + d = p + c + cu
        A[r, id_(t)] = 1; A[r, ip(t)] = -1; A[r, ic(t)] = -1; A[r, icu(t)] = -1; lb[r] = ub[r] = -pv_fc[t]; r += 1
        # p <= EL u ; p >= 0.1 EL u
        A[r, ip(t)] = 1; A[r, iu(t)] = -EL; lb[r] = -np.inf; ub[r] = 0; r += 1
        A[r, ip(t)] = 1; A[r, iu(t)] = -P["el_min"] * EL; lb[r] = 0; ub[r] = np.inf; r += 1
        # battery: e' = e + eta c - d/eta
        A[r, ie(t + 1)] = 1; A[r, ie(t)] = -1; A[r, ic(t)] = -eta; A[r, id_(t)] = 1 / eta; r += 1
        # tank: h' = h + a p + b u - D + sh
        A[r, ih(t + 1)] = 1; A[r, ih(t)] = -1; A[r, ip(t)] = -a; A[r, iu(t)] = -b; A[r, ish(t)] = -1
        lb[r] = ub[r] = -D; r += 1
        # start: s >= u_t - u_{t-1}
        A[r, is_(t)] = 1; A[r, iu(t)] = -1
        if t > 0:
            A[r, iu(t - 1)] = 1; lb[r] = 0
        else:
            lb[r] = -u0
        ub[r] = np.inf; r += 1
    res = milp(cost, constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=integ,
               bounds=Bounds(lo, hi), options={"time_limit": 3, "mip_rel_gap": 1e-2, "disp": False})
    if res.x is None:
        return None
    z = res.x
    return dict(p=z[:H], u=np.round(z[H:2 * H]), c=z[3 * H:4 * H], d=z[4 * H:5 * H])


def realtime_step(pv, setp, on, st, x):
    """Real-time layer for one hour. st = dict(e, h). Returns produced kg, shortfall, curtail, started-flag info."""
    PV_, EL, B, T = x
    D = P["demand_kg_day"] / 24.0
    eta, pb, emin = P["b_eta"], B * P["b_crate"], B * P["b_soc_min"]
    room = max(T - st["h"] + D, 0.0)
    pcap = min(EL, room * SEC_FULL)
    p = 0.0; d = 0.0
    if on:
        target = min(max(setp, pv), pcap)            # never curtail PV the electrolyser could absorb
        p_pv = min(pv, target)
        d = min(target - p_pv, pb, max(st["e"] - emin, 0.0) * eta)
        p = p_pv + d
        if p < P["el_min"] * EL:                     # cannot hold minimum load -> trip
            p, d = 0.0, 0.0
    surplus = pv - min(pv, p)
    c = min(surplus, pb, max(B - st["e"], 0.0) / eta)
    st["e"] += c * eta - d / eta
    m = float(h2_rate(np.array(p), EL))
    st["h"] += m
    sup = min(st["h"], D)
    st["h"] -= sup
    return p, m, D - sup, surplus - c


def run_year(pv1, x, strategy, fc=None, horizon=48, deg=None, seed=0):
    """pv1: (8760,) kW/kWp actual. fc: function(day, start_hour, horizon)-> forecast kW/kWp. Returns metrics."""
    PV_, EL, B, T = x
    pv = pv1 * PV_
    deg = deg or deg_params(EL)
    st = dict(e=0.5 * B, h=0.5 * T)
    h_init = st["h"]
    u_prev = 0
    prod = short = curt = starts = hours = wear = 0.0
    loads = []
    if strategy == "rule":
        sim = hs.simulate(np.array([x]), pv1[None], trace=True)
        p_el = sim["trace"]["el"]
        on = p_el > 0
        starts = int(((on[1:]) & (~on[:-1])).sum() + on[0])
        l = p_el[on] / EL
        wear = float(np.sum(0.5 + 0.5 * l) * deg["c_h"] + starts * deg["c_start"])
        return dict(prod=float(sim["prod"][0, 0]), short=float(sim["short"][0, 0]), curt=float(sim["curt"][0, 0]),
                    starts=starts, hours=int(on.sum()), mean_load=float(l.mean()), wear=wear)
    degrade = strategy != "mpc_energy"
    for day in range(365):
        t0 = day * 24
        Hh = min(horizon, 8760 - t0)
        f = pv[t0:t0 + Hh] if fc is None else fc(t0, Hh) * PV_
        plan = solve_day(f, x, st["e"], st["h"], u_prev, deg, degrade=degrade)
        if plan is None:
            plan = dict(p=np.minimum(f, EL), u=(f >= 0.1 * EL).astype(float))
        for k in range(24):
            t = t0 + k
            on = plan["u"][k] > 0.5
            p, m, sh, cu = realtime_step(pv[t], plan["p"][k], on, st, x)
            run = p > 0
            if run and not u_prev:
                starts += 1
            if run:
                hours += 1; loads.append(p / EL); wear += (0.5 + 0.5 * p / EL) * deg["c_h"]
            u_prev = int(run)
            prod += m; short += sh; curt += cu
    short += max(h_init - st["h"], 0.0)
    wear += starts * deg["c_start"]
    return dict(prod=prod, short=short, curt=curt, starts=starts, hours=hours,
                mean_load=float(np.mean(loads)) if loads else 0.0, wear=float(wear))


# ------------------------------------------------------------------ forecasts
def make_forecasts(pv1, cs1, rng, sigma=0.25, rho=0.6):
    """Day-ahead style forecasts of kW/kWp.
    persistence: yesterday's daily clearness ratio times the clear-sky PV profile of the forecast hours;
    noisy: actual daily energy times a lognormal error (sd sigma, AR(1) across days rho), actual hourly shape."""
    daily = pv1.reshape(365, 24).sum(1)
    dcs = cs1.reshape(365, 24).sum(1)
    ratio = np.where(dcs > 0, daily / np.maximum(dcs, 1e-9), 0)
    eps = np.zeros(366)
    for k in range(1, 366):
        eps[k] = rho * eps[k - 1] + np.sqrt(1 - rho ** 2) * rng.normal(0, sigma)
    err = np.exp(eps - sigma ** 2 / 2)

    def persistence(t0, H):
        d = max(t0 // 24 - 1, 0)
        return np.clip(ratio[d] * cs1[t0:t0 + H], 0, None)

    def noisy(t0, H):
        out = pv1[t0:t0 + H].copy()
        for k in range(0, H, 24):
            dd = min((t0 + k) // 24, 364)
            lead = k // 24                                         # error grows with lead day
            out[k:k + 24] *= err[dd] if lead == 0 else err[dd] * np.exp(rng.normal(0, sigma / 2))
        return np.clip(out, 0, None)
    return persistence, noisy
