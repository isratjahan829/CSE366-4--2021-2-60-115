"""Off-grid PV - battery - PEM electrolyzer - compressed H2 storage model with a rule-based
energy management system (EMS). Vectorised over (designs x scenarios)."""
import numpy as np

P = dict(
    demand_kg_day=100.0,       # constant industrial off-take
    derate=0.86, gamma=-0.004, noct=45.0,
    el_min=0.10,               # electrolyzer minimum load (fraction of rated)
    sec_a=48.0, sec_b=6.0,     # SEC(l) = a + b*l   [kWh/kg], 54 kWh/kg at rated load
    comp=2.5,                  # compression to storage pressure [kWh/kg]
    b_eta=0.95, b_crate=0.5, b_soc_min=0.10,
    tank_init=0.5,
    # economics (USD, 2025 basis), discount rate 7 %
    rate=0.07,
    pv_capex=650, pv_om=15, pv_life=25,
    el_capex=1100, el_om=0.03, el_life=20,
    b_capex=280, b_om=0.02, b_life=12,
    t_capex=500, t_om=0.01, t_life=25,
    water_l_per_kg=15.0,
    backup_price=10.0,         # trucked-in hydrogen for any shortfall [USD/kg]
)
WATER_PRICE = {"A_semiarid_CO": 3.0, "B_humid_NC": 1.5}  # USD/m3 (arid: treatment + haulage)


def geometry(site_cfg, tilt=None):
    """Hourly sun geometry for a south-facing array tilted at latitude (fixed for all scenarios)."""
    import pandas as pd
    import pvlib
    idx = pd.date_range("2017-01-01 00:30", periods=8760, freq="1h", tz=site_cfg["tz"])
    sp = pvlib.solarposition.get_solarposition(idx, site_cfg["lat"], site_cfg["lon"], site_cfg["alt"])
    tilt = site_cfg["lat"] if tilt is None else tilt
    aoi = pvlib.irradiance.aoi(tilt, 180.0, sp["apparent_zenith"], sp["azimuth"]).values
    return dict(cosz=np.cos(np.radians(sp["apparent_zenith"].values)), cosaoi=np.cos(np.radians(aoi)),
                i0=pvlib.irradiance.get_extra_radiation(idx).values, tilt=np.radians(tilt))


def poa(ghi, geo, albedo=0.2):
    """Erbs decomposition + isotropic-sky transposition, vectorised over scenarios (..., 8760)."""
    cosz = np.maximum(geo["cosz"], 0.0)
    up = cosz > 0.065
    kt = np.where(up, ghi / np.maximum(geo["i0"] * cosz, 1e-6), 0.0).clip(0, 1)
    df = np.where(kt <= 0.22, 1 - 0.09 * kt,
                  np.where(kt <= 0.8, 0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4, 0.165))
    dhi = df * ghi
    dni = np.where(up, (ghi - dhi) / np.maximum(cosz, 0.065), 0.0)
    b = geo["tilt"]
    return np.maximum(dni * np.maximum(geo["cosaoi"], 0.0) + dhi * (1 + np.cos(b)) / 2 + ghi * albedo * (1 - np.cos(b)) / 2, 0.0)


def pv_per_kwp(ghi, ta, geo=None, p=P):
    g = ghi if geo is None else poa(ghi, geo)
    tc = ta + g * (p["noct"] - 20.0) / 800.0
    return np.maximum(g / 1000.0 * p["derate"] * (1 + p["gamma"] * (tc - 25.0)), 0.0)


def crf(r, n):
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def annual_fixed_cost(x, p=P):
    pv, el, b, t = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    return (pv * p["pv_capex"] * crf(p["rate"], p["pv_life"]) + pv * p["pv_om"]
            + el * p["el_capex"] * (crf(p["rate"], p["el_life"]) + p["el_om"])
            + b * p["b_capex"] * (crf(p["rate"], p["b_life"]) + p["b_om"])
            + t * p["t_capex"] * (crf(p["rate"], p["t_life"]) + p["t_om"]))


def simulate(x, pv1, p=P, trace=False):
    """x: (N,4) designs [PV kWp, EL kW, battery kWh, tank kg]; pv1: (S,8760) kW per kWp.
    Returns dict of (N,S) arrays."""
    N, S = len(x), len(pv1)
    PV, EL, B, T = [x[:, i][:, None] * np.ones((1, S)) for i in range(4)]
    D = p["demand_kg_day"] / 24.0
    e = B * 0.5
    h = T * p["tank_init"]
    h0 = h.copy()
    pb = B * p["b_crate"]
    emin = B * p["b_soc_min"]
    sec_full = p["sec_a"] + p["sec_b"] + p["comp"]
    prod = np.zeros((N, S)); short = np.zeros((N, S)); curt = np.zeros((N, S)); hours = np.zeros((N, S))
    tr = {"pv": [], "el": [], "soc": [], "tank": []} if trace else None
    for t in range(pv1.shape[1]):
        pv = PV * pv1[:, t][None, :]
        room = np.maximum(T - h + D, 0.0)
        pcap = np.minimum(EL, room * sec_full)
        p_pv = np.minimum(pv, pcap)
        d = np.minimum(np.minimum(pcap - p_pv, pb), np.maximum(e - emin, 0.0) * p["b_eta"])
        run = (p_pv + d) >= p["el_min"] * EL
        d = np.where(run, d, 0.0)
        p_el = np.where(run, p_pv + d, 0.0)
        surplus = pv - np.where(run, p_pv, 0.0)
        c = np.minimum(np.minimum(surplus, pb), np.maximum(B - e, 0.0) / p["b_eta"])
        e = e + c * p["b_eta"] - d / p["b_eta"]
        curt += surplus - c
        l = np.where(EL > 0, p_el / np.maximum(EL, 1e-9), 0.0)
        m = p_el / (p["sec_a"] + p["sec_b"] * l + p["comp"])
        h = h + m
        sup = np.minimum(h, D)
        short += D - sup
        h -= sup
        prod += m
        hours += run
        if trace:
            tr["pv"].append(pv[0, 0]); tr["el"].append(p_el[0, 0]); tr["soc"].append(e[0, 0]); tr["tank"].append(h[0, 0])
    short += np.maximum(h0 - h, 0.0)  # terminal condition: refill the tank to its initial level
    out = dict(prod=prod, short=short, curt=curt, hours=hours)
    if trace:
        out["trace"] = {k: np.array(v) for k, v in tr.items()}
    return out


def costs(x, sim, water_price, p=P):
    fixed = annual_fixed_cost(x, p)[:, None]
    water = sim["prod"] * p["water_l_per_kg"] / 1000.0 * water_price
    backup = sim["short"] * p["backup_price"]
    delivered = p["demand_kg_day"] * 365.0
    total = fixed + water + backup
    return dict(total=total, lcoh=total / delivered, reliability=1 - sim["short"] / delivered,
                fixed=np.broadcast_to(fixed, total.shape))


def cvar(v, alpha=0.9, axis=1):
    q = np.quantile(v, alpha, axis=axis, keepdims=True)
    tail = np.where(v >= q, v, np.nan)
    return np.nanmean(tail, axis=axis)
