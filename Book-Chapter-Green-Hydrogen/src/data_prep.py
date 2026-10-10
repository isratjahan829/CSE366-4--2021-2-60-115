"""Download-free data preparation for the green-hydrogen chapter.

Two open solar datasets (fetched from the pvlib GitHub repository, see README):
  * Site A  - NSRDB PSM3 2017, 40.53N 108.54W, 2168 m (semi-arid high desert, Colorado, USA), 30-min
  * Site B  - TMY3 723170 Greensboro, NC (humid subtropical, USA), hourly

Both are converted to hourly GHI [W/m2] and air temperature [C], a clear-sky GHI is
computed with the Ineichen model, and the hourly clearness index kt = GHI / GHI_cs is
stored per day (365 x 24).  Outputs go to ../data/processed.csv and ../data/processed.xlsx.
"""
import os
import numpy as np
import pandas as pd
import pvlib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

SITES = {
    "A_semiarid_CO": dict(lat=40.53, lon=-108.54, alt=2168, tz="Etc/GMT+7"),
    "B_humid_NC": dict(lat=36.10, lon=-79.95, alt=273, tz="Etc/GMT+5"),
}


def load_site_a():
    df = pd.read_csv(os.path.join(DATA, "nsrdb_psm3_read.csv"), skiprows=2)
    idx = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    df.index = idx.dt.tz_localize(SITES["A_semiarid_CO"]["tz"])
    hourly = df[["GHI", "Temperature"]].astype(float).resample("1h").mean()
    return hourly.rename(columns={"Temperature": "Tair"})


def load_site_b():
    data, _ = pvlib.iotools.read_tmy3(os.path.join(DATA, "tmy3_723170.csv"), map_variables=True)
    # TMY3 stamps are end-of-hour; shift to start-of-hour and force a single year
    idx = pd.date_range("2017-01-01 00:00", periods=8760, freq="1h", tz=SITES["B_humid_NC"]["tz"])
    return pd.DataFrame({"GHI": data["ghi"].values, "Tair": data["temp_air"].values}, index=idx)


def add_clearsky(df, site):
    s = SITES[site]
    loc = pvlib.location.Location(s["lat"], s["lon"], s["tz"], s["alt"])
    mid = df.index + pd.Timedelta(minutes=30)  # hour-centred sun position
    cs = loc.get_clearsky(mid, model="ineichen")
    df = df.copy()
    df["GHI_cs"] = cs["ghi"].values
    kt = np.where(df["GHI_cs"] > 20.0, df["GHI"] / df["GHI_cs"].clip(lower=1e-6), 0.0)
    df["kt"] = np.clip(kt, 0.0, 1.3)
    return df


def build():
    frames = []
    for site, loader in [("A_semiarid_CO", load_site_a), ("B_humid_NC", load_site_b)]:
        df = add_clearsky(loader(), site).iloc[:8760]
        df["site"] = site
        df["doy"] = np.repeat(np.arange(365), 24)
        df["hour"] = np.tile(np.arange(24), 365)
        frames.append(df.reset_index(names="time"))
    out = pd.concat(frames, ignore_index=True)
    out["time"] = out["time"].astype(str)
    out.to_csv(os.path.join(DATA, "processed.csv"), index=False)
    summary = out.groupby("site").agg(GHI_kWh_m2_yr=("GHI", lambda x: x.sum() / 1000),
                                      Tair_mean=("Tair", "mean"), kt_mean=("kt", lambda x: x[x > 0].mean()))
    with pd.ExcelWriter(os.path.join(DATA, "processed.xlsx")) as xw:
        summary.to_excel(xw, sheet_name="summary")
        for site, g in out.groupby("site"):
            g.to_excel(xw, sheet_name=site[:31], index=False)
    print(summary)
    return out


def daily_arrays(df, site):
    g = df[df.site == site]
    kt = g["kt"].values.reshape(365, 24)
    ta = g["Tair"].values.reshape(365, 24)
    cs = g["GHI_cs"].values.reshape(365, 24)
    ghi = g["GHI"].values.reshape(365, 24)
    return kt, ta, cs, ghi


if __name__ == "__main__":
    build()
