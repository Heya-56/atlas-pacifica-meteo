#!/usr/bin/env python3
"""
Isobares + centres de pression (D/A) -> GeoJSON
Source : ARPEGE 0.25° (Météo-France open data) avec repli GFS 0.25° (NOAA NOMADS).
Domaine : Pacifique ou Polynésie française. Longitudes en 0–360 (carte centrée Pacifique).

Usage :
  python isobars.py --domain pacific --out out/
  python isobars.py --domain pf --source gfs --step 2
  python isobars.py --grib fichier.grib2          # fichier local
"""
import argparse, datetime as dt, json, os, sys, tempfile, urllib.request

import numpy as np
import xarray as xr
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOMAINS = {  # lat_min, lat_max, lon_min, lon_max (0–360)
    "pacific": (-50, 30, 130, 270),
    "pf":      (-32, -5, 205, 230),  # Polynésie française élargie
}

# ---------------------------------------------------------------- Téléchargement
def latest_runs(lag_h=5, n=4):
    """Runs synoptiques (00/06/12/18Z) les plus récents probablement disponibles."""
    t = dt.datetime.utcnow() - dt.timedelta(hours=lag_h)
    t = t.replace(minute=0, second=0, microsecond=0, hour=t.hour - t.hour % 6)
    return [t - dt.timedelta(hours=6 * i) for i in range(n)]

def fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "hinova-isobars/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest

ARPEGE_ERR = []

def get_arpege(tmp):
    # Paquet SP1 (surface) ARPEGE 0.25° publié sur meteo.data.gouv.fr
    for run in latest_runs():
        iso = run.strftime("%Y-%m-%dT%H:00:00Z")
        url = (f"https://object.data.gouv.fr/meteofrance-pnt/pnt/{iso}/arpege/025/SP1/"
               f"arpege__025__SP1__000H024H__{iso}.grib2")
        try:
            return fetch(url, os.path.join(tmp, "arpege.grib2")), run, "ARPEGE 0.25"
        except Exception as e:
            print(f"[arpege] {iso} indisponible ({e})", file=sys.stderr)
            ARPEGE_ERR.append(f"{url} -> {e}")
    return None

def get_gfs(tmp, dom):
    la0, la1, lo0, lo1 = dom
    for run in latest_runs(lag_h=4):
        d, h = run.strftime("%Y%m%d"), run.strftime("%H")
        url = ("https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
               f"dir=%2Fgfs.{d}%2F{h}%2Fatmos&file=gfs.t{h}z.pgrb2.0p25.f000"
               "&var_PRMSL=on&lev_mean_sea_level=on&subregion="
               f"&toplat={la1}&bottomlat={la0}&leftlon={lo0}&rightlon={lo1}")
        try:
            return fetch(url, os.path.join(tmp, "gfs.grib2")), run, "GFS 0.25"
        except Exception as e:
            print(f"[gfs] {d}{h} indisponible ({e})", file=sys.stderr)
    return None

# ---------------------------------------------------------------- Lecture
def load_pmsl(path, dom):
    ds = xr.open_dataset(path, engine="cfgrib",
                         backend_kwargs={"filter_by_keys": {"typeOfLevel": "meanSea"},
                                         "indexpath": ""})
    var = next(v for v in ("prmsl", "msl", "pmsl") if v in ds)
    da = ds[var]
    if "step" in da.dims:           # paquet multi-échéances : analyse uniquement
        da = da.isel(step=0)
    da = da.assign_coords(longitude=(da.longitude % 360)).sortby("longitude").sortby("latitude")
    la0, la1, lo0, lo1 = dom
    da = da.sel(latitude=slice(la0, la1), longitude=slice(lo0, lo1))
    hpa = da.values / 100.0 if float(da.max()) > 2000 else da.values
    return da.latitude.values, da.longitude.values, hpa

# ---------------------------------------------------------------- Isobares
def isobars(lat, lon, p, step):
    levels = np.arange(np.floor(p.min() / step) * step, np.ceil(p.max() / step) * step + step, step)
    cs = plt.contour(lon, lat, p, levels=levels)
    feats = []
    for lev, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            if len(seg) < 3:
                continue
            feats.append({"type": "Feature",
                          "properties": {"hpa": int(lev), "major": int(lev) % 4 == 0},
                          "geometry": {"type": "LineString",
                                       "coordinates": np.round(seg, 3).tolist()}})
    plt.close("all")
    return feats

# ---------------------------------------------------------------- Centres D / A
def centers(lat, lon, p, res_deg, radius_deg=6.0, min_depth=2.0):
    """Extremums locaux sur un rayon donné, filtrés par l'écart à la moyenne alentour."""
    size = max(3, int(round(2 * radius_deg / res_deg)) | 1)
    mean = uniform_filter(p, size=size, mode="nearest")
    out = []
    for kind, filt, sign in (("L", minimum_filter, -1), ("H", maximum_filter, 1)):
        ext = filt(p, size=size, mode="nearest") == p
        ext[[0, -1], :] = ext[:, [0, -1]] = False          # pas de faux centres en bordure
        for i, j in zip(*np.where(ext)):
            depth = sign * (p[i, j] - mean[i, j])
            if depth >= min_depth:
                out.append({"type": "Feature",
                            "properties": {"type": kind,
                                           "label": "D" if kind == "L" else "A",
                                           "hpa": round(float(p[i, j]), 1),
                                           "intensity": round(float(depth), 1)},
                            "geometry": {"type": "Point",
                                         "coordinates": [round(float(lon[j]), 2), round(float(lat[i]), 2)]}})
    return out

# ---------------------------------------------------------------- Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=DOMAINS, default="pacific")
    ap.add_argument("--source", choices=["auto", "arpege", "gfs"], default="auto")
    ap.add_argument("--grib", help="fichier GRIB2 local (ignore le téléchargement)")
    ap.add_argument("--step", type=float, default=2.0, help="intervalle isobares (hPa)")
    ap.add_argument("--radius", type=float, default=6.0, help="rayon de détection D/A (degrés)")
    ap.add_argument("--min-depth", type=float, default=2.0, help="écart min. à l'environnement (hPa)")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    dom = DOMAINS[a.domain]
    tmp = tempfile.mkdtemp()
    if a.grib:
        got = (a.grib, None, "local")
    else:
        got = None
        if a.source in ("auto", "arpege"):
            got = get_arpege(tmp)
        if not got and a.source in ("auto", "gfs"):
            got = get_gfs(tmp, dom)
    if not got:
        sys.exit("Aucune donnée disponible.")
    path, run, model = got

    lat, lon, p = load_pmsl(path, dom)
    res = float(abs(lat[1] - lat[0]))
    meta = {"model": model, "domain": a.domain, "bbox_0_360": dom,
            "run": run.strftime("%Y-%m-%dT%H:00Z") if run else None,
            "generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
            "pmin": round(float(p.min()), 1), "pmax": round(float(p.max()), 1),
            "arpege_errors": ARPEGE_ERR[:2]}

    iso = isobars(lat, lon, p, a.step)
    ctr = centers(lat, lon, p, res, a.radius, a.min_depth)

    os.makedirs(a.out, exist_ok=True)
    for name, feats in (("isobars", iso), ("centers", ctr)):
        with open(os.path.join(a.out, f"{name}_{a.domain}.geojson"), "w") as f:
            json.dump({"type": "FeatureCollection", "metadata": meta, "features": feats}, f,
                      separators=(",", ":"))
    with open(os.path.join(a.out, f"meta_{a.domain}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{model} {meta['run']} | {len(iso)} isobares | "
          f"{sum(c['properties']['type']=='L' for c in ctr)} D / "
          f"{sum(c['properties']['type']=='H' for c in ctr)} A")

if __name__ == "__main__":
    main()
