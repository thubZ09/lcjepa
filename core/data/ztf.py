from __future__ import annotations
import csv
import io
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
import numpy as np

IRSA_LC_URL = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
_REQUIRED_COLS = ("oid", "mjd", "mag", "magerr", "catflags", "filtercode")

@dataclass
class ZTFCurve:
    oid: int
    ra: float
    dec: float
    band: str
    mjd: np.ndarray
    mag: np.ndarray
    magerr: np.ndarray
    label: str | None = None

def build_query_url(
    ra: float,
    dec: float,
    band: str,
    radius_arcsec: float,
    collection: str | None = None,
    nobs_min: int | None = None,
) -> str:
    radius_deg = radius_arcsec / 3600.0
    params = {
        "POS": f"CIRCLE {ra} {dec} {radius_deg:.6f}",
        "BANDNAME": band,
        "FORMAT": "CSV",
        "BAD_CATFLAGS_MASK": "32768", 
    }
    if collection:
        params["COLLECTION"] = collection
    if nobs_min:
        params["NOBS_MIN"] = str(nobs_min)
    return f"{IRSA_LC_URL}?{urllib.parse.urlencode(params)}"

def parse_lightcurve_csv(
    text: str,
    band: str,
    max_magerr: float = 0.2,
    label: str | None = None,
) -> list[ZTFCurve]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return []
    missing = [c for c in _REQUIRED_COLS if c not in reader.fieldnames]
    if missing:
        raise ValueError(f"IRSA CSV missing expected columns: {missing}")
    by_oid: dict[int, dict[str, list[float]]] = {}
    coords: dict[int, tuple[float, float]] = {}
    for row in reader:
        if row["filtercode"] not in (band, f"z{band}"): 
            continue
        try:
            mag = float(row["mag"])
            magerr = float(row["magerr"])
            mjd = float(row["mjd"])
            oid = int(row["oid"])
        except (ValueError, KeyError):
            continue
        if not (np.isfinite(mag) and np.isfinite(magerr)):
            continue
        if magerr > max_magerr:
            continue
        d = by_oid.setdefault(oid, {"mjd": [], "mag": [], "magerr": []})
        d["mjd"].append(mjd)
        d["mag"].append(mag)
        d["magerr"].append(magerr)
        if oid not in coords and "ra" in row and "dec" in row:
            try:
                coords[oid] = (float(row["ra"]), float(row["dec"]))
            except ValueError:
                coords[oid] = (float("nan"), float("nan"))
    curves: list[ZTFCurve] = []
    for oid, d in by_oid.items():
        order = np.argsort(d["mjd"])
        ra, dec = coords.get(oid, (float("nan"), float("nan")))
        curves.append(
            ZTFCurve(
                oid=oid,
                ra=ra,
                dec=dec,
                band=band,
                mjd=np.asarray(d["mjd"], dtype=np.float64)[order],
                mag=np.asarray(d["mag"], dtype=np.float32)[order],
                magerr=np.asarray(d["magerr"], dtype=np.float32)[order],
                label=label,
            )
        )
    return curves

def curves_from_dataframe(df, band: str, max_magerr: float, label: str | None) -> list[ZTFCurve]:
    import numpy as _np
    out: list[ZTFCurve] = []
    sub = df[df["filtercode"].isin([band, f"z{band}"])] if "filtercode" in df else df
    sub = sub[sub["magerr"] <= max_magerr]
    for oid, g in sub.groupby("oid"):
        order = _np.argsort(g["mjd"].to_numpy())
        ra = float(g["ra"].iloc[0]) if "ra" in g else float("nan")
        dec = float(g["dec"].iloc[0]) if "dec" in g else float("nan")
        out.append(
            ZTFCurve(
                oid=int(oid), ra=ra, dec=dec, band=band,
                mjd=g["mjd"].to_numpy(_np.float64)[order],
                mag=g["mag"].to_numpy(_np.float32)[order],
                magerr=g["magerr"].to_numpy(_np.float32)[order],
                label=label,
            )
        )
    return out

def fetch_lightcurves(
    ra: float,
    dec: float,
    band: str,
    radius_arcsec: float = 1.5,
    max_magerr: float = 0.2,
    label: str | None = None,
    timeout: float = 60.0,
    retries: int = 3,
    backoff: float = 2.0,
    prefer_ztfquery: bool = True,
) -> list[ZTFCurve]:
    if prefer_ztfquery:
        try:
            from ztfquery import lightcurve as _lc  # type: ignore

            q = _lc.LCQuery.from_position(ra, dec, radius_arcsec)
            return curves_from_dataframe(q.data, band, max_magerr, label)
        except ImportError:
            pass  
        except Exception:  
            pass
    url = build_query_url(ra, dec, band, radius_arcsec)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp: 
                text = resp.read().decode("utf-8", errors="replace")
            return parse_lightcurve_csv(text, band, max_magerr=max_magerr, label=label)
        except Exception as e:  
            last_err = e
            time.sleep(backoff**attempt)
    raise RuntimeError(f"IRSA query failed after {retries} attempts: {last_err}")

def best_curve(curves: list[ZTFCurve]) -> ZTFCurve | None:
    return max(curves, key=lambda c: c.mjd.size) if curves else None