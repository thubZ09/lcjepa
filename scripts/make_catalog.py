from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np

RA_ALIASES = ("RAJ2000", "RAdeg", "RA", "ra", "_RAJ2000")
DEC_ALIASES = ("DEJ2000", "DEdeg", "DEC", "DE", "dec", "_DEJ2000")

def _find_col(colnames: list[str], aliases: tuple[str, ...], kind: str) -> str:
    for a in aliases:
        if a in colnames:
            return a
    lower = {c.lower(): c for c in colnames}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    raise ValueError(f"Could not find a {kind} column among {colnames}")

def dedupe_by_proximity(coords: np.ndarray, radius_arcsec: float = 2.0) -> np.ndarray:
    r_deg = radius_arcsec / 3600.0
    kept: list[np.ndarray] = []
    for c in coords:
        cos_dec = np.cos(np.radians(c[1]))
        dup = False
        for k in kept:
            d_ra = (c[0] - k[0]) * cos_dec
            d_dec = c[1] - k[1]
            if d_ra * d_ra + d_dec * d_dec < r_deg * r_deg:
                dup = True
                break
        if not dup:
            kept.append(c)
    return np.asarray(kept)

def parse_vizier_tsv(path: str) -> np.ndarray:
    ra_out: list[float] = []
    dec_out: list[float] = []
    ra_i = dec_i = -1
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            cols = [p.strip() for p in parts]
            if "_RAJ2000" in cols and "_DEJ2000" in cols:
                ra_i, dec_i = cols.index("_RAJ2000"), cols.index("_DEJ2000")
                continue
            if ra_i < 0 or len(parts) <= max(ra_i, dec_i):
                continue
            try:  
                ra_out.append(float(parts[ra_i]))
                dec_out.append(float(parts[dec_i]))
            except ValueError:
                continue
    if not ra_out:
        raise RuntimeError(
            f"No _RAJ2000/_DEJ2000 data rows found in {path}. "
            "Re-download with -out.add=_RAJ2000,_DEJ2000 in the VizieR URL."
        )
    coords = np.column_stack([ra_out, dec_out])
    coords = dedupe_by_proximity(coords, radius_arcsec=2.0)
    return coords

def read_clagn(path: str) -> np.ndarray:
    head = open(path, encoding="utf-8", errors="replace").read(2048)
    if head.startswith("#") and "VizieR" in head:
        coords = parse_vizier_tsv(path)
        print(f"[catalog] CLAGN positives: {len(coords)} unique objects "
              f"(VizieR TSV, proximity-deduped at 2\") from {path}")
        return coords
    from astropy.table import Table
    t = None
    errors = []
    for fmt in ("ascii", "ascii.tab", "ascii.csv", "votable", "fits"):
        try:
            t = Table.read(path, format=fmt)
            break
        except Exception as e: 
            errors.append(f"{fmt}: {e}")
    if t is None:
        raise RuntimeError(f"Could not parse {path} with any known format:\n" + "\n".join(errors))
    ra_col = _find_col(t.colnames, RA_ALIASES, "RA")
    dec_col = _find_col(t.colnames, DEC_ALIASES, "Dec")
    ra = np.asarray(t[ra_col], dtype=float)
    dec = np.asarray(t[dec_col], dtype=float)
    good = np.isfinite(ra) & np.isfinite(dec)
    coords = np.column_stack([ra[good], dec[good]])
    coords = dedupe_by_proximity(coords, radius_arcsec=2.0)
    print(f"[catalog] CLAGN positives: {len(coords)} unique objects "
          f"(cols {ra_col}/{dec_col}) from {path}")
    return coords

def read_dr16q(path: str, dec_min: float, rmag_max: float) -> np.ndarray:
    from astropy.io import fits
    with fits.open(path, memmap=True) as hdul:
        d = hdul[1].data
        cols = d.columns.names
        ra = np.asarray(d["RA"], dtype=float)
        dec = np.asarray(d["DEC"], dtype=float)
        mask = np.isfinite(ra) & np.isfinite(dec) & (dec > dec_min)
        if "PSFMAG" in cols:
            psf = np.asarray(d["PSFMAG"], dtype=float)
            rmag = psf[:, 2] if psf.ndim == 2 and psf.shape[1] >= 3 else None
            if rmag is not None:
                mask &= np.isfinite(rmag) & (rmag > 0) & (rmag < rmag_max)
            else:
                print("[catalog] WARNING: PSFMAG has unexpected shape; skipping r-mag cut")
        else:
            print("[catalog] WARNING: no PSFMAG column; skipping r-mag cut")
    out = np.column_stack([ra[mask], dec[mask]])
    print(f"[catalog] DR16Q controls passing cuts (dec>{dec_min}, r<{rmag_max}): {len(out)}")
    return out

def remove_near(controls: np.ndarray, positives: np.ndarray, radius_arcsec: float) -> np.ndarray:
    r_deg = radius_arcsec / 3600.0
    keep = np.ones(len(controls), dtype=bool)
    cos_dec = np.cos(np.radians(controls[:, 1]))
    for pra, pdec in positives:
        d_ra = (controls[:, 0] - pra) * cos_dec
        d_dec = controls[:, 1] - pdec
        near = (d_ra**2 + d_dec**2) < r_deg**2
        keep &= ~near
    removed = int((~keep).sum())
    if removed:
        print(f"[catalog] removed {removed} controls within {radius_arcsec}\" of a CLAGN")
    return controls[keep]

def main() -> None:
    ap = argparse.ArgumentParser(description="Build ra,dec,label catalog for download_ztf.py")
    ap.add_argument("--clagn", required=True, help="Guo+2025 table (VizieR TSV/CSV/FITS).")
    ap.add_argument("--dr16q", required=True, help="DR16Q_v4.fits path.")
    ap.add_argument("--out", default="data/my_catalog.csv")
    ap.add_argument("--n-controls", type=int, default=4000)
    ap.add_argument("--n-unlabeled", type=int, default=0,
                    help="Extra DR16Q quasars with NO label (blank) — used only to "
                         "scale the self-supervised pretraining corpus; the eval "
                         "ignores unlabeled curves. Drawn disjoint from controls.")
    ap.add_argument("--dec-min", type=float, default=-30.0, help="ZTF footprint cut.")
    ap.add_argument("--rmag-max", type=float, default=20.0, help="Control brightness cut.")
    ap.add_argument("--match-radius", type=float, default=3.0,
                    help="Arcsec radius to purge CLAGN from controls.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    positives = read_clagn(args.clagn)
    controls = read_dr16q(args.dr16q, args.dec_min, args.rmag_max)
    controls = remove_near(controls, positives, args.match_radius)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(controls))
    labeled_controls = controls[perm[: args.n_controls]]
    unlabeled = controls[perm[args.n_controls : args.n_controls + args.n_unlabeled]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ra", "dec", "label"])
        for ra, dec in positives:
            w.writerow([f"{ra:.6f}", f"{dec:.6f}", 1])
        for ra, dec in labeled_controls:
            w.writerow([f"{ra:.6f}", f"{dec:.6f}", 0])
        for ra, dec in unlabeled:
            w.writerow([f"{ra:.6f}", f"{dec:.6f}", ""]) 
    print(f"[catalog] wrote {len(positives)} positives + {len(labeled_controls)} controls "
          f"+ {len(unlabeled)} unlabeled -> {out}")

if __name__ == "__main__":
    main()