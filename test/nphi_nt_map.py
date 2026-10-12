import argparse
import csv
import gzip
import json
import math
import multiprocessing
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from scipy.optimize import brentq

from mn_fft_sweep import parse_expr, read_reference

SPIN, SEMILATUS, ECC = 0.5, 10.0, 0.3
MS = tuple(range(21))
NS = tuple(range(-20, 21))
HALF = 5e-9
REF_TOL = 1e-9
XCHECK_TOL = 1e-9
N_REF, N_REF_MAX, NPHI_REF = 2 ** 14, 2 ** 17, 2 ** 20
GROUP = {"interior": (3,), "face": (0, 1, 2)}
QNAMES = ("PhiS", "dPhiS_dr", "dPhiS_dth", "src")
R_IN_S = "34099538200381/10000000000000"
R_OUT_S = "9274383314342119/500000000000000"
R_INTERIOR = ("7/2", "5", "13/2", "15/2", "39/5", "17/2", "9", "10", "11", "12", "13", "14",
              "71/5", "15", "33/2", "92/5")
DTH = ("1/2", "1/5", "1/10", "1/20", "1/50", "1/100", "1/200", "1/500", "1/1000", "1/2000",
       "1/5000", "1/10000")
THETA_FACE = "ArcSin[1/2]"
XCHECK = (("11", "1/100"), ("11", "1/1000"), ("11", "1/10000"), ("9", "1/10"), ("13", "1/1000"),
          ("13/2", "1/100"), ("33/2", "1/5"), (R_IN_S, "1/10"), (R_OUT_S, "1/100"), ("11", THETA_FACE))
SUMMARY = ("label", "r", "dth", "kind", "verified", "N_ref", "nphi_ref", "N_t_req", "E_half_N_t_req",
           "Nphi_pair", "Nphi_safe", "Nphi_all", "E_joint_pair", "E_joint_safe", "E_math", "imag_math", "seconds")
COLUMNS = ("label", "r", "dth", "kind", "N_ref", "N_t", "nphi_max", "E", "E_phi", "worst_m", "worst_n",
           "worst_q")


def tortoise(r, a=SPIN):
    rp, rm = 1 + math.sqrt(1 - a * a), 1 - math.sqrt(1 - a * a)
    return r + 2 * rp / (rp - rm) * math.log((r - rp) / 2) - 2 * rm / (rp - rm) * math.log((r - rm) / 2)


def r_from_tortoise(rs, a=SPIN):
    rp = 1 + math.sqrt(1 - a * a)
    return brentq(lambda r: tortoise(r, a) - rs, rp + 1e-12, 1e3, xtol=1e-15, rtol=4e-16)


RSTAR_P = tortoise(SEMILATUS)


def point(rs, ds, kind):
    return {"label": f"r={rs},dth={ds}", "r": parse_expr(rs), "dth": parse_expr(ds), "kind": kind}


def points():
    pts = [point(rs, ds, "interior") for rs in R_INTERIOR for ds in DTH]
    pts += [point(rs, ds, "face") for rs in (R_IN_S, R_OUT_S) for ds in DTH]
    pts += [point(rs, THETA_FACE, "face") for rs in R_INTERIOR]
    return pts


def bins(A, N, Nc):
    B = A.reshape(A.shape[0], A.shape[1], N // Nc, Nc).sum(axis=2)
    return B[:, :, [n % Nc for n in NS]]


def point_error(B, R, qs):
    best = (-1.0, -1, 0, -1)
    for q in qs:
        d = np.abs(B[:, q] - R[:, q])
        e = d.max() / np.abs(R[:, q]).max()
        if not e <= best[0]:
            i, k = np.unravel_index(int(np.nanargmax(d)) if np.isfinite(d).any() else 0, d.shape)
            best = (e, MS[i], NS[k], q)
            if not np.isfinite(e):
                break
    return best


def required_level(levels, errs, tol):
    for i in range(len(levels)):
        if all(e <= tol for e in errs[i:]):
            return levels[i]
    return None


def nphi_all(levels_t, N, nt_req, levels_phi, e_phi, tol):
    req = [required_level(levels_phi, [e_phi[(Nt, L)] for L in levels_phi], tol)
           for Nt in levels_t + [N] if Nt >= (nt_req or N)]
    return None if None in req else max(req)


def crosscheck_error(R, xref, label, qs):
    vals = np.zeros((len(MS), 4, len(NS)))
    for i, m in enumerate(MS):
        for q in qs:
            vals[i, q] = [np.nan if v is None else v for v in (xref["mn"][(label, m, n)][QNAMES[q]] for n in NS)]
    e = point_error(R.real, vals, qs)[0]
    imag = max(np.abs(R[:, q].imag).max() / np.abs(vals[:, q]).max() for q in qs)
    return e, imag


_insp = None


def _init():
    global _insp
    from inspectre import Inspectre
    _insp = Inspectre(spin=SPIN, semilatus_rectum=SEMILATUS, eccentricity=ECC)


def reference_run(insp, r, theta, n_ref, n_ref_max, nphi_ref, qs):
    N = n_ref
    while True:
        A, info = insp.fft_nmodes_fast(MS, r, theta, N, "numerical", nphi_ref)
        ref = bins(A, N, N)
        e_half = point_error(bins(A, N, N // 2), ref, qs)[0]
        ok = info["capped"] == 0 and info["c1_zero"] == 0 and e_half <= REF_TOL
        if ok or N >= n_ref_max:
            return A, info, N, ok
        N *= 2


def run_point(job):
    pt, cfg = job
    r, theta, qs = pt["r"], math.pi / 2 + pt["dth"], GROUP[pt["kind"]]
    t0 = time.time()
    A, info, N, verified = reference_run(_insp, r, theta, cfg["n_ref"], cfg["n_ref_max"], cfg["nphi_ref"], qs)
    nphi_used = int(info["nphi_max"])
    levels_t = [2 ** k for k in range(6, int(math.log2(N)))]
    ref = bins(A, N, N)
    ref_at = {Nt: bins(A, N, Nt) for Nt in levels_t + [N]}
    base = (pt["label"], r, pt["dth"], pt["kind"], N)
    rows, e_t = [], []
    for Nt in levels_t:
        e, wm, wn, wq = point_error(ref_at[Nt], ref, qs)
        e_t.append(e)
        rows.append(base + (Nt, 0, e, 0.0, wm, wn, QNAMES[wq]))
    nt_req = required_level(levels_t, e_t, HALF)
    e_below = e_t[levels_t.index(nt_req) - 1] if nt_req and nt_req > levels_t[0] else None
    levels_phi = [2 ** k for k in range(6, int(math.log2(nphi_used)) + 1)]
    e_phi, e_joint = {}, {}
    for L in levels_phi:
        AL = _insp.fft_nmodes_fast(MS, r, theta, N, "numerical", L)[0]
        for Nt in levels_t + [N]:
            BL = bins(AL, N, Nt)
            e, wm, wn, wq = point_error(BL, ref, qs)
            ep = point_error(BL, ref_at[Nt], qs)[0]
            e_phi[(Nt, L)], e_joint[(Nt, L)] = ep, e
            rows.append(base + (Nt, L, e, ep, wm, wn, QNAMES[wq]))
    pair = required_level(levels_phi, [e_phi[(nt_req, L)] for L in levels_phi], HALF) if nt_req else None
    safe = required_level(levels_phi, [e_phi[(N, L)] for L in levels_phi], HALF)
    e_math = imag = None
    if cfg["xref"] is not None and (pt["label"], 0, 0) in cfg["xref"]["mn"]:
        e_math, imag = crosscheck_error(ref, cfg["xref"], pt["label"], qs)
    summary = dict(label=pt["label"], r=r, dth=pt["dth"], kind=pt["kind"], verified=verified, N_ref=N,
                   nphi_ref=nphi_used, N_t_req=nt_req, E_half_N_t_req=e_below, Nphi_pair=pair, Nphi_safe=safe,
                   Nphi_all=nphi_all(levels_t, N, nt_req, levels_phi, e_phi, HALF),
                   E_joint_pair=e_joint.get((nt_req, pair)), E_joint_safe=e_joint.get((N, safe)),
                   E_math=e_math, imag_math=imag, seconds=time.time() - t0)
    return rows, summary


def cost_order(pt):
    inside = 100 / 13 < pt["r"] < 100 / 7
    return (not inside, pt["dth"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--crosscheck")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--points", nargs="+")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-ref", type=int, default=N_REF)
    ap.add_argument("--n-ref-max", type=int, default=N_REF_MAX)
    ap.add_argument("--nphi-ref", type=int, default=NPHI_REF)
    args = ap.parse_args(argv)

    pts = points()
    if args.points:
        pts = [p for p in pts if p["label"] in args.points]
        missing = set(args.points) - {p["label"] for p in pts}
        if missing:
            raise SystemExit(f"unknown points: {sorted(missing)}")
    xref = read_reference(args.crosscheck) if args.crosscheck else None
    cfg = dict(n_ref=args.n_ref, n_ref_max=args.n_ref_max, nphi_ref=args.nphi_ref, xref=xref)
    jobs = [(p, cfg) for p in sorted(pts, key=cost_order)]

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "nphi_nt_map.csv.gz")
    summ = os.path.join(args.out_dir, "nphi_nt_map_summary.csv")
    meta_out = os.path.join(args.out_dir, "nphi_nt_map.meta.json")
    parts = [p + ".part" for p in (out, summ, meta_out)]
    summaries = []
    t0 = time.time()
    try:
        with gzip.open(parts[0], "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            with multiprocessing.Pool(args.workers, initializer=_init) as pool:
                for k, (rows, s) in enumerate(pool.imap_unordered(run_point, jobs), 1):
                    writer.writerows(rows)
                    summaries.append(s)
                    print(f"{k}/{len(jobs)} {s['label']} N_t_req={s['N_t_req']} Nphi_pair={s['Nphi_pair']} "
                          f"Nphi_safe={s['Nphi_safe']} verified={s['verified']} {s['seconds']:.0f} s", flush=True)
        with open(parts[1], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY)
            writer.writeheader()
            writer.writerows(sorted(summaries, key=lambda s: s["label"]))
        meta = {"orbit": [SPIN, SEMILATUS, ECC], "rstar_p": RSTAR_P,
                "worldtube": {"r_in": parse_expr(R_IN_S), "r_out": parse_expr(R_OUT_S), "cos_theta": [-0.5, 0.5]},
                "target": 2 * HALF, "half": HALF, "ref_tol": REF_TOL, "xcheck_tol": XCHECK_TOL,
                "n_ref": args.n_ref, "n_ref_max": args.n_ref_max, "nphi_ref": args.nphi_ref,
                "crosscheck": os.path.abspath(args.crosscheck) if args.crosscheck else None,
                "points": len(pts), "wall_seconds": time.time() - t0}
        with open(parts[2], "w") as f:
            json.dump(meta, f, indent=1)
        for p, final in zip(parts, (out, summ, meta_out)):
            os.replace(p, final)
    finally:
        for p in parts:
            if os.path.exists(p):
                os.remove(p)
    print(f"wrote {out}, {summ} and {meta_out}")


if __name__ == "__main__":
    main()
