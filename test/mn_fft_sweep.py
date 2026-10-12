import argparse
import csv
import gzip
import json
import math
import multiprocessing
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

QUANTITIES = ("PhiS", "dPhiS_dr", "dPhiS_dth", "src")
MS = (0, 2, 10)
NS = tuple(range(-20, 21))
R_MIN, R_MAX = 100 / 13, 100 / 7
ORBIT = (0.5, 10.0, 0.3)
COLUMNS = ("label", "r", "dth", "region", "route", "m", "N", "quantity", "n",
           "ref", "re", "im", "err", "nphi_max", "capped", "c1_zero",
           "newton_residual", "newton_maxit", "seconds")
DEFAULT_N = tuple(2 ** k for k in range(6, 17))


def parse_expr(s):
    m = re.fullmatch(r"ArcSin\[(-?\d+)/(\d+)\]", s)
    if m:
        return math.asin(int(m.group(1)) / int(m.group(2)))
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m:
        return int(m.group(1)) / int(m.group(2))
    if re.fullmatch(r"-?\d+", s):
        return float(int(s))
    raise ValueError(f"unsupported expression {s!r}")


def parse_value(s):
    return None if s == "-" else float(s)


def read_reference(path):
    ref = {"points": [], "mn": {}, "end": False,
           "skipped": {"digits": 0, "adaptive": 0, "imag": 0, "check": 0}}
    points, mn, done = {}, {}, set()
    with open(path) as f:
        for line in f:
            w = line.split()
            if not w:
                continue
            if w[0] == "orbit":
                ref["orbit"] = tuple(parse_expr(x) for x in w[1:4])
            elif w[0] == "constants":
                ref["constants"] = tuple(float(x) for x in w[1:3])
            elif w[0] == "freq":
                ref["freq"] = tuple(float(x) for x in w[1:5])
            elif w[0] == "point":
                points[w[1]] = {"label": w[1], "r": parse_expr(w[2]),
                                "dth": parse_expr(w[3]), "lambda_c": parse_value(w[4])}
            elif w[0] == "mn":
                mn[(w[1], int(w[2]), int(w[3]))] = dict(
                    zip(QUANTITIES, (parse_value(x) for x in w[4:8])))
            elif w[0] == "done":
                done.add(w[1])
            elif w[0] == "end":
                ref["end"] = True
            elif w[0] in ref["skipped"]:
                ref["skipped"][w[0]] += 1
            else:
                raise ValueError(f"unknown record {w[0]!r}")
    ref["points"] = [p for label, p in points.items() if label in done]
    ref["mn"] = {k: v for k, v in mn.items() if k[0] in done}
    return ref


def region(r, dth):
    if r < R_MIN or r > R_MAX:
        return "outside"
    return "plane-inside" if dth == 0 else "inside"


def reference_vectors(ref, label, m):
    vec = {}
    for q in QUANTITIES:
        vals = [ref["mn"][(label, m, n)][q] for n in NS]
        if all(v is not None for v in vals):
            vec[q] = np.array(vals)
    return vec


def mode_errors(values, refvec, scale):
    return np.abs(values - refvec) / scale


def error_scales(vec):
    scales = {q: np.abs(v).max() for q, v in vec.items()}
    for q in scales:
        if scales[q] == 0:
            scales[q] = scales["dPhiS_dr"]
    return scales


def bins(A, N):
    return np.array([A[n % N] for n in NS])


def orbit_differences(ref, insp):
    mine = {"E": insp.energy, "L": insp.angular_momentum,
            "Omega_r": insp.omega_r, "Omega_phi": insp.omega_phi}
    theirs = {"E": ref["constants"][0], "L": ref["constants"][1],
              "Omega_r": ref["freq"][0], "Omega_phi": ref["freq"][1]}
    return {k: abs(mine[k] - theirs[k]) / abs(theirs[k]) for k in mine}


def meta_path(out):
    return re.sub(r"\.csv(\.gz)?$", "", out) + ".meta.json"


_insp = None
_ref = None


def _init(path):
    global _insp, _ref
    from inspectre import Inspectre
    _insp = Inspectre(spin=ORBIT[0], semilatus_rectum=ORBIT[1], eccentricity=ORBIT[2])
    _ref = read_reference(path)


def run_job(job):
    point, route, Ns = job
    label, r, dth = point["label"], point["r"], point["dth"]
    theta = math.pi / 2 + dth
    rows, calls = [], []
    vecs = {m: reference_vectors(_ref, label, m) for m in MS}
    for N in Ns:
        t0 = time.time()
        A, info = _insp.fft_nmodes_fast(list(MS), r, theta, N, route)
        seconds = time.time() - t0
        calls.append(dict(label=label, route=route, N=N, seconds=seconds, **info))
        for i, m in enumerate(MS):
            scales = error_scales(vecs[m])
            for qi, q in enumerate(QUANTITIES):
                if q not in vecs[m]:
                    continue
                values = bins(A[i, qi], N)
                err = mode_errors(values, vecs[m][q], scales[q])
                for k, n in enumerate(NS):
                    rows.append((label, r, dth, region(r, dth), route, m, N, q, n,
                                 vecs[m][q][k], values[k].real, values[k].imag, err[k],
                                 info["nphi_max"], info["capped"], info["c1_zero"],
                                 info["newton_residual"], info["newton_maxit"], seconds))
    return rows, calls


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "mn_fft_sweep.csv.gz"))
    ap.add_argument("--N", type=int, nargs="+", default=list(DEFAULT_N))
    ap.add_argument("--points", nargs="+")
    ap.add_argument("--routes", nargs="+", default=["analytic", "numerical"],
                    choices=["analytic", "numerical"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)
    if min(args.N) < 2 * max(NS) + 1:
        raise SystemExit(f"every N must be at least {2 * max(NS) + 1}, got {min(args.N)}")

    ref = read_reference(args.reference)
    if ref["orbit"] != ORBIT:
        raise SystemExit(f"reference orbit {ref['orbit']} differs from {ORBIT}")
    if not ref["end"]:
        print("warning: the reference has no end line, so its final checks are not validated")
    _init(args.reference)
    diffs = orbit_differences(ref, _insp)
    print("relative differences from the reference:",
          ", ".join(f"{k} {v:.2e}" for k, v in diffs.items()))

    points = ref["points"]
    if args.points:
        points = [p for p in points if p["label"] in args.points]
        missing = set(args.points) - {p["label"] for p in points}
        if missing:
            raise SystemExit(f"unknown points: {sorted(missing)}")
    order = {"plane-inside": 0, "inside": 1, "outside": 2}
    jobs = [(p, route, args.N) for route in args.routes for p in points]
    jobs.sort(key=lambda j: (order[region(j[0]["r"], j[0]["dth"])], j[1] == "analytic"))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    part_out, part_meta = args.out + ".part", meta_path(args.out) + ".part"
    calls = []
    t0 = time.time()
    try:
        with gzip.open(part_out, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            with multiprocessing.Pool(args.workers, initializer=_init,
                                      initargs=(args.reference,)) as pool:
                for k, (rows, job_calls) in enumerate(pool.imap_unordered(run_job, jobs), 1):
                    writer.writerows(rows)
                    calls.extend(job_calls)
                    print(f"{k}/{len(jobs)} {job_calls[0]['label']} {job_calls[0]['route']} "
                          f"{sum(c['seconds'] for c in job_calls):.1f} s", flush=True)
        meta = {"reference": os.path.abspath(args.reference), "orbit": list(ORBIT),
                "relative_differences": diffs, "end_line": ref["end"],
                "skipped_lines": ref["skipped"], "N": args.N, "routes": args.routes,
                "wall_seconds": time.time() - t0, "calls": calls}
        with open(part_meta, "w") as f:
            json.dump(meta, f, indent=1)
        os.replace(part_out, args.out)
        os.replace(part_meta, meta_path(args.out))
    finally:
        for p in (part_out, part_meta):
            if os.path.exists(p):
                os.remove(p)
    print(f"wrote {args.out} and {meta_path(args.out)}")


if __name__ == "__main__":
    main()
