import csv
import gzip
import json
import math
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from inspectre import Inspectre
from mn_fft_sweep import parse_expr, read_reference
import nphi_nt_map as nm

SELECTED = set(sys.argv[1:])
I = Inspectre(spin=0.5, semilatus_rectum=10.0, eccentricity=0.3)
results = []


def check(name, fn):
    if SELECTED and name not in SELECTED:
        return
    try:
        ok, val = fn()
    except Exception as exc:
        ok, val = False, f"{type(exc).__name__}: {exc}"
    results.append(bool(ok))
    print("check", name, "PASS" if ok else "FAIL", val, flush=True)


def c_worldtube():
    r_in, r_out = parse_expr(nm.R_IN_S), parse_expr(nm.R_OUT_S)
    d = [abs(nm.RSTAR_P - 12.775940), abs(nm.tortoise(r_in) - (nm.RSTAR_P - 10)), abs(nm.tortoise(r_out) - (nm.RSTAR_P + 10))]
    return d[0] <= 1e-6 and max(d[1:]) <= 1e-10, [f"{x:.1e}" for x in d]


def c_points():
    pts = nm.points()
    r_in, r_out = parse_expr(nm.R_IN_S), parse_expr(nm.R_OUT_S)
    kinds = [p["kind"] for p in pts]
    ok = (kinds.count("interior") == 192 and kinds.count("face") == 40
          and len({p["label"] for p in pts}) == len(pts)
          and all(p["dth"] > 0 and r_in - 1e-12 <= p["r"] <= r_out + 1e-12 and p["dth"] <= math.pi / 6 + 1e-12
                  for p in pts))
    return ok, (kinds.count("interior"), kinds.count("face"), len({p["label"] for p in pts}))


def c_measure():
    rng = np.random.default_rng(1)
    R = rng.normal(size=(len(nm.MS), 4, len(nm.NS))) + 1j * rng.normal(size=(len(nm.MS), 4, len(nm.NS)))
    B = R.copy()
    B[5, 1, 7] += 3e-9
    B[2, 3, 30] += 1e-9j
    e1, m1, n1, q1 = nm.point_error(B, R, (0, 1, 2))
    e2, m2, n2, q2 = nm.point_error(B, R, (3,))
    e3 = nm.point_error(B, R, (0, 1, 2, 3))[0]
    C = R.copy()
    C[0, 3, 0] = np.nan
    e4 = nm.point_error(C, R, (3,))[0]
    w1, w2 = 3e-9 / np.abs(R[:, 1]).max(), 1e-9 / np.abs(R[:, 3]).max()
    ok = (abs(e1 - w1) < 1e-6 * w1 and (m1, n1, q1) == (5, nm.NS[7], 1)
          and abs(e2 - w2) < 1e-6 * w2 and (m2, n2, q2) == (2, nm.NS[30], 3)
          and e3 == max(e1, e2) and not np.isfinite(e4))
    return ok, (e1, e2, e3, e4)


def c_ladder():
    lv = [64, 128, 256, 512, 1024]
    a = nm.required_level(lv, [1e-3, 4e-9, 6e-9, 3e-9, 1e-9], 5e-9)
    b = nm.required_level(lv, [1e-3, 1e-4, 1e-5, 1e-6, 6e-9], 5e-9)
    c = nm.required_level(lv, [1e-9] * 5, 5e-9)
    d = nm.required_level(lv, [1e-3, float("nan"), 1e-9, 1e-9, 1e-9], 5e-9)
    phi = [64, 128, 256, 512]
    table = {64: [1, 1, 1, 0], 128: [1e-3, 1e-9, 1e-9, 0], 256: [1e-3, 1e-3, 1e-9, 0], 512: [1e-3, 1e-9, 1e-9, 0]}
    e_phi = {(nt, L): e for nt, errs in table.items() for L, e in zip(phi, errs)}
    f = nm.nphi_all([64, 128, 256], 512, 128, phi, e_phi, 5e-9)
    g = nm.nphi_all([64, 128, 256], 512, None, phi, e_phi, 5e-9)
    return (a, b, c, d, f, g) == (512, None, 64, 256, 256, 128), (a, b, c, d, f, g)


def c_alias():
    th = math.pi / 2 + 1e-2
    A, _ = I.fft_nmodes_fast(nm.MS, 11.0, th, 1024, "numerical")
    D, _ = I.fft_nmodes_fast(nm.MS, 11.0, th, 512, "numerical")
    e = nm.point_error(nm.bins(A, 1024, 512), nm.bins(D, 512, 512), (0, 1, 2, 3))[0]
    return e <= 1e-12, f"{e:.1e}"


def c_reference():
    th = math.pi / 2 + 1e-4
    A, info, N, ok = nm.reference_run(I, 11.0, th, 2 ** 14, 2 ** 14, 2 ** 20, (3,))
    top = int(info["nphi_max"])
    T, _ = I.fft_nmodes_fast(nm.MS, 11.0, th, N, "numerical", top)
    e = nm.point_error(nm.bins(T, N, N), nm.bins(A, N, N), (3,))[0]
    good = ok and info["capped"] == 0 and info["c1_zero"] == 0 and e == 0.0
    return good, (ok, info["capped"], info["c1_zero"], top, e)


def c_regression():
    th = math.pi / 2 + 1e-3
    A, info, N, ok = nm.reference_run(I, 11.0, th, 2 ** 14, 2 ** 14, 2 ** 20, (3,))
    e = nm.point_error(nm.bins(A, N, 512), nm.bins(A, N, N), (3,))[0]
    return ok and 4.9e-9 / 2 <= e <= 4.9e-9 * 2, (ok, f"{e:.2e}")


def c_end_to_end():
    script = [sys.executable, os.path.join(ROOT, "test", "nphi_nt_map.py")]
    files = ["nphi_nt_map.csv.gz", "nphi_nt_map.meta.json", "nphi_nt_map_summary.csv"]
    with tempfile.TemporaryDirectory() as tmp:
        good = subprocess.run(script + ["--out-dir", tmp, "--points", "r=11,dth=1/100", "r=11,dth=1/10000",
                                        "--n-ref", "1024", "--n-ref-max", "2048", "--nphi-ref", "1048576",
                                        "--workers", "2"], capture_output=True, text=True)
        names = sorted(os.listdir(tmp))
        rows = summary = meta = None
        if good.returncode == 0:
            with gzip.open(os.path.join(tmp, files[0]), "rt") as f:
                rows = list(csv.DictReader(f))
            with open(os.path.join(tmp, files[2])) as f:
                summary = {s["label"]: s for s in csv.DictReader(f)}
            with open(os.path.join(tmp, files[1])) as f:
                meta = json.load(f)
        before = {n: os.path.getmtime(os.path.join(tmp, n)) for n in names}
        bad = subprocess.run(script + ["--out-dir", tmp, "--points", "r=11,dth=1/100", "--n-ref", "1024",
                                       "--nphi-ref", "100", "--workers", "1"], capture_output=True, text=True)
        after = {n: os.path.getmtime(os.path.join(tmp, n)) for n in sorted(os.listdir(tmp))}
    with tempfile.TemporaryDirectory() as tmp:
        low = subprocess.run(script + ["--out-dir", tmp, "--points", "r=11,dth=1/10000", "--n-ref", "128",
                                       "--n-ref-max", "128", "--workers", "1"], capture_output=True, text=True)
        s3 = {}
        if low.returncode == 0:
            with open(os.path.join(tmp, files[2])) as f:
                s3 = next(csv.DictReader(f))
    s1 = summary["r=11,dth=1/100"] if summary else {}
    s2 = summary["r=11,dth=1/10000"] if summary else {}
    levels = sorted({int(r["N_t"]) for r in rows if r["label"] == "r=11,dth=1/100"}) if rows else []
    ok = (good.returncode == 0 and names == files and tuple(s1.keys()) == nm.SUMMARY
          and s1["N_ref"] == "2048" and s1["verified"] == "True" and s1["N_t_req"] != "" and levels[-1] == 2048
          and s2["verified"] == "False" and meta["points"] == 2
          and int(s2["Nphi_all"]) >= max(int(s2["Nphi_pair"]), int(s2["Nphi_safe"]))
          and bad.returncode != 0 and after == before
          and low.returncode == 0 and s3["verified"] == "False" and s3["N_t_req"] == "" and s3["Nphi_pair"] == ""
          and s3["E_joint_pair"] == "" and s3["Nphi_safe"] != "" and s3["Nphi_all"] == s3["Nphi_safe"])
    return ok, (good.returncode, names, s1.get("N_ref"), s1.get("N_t_req"), s2.get("verified"), s2.get("N_t_req"),
                s2.get("Nphi_all"), bad.returncode, after == before, low.returncode, s3.get("N_t_req"),
                s3.get("Nphi_all"))


def c_crosscheck_reader():
    labels = ("r=11,dth=ArcSin[1/2]", f"r={nm.R_IN_S},dth=1/10", "r=9,dth=1/10")
    rng = np.random.default_rng(2)
    R = rng.normal(size=(len(nm.MS), 4, len(nm.NS))) + 1e-12j * rng.normal(size=(len(nm.MS), 4, len(nm.NS)))
    lines = ["orbit 1/2 10 3/10", "constants 9.5765130910241352621E-1 3.6055864458830043300E0",
             "freq 2.0048270768041681811E-2 2.7881254842826062220E-2 3.1340285553183043873E2 2.3519281906603015394E0"]
    for lab, (rs, ds) in zip(labels, (("11", "ArcSin[1/2]"), (nm.R_IN_S, "1/10"), ("9", "1/10"))):
        lines.append(f"point {lab} {rs} {ds} -")
        for i, m in enumerate(nm.MS):
            for k, n in enumerate(nm.NS):
                v = [f"{R[i, q, k].real:.17E}" for q in range(3)]
                if lab == labels[2]:
                    v[1] = "-"
                lines.append(f"mn {lab} {m} {n} {v[0]} {v[1]} {v[2]} - -")
            lines.append(f"digits {lab} {m} 16.0 16.0 16.0 - -")
        lines.append(f"digits {lab} all 15.9")
        lines.append(f"done {lab}")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.txt")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        xref = read_reference(path)
    B = R.copy()
    B[3, 0, 10] += 2e-10
    e, imag = nm.crosscheck_error(B, xref, labels[1], (0, 1, 2))
    e_missing = nm.crosscheck_error(B, xref, labels[2], (0, 1, 2))[0]
    want = 2e-10 / np.abs(R[:, 0].real).max()
    pts = [p["label"] for p in xref["points"]]
    ok = (pts == list(labels) and abs(xref["points"][0]["dth"] - math.pi / 6) < 1e-15
          and abs(e - want) / want < 1e-6 and 0 < imag < 1e-10 and not np.isfinite(e_missing))
    return ok, (pts, f"{e:.3e}", f"{want:.3e}", f"{imag:.1e}", e_missing)


check("worldtube", c_worldtube)
check("points", c_points)
check("measure", c_measure)
check("ladder", c_ladder)
check("alias", c_alias)
check("reference", c_reference)
check("regression", c_regression)
check("end-to-end", c_end_to_end)
check("crosscheck-reader", c_crosscheck_reader)
sys.exit(0 if results and all(results) else 1)
