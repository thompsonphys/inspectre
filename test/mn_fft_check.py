import csv
import gzip
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

import inspectre_c
import kerrgeodesics
from inspectre import Inspectre
from mn_fft_sweep import (COLUMNS, NS, QUANTITIES, bins, meta_path, mode_errors,
                          parse_expr, read_reference, reference_vectors)

REF = sys.argv[1]
SELECTED = set(sys.argv[2:])
I = Inspectre(spin=0.5, semilatus_rectum=10.0, eccentricity=0.3)
OUT = (107 / 7, math.pi / 2 + 2 / 25, "r=107/7,dth=2/25")
IN = (11.0, math.pi / 2 + 1 / 200, "r=11,dth=1/200")
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


def calc_m(m, lam, r, theta):
    PhiS, d, dd, s = I.eval_at_lambda_fast(m, lam, r, theta)
    return np.array([PhiS[0] + 1j * PhiS[1], d[2] + 1j * d[3], d[4] + 1j * d[5], s[0] + 1j * s[1]])


def lambda_c(label):
    return next(p["lambda_c"] for p in read_reference(REF)["points"] if p["label"] == label)


def max_errors(route, N, m=2):
    r, theta, label = OUT
    A, _ = I.fft_nmodes_fast([m], r, theta, N, route)
    vec = reference_vectors(read_reference(REF), label, m)
    return [mode_errors(bins(A[0, qi], N), vec[q], np.abs(vec[q]).max()).max()
            for qi, q in enumerate(QUANTITIES)]


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def c_parser():
    ref = read_reference(REF)
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "bad.txt")
        with open(bad, "w") as f:
            f.write("orbit 1/2 10 3/10\nfoo 1 2\n")
        rejects = raises(lambda: parse_expr("Sqrt[2]")) and raises(lambda: read_reference(bad))
    ok = (len(ref["points"]) == 23 and len(ref["mn"]) == 2829 and rejects
          and parse_expr("ArcSin[2/5]") == math.asin(0.4) and parse_expr("87/13") == 87 / 13)
    return ok, (len(ref["points"]), len(ref["mn"]), rejects)


def c_orbit():
    ref = read_reference(REF)
    diffs = [abs(I.energy - ref["constants"][0]) / ref["constants"][0],
             abs(I.angular_momentum - ref["constants"][1]) / ref["constants"][1],
             abs(I.omega_r - ref["freq"][0]) / ref["freq"][0],
             abs(I.omega_phi - ref["freq"][1]) / ref["freq"][1]]
    return ref["orbit"] == (0.5, 10.0, 0.3) and max(diffs) <= 1e-14, [f"{d:.1e}" for d in diffs]


def c_time_grid():
    _, info = I.fft_nmodes_fast([2], OUT[0], OUT[1], 4096, "analytic")
    bound = 4 * math.ulp(read_reference(REF)["freq"][2])
    return info["newton_residual"] <= bound and info["newton_maxit"] == 0, (info["newton_residual"], bound, info["newton_maxit"])


def c_refactor():
    half = inspectre_c.inspectre_radial_mino_period(I._orbpar) / 2
    h = hashlib.sha256()
    for m in (0, 2, 10):
        for lam in (0.0, 0.3, 1.1, half):
            PhiS, d, dd, s = I.eval_at_lambda_fast(m, lam, 11.0, math.pi / 2 + 0.02)
            h.update(struct.pack("<32d", *(PhiS + d + dd + s)))
    digest = h.hexdigest()
    return digest == "fcb01297b2435b9de3817b109b12b32247c5e0054eab9b47f07095e16cad08ef", digest[:16]


def c_slice():
    r, theta, _ = OUT
    C, nphi, _ = I.mmode_projected_fast([0, 2], r, theta, 0.7)
    H, nhalf, capped = I.mmode_projected_fast([0, 2], r, theta, 0.7, nphi_max=nphi // 2)
    A = np.array([calc_m(m, 0.7, r, theta) for m in (0, 2)])
    agree = np.abs(C - A) / np.abs(A)
    change = np.abs(C - H) / np.abs(C)
    lc = lambda_c(IN[2])
    Ci, nin, _ = I.mmode_projected_fast([0, 2], IN[0], IN[1], lc)
    Ai = np.array([calc_m(m, lc, IN[0], IN[1]) for m in (0, 2)])
    agree_in = np.abs(Ci - Ai)[:, :3] / np.abs(Ai)[:, :3]
    ok = (agree[:, :3].max() <= 1e-12 and change[:, :3].max() <= 1e-14
          and agree[:, 3].max() <= 1e-9 and change[:, 3].max() <= 1e-12
          and agree_in.max() <= 1e-11 and nhalf == nphi // 2 and capped == 1)
    return ok, dict(nphi=nphi, agree=f"{agree[:, :3].max():.1e}", change=f"{change[:, :3].max():.1e}",
                    agree_src=f"{agree[:, 3].max():.1e}", change_src=f"{change[:, 3].max():.1e}",
                    nphi_in=nin, agree_in=f"{agree_in.max():.1e}")


def c_slice_edges():
    lc = lambda_c(IN[2])
    _, nphi, capped = I.mmode_projected_fast([2], IN[0], IN[1], lc, nphi_max=64)
    psi = kerrgeodesics.korb_psifromla(0.7, I._orbpar)
    rp = kerrgeodesics.korb_rfrompsi(psi, I._orbpar)
    C, n0, c0 = I.mmode_projected_fast([0, 2], rp, math.pi / 2, 0.7)
    ok = nphi == 64 and capped == 1 and n0 == 0 and c0 == 0 and bool(np.all(np.isnan(C.real)))
    return ok, (nphi, capped, n0, int(np.isnan(C.real).sum()))


def c_route_a():
    errs = max_errors("analytic", 4096)
    return max(errs) <= 1e-11, [f"{e:.1e}" for e in errs]


def c_route_n():
    errs = max_errors("numerical", 4096)
    return max(errs) <= 1e-11, [f"{e:.1e}" for e in errs]


def c_spline():
    r, theta, label = OUT
    A, _ = I.fft_nmodes_fast([2], r, theta, 4096, "analytic")
    re, im = I.fft_source_nmodes_fast(2, r, theta, 0.0, 4096)
    S = np.array(re) + 1j * np.array(im)
    vec = reference_vectors(read_reference(REF), label, 2)["src"]
    d = np.abs(bins(A[0, 3], 4096) - bins(S, 4096)).max() / np.abs(vec).max()
    return d <= 1e-11, f"{d:.1e}"


def c_output():
    sweep = [sys.executable, os.path.join(ROOT, "test", "mn_fft_sweep.py")]
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "s.csv.gz")
        subprocess.run(sweep + [REF, "--out", out, "--N", "64", "--points", OUT[2], "--workers", "1"],
                       check=True, capture_output=True)
        with gzip.open(out, "rt", newline="") as f:
            rows = list(csv.reader(f))
        with open(meta_path(out)) as f:
            meta = json.load(f)
        other = os.path.join(tmp, "other.txt")
        with open(REF) as f, open(other, "w") as g:
            g.write(f.read().replace("orbit 1/2 10 3/10", "orbit 1/2 10 1/5", 1))
        unknown = subprocess.run(sweep + [REF, "--out", os.path.join(tmp, "u.csv.gz"), "--N", "64",
                                          "--points", "r=1,dth=0", "--workers", "1"], capture_output=True)
        orbit = subprocess.run(sweep + [other, "--out", os.path.join(tmp, "o.csv.gz"), "--N", "64",
                                        "--workers", "1"], capture_output=True)
        no_output = not os.path.exists(os.path.join(tmp, "u.csv.gz")) and not os.path.exists(os.path.join(tmp, "o.csv.gz"))
        failed = [subprocess.run(sweep + [REF, "--out", out, "--points", OUT[2]] + extra, capture_output=True).returncode
                  for extra in (["--N", "32", "--workers", "1"],
                                ["--N", "64", "--routes", "Analytic", "--workers", "1"],
                                ["--N", "64", "--workers", "0"])]
        with gzip.open(out, "rt", newline="") as f:
            kept = len(list(csv.reader(f))) - 1
        with open(meta_path(out)) as f:
            kept_calls = len(json.load(f)["calls"])
        leftovers = sorted(n for n in os.listdir(tmp) if n.endswith(".part"))
    ok = (tuple(rows[0]) == COLUMNS and len(rows) - 1 == 984 and len(meta["calls"]) == 2
          and meta["end_line"] is False and meta["skipped_lines"]["check"] == 7
          and unknown.returncode != 0 and orbit.returncode != 0 and no_output
          and all(rc != 0 for rc in failed) and kept == 984 and kept_calls == 2 and not leftovers)
    return ok, (len(rows) - 1, len(meta["calls"]), meta["skipped_lines"], unknown.returncode, orbit.returncode, no_output,
                failed, kept, kept_calls, leftovers)


def c_inputs():
    r, theta, _ = OUT
    bad = [lambda: I.fft_nmodes_fast([2], r, theta, 32, "analytic"),
           lambda: I.fft_nmodes_fast([], r, theta, 64, "analytic"),
           lambda: I.fft_nmodes_fast([-1], r, theta, 64, "analytic"),
           lambda: I.fft_nmodes_fast([21], r, theta, 64, "analytic"),
           lambda: I.fft_nmodes_fast([2], r, theta, 64, "numerical", nphi_max=100),
           lambda: I.fft_nmodes_fast([40], r, theta, 64, "numerical", nphi_max=64),
           lambda: I.fft_nmodes_fast([2], r, theta, 64, "other"),
           lambda: I.mmode_projected_fast([2], r, theta, 0.7, nphi_max=32),
           lambda: I.mmode_projected_fast([-1], r, theta, 0.7)]
    raised = []
    for f in bad:
        try:
            f()
            raised.append(False)
        except ValueError:
            raised.append(True)
    return all(raised), raised


def c_fft_edges():
    rp0 = kerrgeodesics.korb_rfrompsi(kerrgeodesics.korb_psifromla(0.0, I._orbpar), I._orbpar)
    on = {}
    for route in ("analytic", "numerical"):
        A, info = I.fft_nmodes_fast([0, 2], rp0, math.pi / 2, 64, route)
        on[route] = (info["c1_zero"], bool(np.all(np.isnan(A))))
    _, info = I.fft_nmodes_fast([2], IN[0], IN[1], 64, "numerical", nphi_max=64)
    ok = all(v == (1.0, True) for v in on.values()) and info["capped"] >= 1 and info["nphi_max"] == 64
    return ok, (on, info["capped"], info["nphi_max"])


check("parser", c_parser)
check("orbit", c_orbit)
check("time-grid", c_time_grid)
check("refactor", c_refactor)
check("slice", c_slice)
check("slice-edges", c_slice_edges)
check("route-a", c_route_a)
check("route-n", c_route_n)
check("spline", c_spline)
check("output", c_output)
check("inputs", c_inputs)
check("fft-edges", c_fft_edges)
sys.exit(0 if results and all(results) else 1)
