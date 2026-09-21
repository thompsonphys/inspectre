"""Grade the original implementations against impl="refactored" and calc_m_extended.

Evaluators: original (upstream 07a31ce), original_agm (upstream plus the
complement-AGM evaluation of K and E), refactored (the context-based code in
use), extended (calc_m_extended, the long-double reference).

original calc_m and the refactored calc_m_core are token-identical apart from
the elliptic call, so original_agm and refactored agree bitwise by
construction. Part 2 asserts that agreement as a regression guard: it holds
only while the refactored expression body stays an unmodified copy of
upstream, and breaks if anything there drifts.

Part 1  non-mode PhiS and src, original vs refactored.
Part 2  m-mode PhiS and src, every double path vs calc_m_extended; refactored
        held to original_agm bitwise.
Part 3  original method surface and per-impl global-state guard.

Run inside the spectre conda env:
    python test/test_original_impls.py
"""
import gc
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "effectivesource"))
from inspectre import EffectiveSource, Inspectre

M, A, P, ECC = 1.0, 0.5, 10.0, 0.3
LAM_FRAC = 0.37

failures = 0


def check(name, cond, detail=""):
    """(name, bool, detail) -> None; prints a result row and counts failures."""
    global failures
    print(f"[{'ok ' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures += 1


def bitwise_equal(a, b):
    """(a, b) -> True if every component matches exactly, NaN positions included."""
    return bool(np.array_equal(np.asarray(a, float), np.asarray(b, float), equal_nan=True))


def rel(x, ref):
    """(x, ref) -> max componentwise relative error of x against ref."""
    x = np.atleast_1d(x).astype(float)
    ref = np.atleast_1d(ref).astype(float)
    return float(np.max(np.abs(x - ref) / np.maximum(np.abs(ref), 1e-300)))


insp = Inspectre(spin=A, semilatus_rectum=P, eccentricity=ECC)
lam = LAM_FRAC * insp.mino_period_r
r_p, th_p, ph_p = insp.r_from_lambda(lam), math.pi / 2, insp.phi_from_lambda(lam)
ur, E, L = insp.four_velocity_equatorial(lam), insp.energy, insp.angular_momentum

original = EffectiveSource(mass=M, spin=A, impl="original")
original_agm = EffectiveSource(mass=M, spin=A, impl="original_agm")
refactored = EffectiveSource(mass=M, spin=A, impl="refactored")
for es in (original, original_agm, refactored):
    es.set_particle(r_p, th_p, ph_p, E, L, ur)
ctx = refactored._ef

check("original and original_agm are live at once", True,
      "(separate shared objects, independent particle statics)")

print(f"\norbit a={A} p={P} e={ECC}; particle r_p={r_p:.15g} ur={ur:.15g}\n")

print("Part 1: original vs refactored, non-mode outputs (expect bitwise equal)")
worst_phis = worst_src = 0.0
for dr in (1e-1, 1e-2, 1e-3, 1e-4):
    for dth in (0.0, 1e-3, 1e-2):
        rf, thf = r_p + dr, th_p + dth
        worst_phis = max(worst_phis,
                         abs(original.phi_s(rf, thf, 0.0) - refactored.phi_s(rf, thf, 0.0)))
        worst_src = max(worst_src,
                        abs(original.source(rf, thf, 0.0) - refactored.source(rf, thf, 0.0)))
check("phi_s   bitwise identical", worst_phis == 0.0, f"(max abs diff {worst_phis:.3e})")
check("source  bitwise identical", worst_src == 0.0, f"(max abs diff {worst_src:.3e})")

print("\nPart 2: m-mode relative error vs calc_m_extended")
print(f"  {'dr':>7} {'dth':>7} {'m':>3} | {'PhiS orig':>10} {'PhiS +agm':>10} {'PhiS refac':>10}"
      f" | {'src orig':>10} {'src +agm':>10} {'src refac':>10}")
EVALS = ("original", "original_agm", "refactored")
SRC = {"original": original, "original_agm": original_agm, "refactored": refactored}
worst = {k: {"PhiS": 0.0, "src": 0.0} for k in EVALS}
near_ratio = np.inf
refactored_drift = []
for dr in (1e-1, 1e-2, 1e-3, 1e-4):
    for dth in (0.0, 1e-3):
        rf, thf = r_p + dr, th_p + dth
        dr_x = rf - r_p
        for m in (2, 10):
            gPhiS, _gd, gsrc = ctx.calc_m_extended(m, dr_x, dth)
            out = {k: SRC[k].calc_m(m, rf, thf) for k in EVALS}
            er = {}
            for k in EVALS:
                er[k] = (rel(out[k]["PhiS"], gPhiS), rel(out[k]["src"], gsrc))
                worst[k]["PhiS"] = max(worst[k]["PhiS"], er[k][0])
                worst[k]["src"] = max(worst[k]["src"], er[k][1])
            for block in ("PhiS", "dPhiS", "d2PhiS", "src"):
                if not bitwise_equal(out["original_agm"][block], out["refactored"][block]):
                    refactored_drift.append(f"{block} at dr={dr:.0e} dth={dth:.0e} m={m}")
            print(f"  {dr:7.0e} {dth:7.0e} {m:3d} | {er['original'][0]:10.3e} "
                  f"{er['original_agm'][0]:10.3e} {er['refactored'][0]:10.3e} "
                  f"| {er['original'][1]:10.3e} {er['original_agm'][1]:10.3e} "
                  f"{er['refactored'][1]:10.3e}")
            if dr <= 1e-4 and dth == 0.0 and er["original_agm"][0] > 0.0:
                near_ratio = min(near_ratio, er["original"][0] / er["original_agm"][0])

print()
check("original PhiS_m degrades near the particle", worst["original"]["PhiS"] > 1e-10,
      f"(worst {worst['original']['PhiS']:.3e})")
check("original_agm PhiS_m holds machine precision", worst["original_agm"]["PhiS"] < 1e-13,
      f"(worst {worst['original_agm']['PhiS']:.3e})")
check("refactored PhiS_m holds machine precision", worst["refactored"]["PhiS"] < 1e-13,
      f"(worst {worst['refactored']['PhiS']:.3e})")
check("refactored expression body is still an unmodified copy of upstream",
      not refactored_drift,
      "(refactored == original + AGM, bitwise on PhiS/dPhiS/d2PhiS/src)"
      if not refactored_drift else f"(drift in: {', '.join(refactored_drift[:3])})")
check("the elliptic call alone accounts for the puncture gain", near_ratio > 100.0,
      f"(original/original_agm error ratio {near_ratio:.3g} at dr=1e-4, dtheta=0)")
check("src_m is not helped by the elliptic fix",
      worst["original_agm"]["src"] > 1e-4,
      f"(original_agm worst {worst['original_agm']['src']:.3e}, "
      f"original {worst['original']['src']:.3e}, "
      f"refactored {worst['refactored']['src']:.3e})")

print("\nPart 3: original surface and global-state guard")
absent = [n for n in ("calc_offset", "calc_m_offset", "calc_PhiS_offset",
                      "calc_m_split", "calc_m_extended", "get_alpha", "_ctx")
          if not hasattr(original._ef, n)]
check("original exposes none of the post-refactor methods", len(absent) == 7,
      f"(absent: {len(absent)}/7)")

for impl in ("original", "original_agm"):
    try:
        EffectiveSource(mass=M, spin=A, impl=impl)
        check(f"second live {impl} instance is refused", False, "(no error raised)")
    except RuntimeError as exc:
        check(f"second live {impl} instance is refused", "global state" in str(exc))

try:
    EffectiveSource(mass=M, spin=A, mode="equatorial")
    check("mode= is refused", False, "(no error raised)")
except TypeError as exc:
    check("mode= is refused", "orbit=" in str(exc) and "impl=" in str(exc))

try:
    EffectiveSource(mass=M, spin=A, orbit="circular", impl="original")
    check("circular+original is refused with a pointer to the source", False, "(no error)")
except NotImplementedError as exc:
    check("circular+original is refused with a pointer to the source",
          "kerr-circular.c" in str(exc))

del original, original_agm, SRC
gc.collect()
try:
    insp_original = Inspectre(spin=A, semilatus_rectum=P, eccentricity=ECC, impl="original")
    check("releasing an original instance clears the guard", True)
except RuntimeError as exc:
    check("releasing an original instance clears the guard", False, f"({exc})")
    insp_original = None

if insp_original is not None:
    for impl in ("original", "original_agm"):
        insp_original.es.impl = impl
        try:
            insp_original._ctx
            check(f"Inspectre(impl={impl!r})._ctx is refused", False, "(no error raised)")
        except RuntimeError as exc:
            check(f"Inspectre(impl={impl!r})._ctx is refused",
                  "no effsource_equatorial_ctx" in str(exc))
    insp_original.es.impl = "original"

    insp_original.set_particle(r_p, th_p, ph_p, ur)
    check("Inspectre(impl='original') still evaluates phi_s",
          np.isfinite(insp_original.phi_s(r_p + 1e-2, th_p, 0.0)))

print(f"\n{'ALL PASS' if failures == 0 else str(failures) + ' FAILURE(S)'}")
sys.exit(1 if failures else 0)
