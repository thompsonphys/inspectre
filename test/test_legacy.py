"""Grade the legacy modes against mode="equatorial" and calc_m_gold.

Evaluators: legacy (upstream 07a31ce), legacy_elliptic (upstream plus the
complement-AGM evaluation of K and E), equatorial (ctx), gold (reference).

legacy calc_m and ctx calc_m_core are token-identical apart from the elliptic
call, so legacy_elliptic and ctx agree bitwise by construction. Part 2 asserts
that agreement as a regression guard: it holds only while the ctx expression
body stays an unmodified copy of upstream, and breaks if anything there drifts.

Part 1  non-mode PhiS and src, legacy vs equatorial.
Part 2  m-mode PhiS and src, every double path vs calc_m_gold; ctx held to
        legacy_elliptic bitwise.
Part 3  legacy method surface and per-mode global-state guard.

Run inside the spectre conda env:
    python test/test_legacy.py
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

legacy = EffectiveSource(mass=M, spin=A, mode="legacy")
ellfix = EffectiveSource(mass=M, spin=A, mode="legacy_elliptic")
modern = EffectiveSource(mass=M, spin=A, mode="equatorial")
for es in (legacy, ellfix, modern):
    es.set_particle(r_p, th_p, ph_p, E, L, ur)
ctx = modern._ef

check("legacy and legacy_elliptic are live at once", True,
      "(separate shared objects, independent particle statics)")

print(f"\norbit a={A} p={P} e={ECC}; particle r_p={r_p:.15g} ur={ur:.15g}\n")

print("Part 1: legacy vs equatorial, non-mode outputs (expect bitwise equal)")
worst_phis = worst_src = 0.0
for dr in (1e-1, 1e-2, 1e-3, 1e-4):
    for dth in (0.0, 1e-3, 1e-2):
        rf, thf = r_p + dr, th_p + dth
        worst_phis = max(worst_phis, abs(legacy.phi_s(rf, thf, 0.0) - modern.phi_s(rf, thf, 0.0)))
        worst_src = max(worst_src, abs(legacy.source(rf, thf, 0.0) - modern.source(rf, thf, 0.0)))
check("phi_s   bitwise identical", worst_phis == 0.0, f"(max abs diff {worst_phis:.3e})")
check("source  bitwise identical", worst_src == 0.0, f"(max abs diff {worst_src:.3e})")

print("\nPart 2: m-mode relative error vs calc_m_gold")
print(f"  {'dr':>7} {'dth':>7} {'m':>3} | {'PhiS lgcy':>10} {'PhiS ell':>10} {'PhiS ctx':>10}"
      f" | {'src lgcy':>10} {'src ell':>10} {'src ctx':>10}")
EVALS = ("legacy", "ellfix", "ctx")
worst = {k: {"PhiS": 0.0, "src": 0.0} for k in EVALS}
near_ratio = np.inf
ctx_drift = []
for dr in (1e-1, 1e-2, 1e-3, 1e-4):
    for dth in (0.0, 1e-3):
        rf, thf = r_p + dr, th_p + dth
        dr_x = rf - r_p
        for m in (2, 10):
            gPhiS, _gd, gsrc = ctx.calc_m_gold(m, dr_x, dth)
            out = {"legacy": legacy.calc_m(m, rf, thf),
                   "ellfix": ellfix.calc_m(m, rf, thf),
                   "ctx": modern.calc_m(m, rf, thf)}
            er = {}
            for k in EVALS:
                er[k] = (rel(out[k]["PhiS"], gPhiS), rel(out[k]["src"], gsrc))
                worst[k]["PhiS"] = max(worst[k]["PhiS"], er[k][0])
                worst[k]["src"] = max(worst[k]["src"], er[k][1])
            for block in ("PhiS", "dPhiS", "d2PhiS", "src"):
                if not bitwise_equal(out["ellfix"][block], out["ctx"][block]):
                    ctx_drift.append(f"{block} at dr={dr:.0e} dth={dth:.0e} m={m}")
            print(f"  {dr:7.0e} {dth:7.0e} {m:3d} | {er['legacy'][0]:10.3e} "
                  f"{er['ellfix'][0]:10.3e} {er['ctx'][0]:10.3e} | {er['legacy'][1]:10.3e} "
                  f"{er['ellfix'][1]:10.3e} {er['ctx'][1]:10.3e}")
            if dr <= 1e-4 and dth == 0.0 and er["ellfix"][0] > 0.0:
                near_ratio = min(near_ratio, er["legacy"][0] / er["ellfix"][0])

print()
check("legacy PhiS_m degrades near the particle", worst["legacy"]["PhiS"] > 1e-10,
      f"(worst {worst['legacy']['PhiS']:.3e})")
check("legacy_elliptic PhiS_m holds machine precision", worst["ellfix"]["PhiS"] < 1e-13,
      f"(worst {worst['ellfix']['PhiS']:.3e})")
check("ctx PhiS_m holds machine precision", worst["ctx"]["PhiS"] < 1e-13,
      f"(worst {worst['ctx']['PhiS']:.3e})")
check("ctx expression body is still an unmodified copy of upstream",
      not ctx_drift,
      "(ctx == upstream + AGM, bitwise on PhiS/dPhiS/d2PhiS/src)"
      if not ctx_drift else f"(drift in: {', '.join(ctx_drift[:3])})")
check("the elliptic call alone accounts for the puncture gain", near_ratio > 100.0,
      f"(legacy/ellfix error ratio {near_ratio:.3g} at dr=1e-4, dtheta=0)")
check("src_m is not helped by the elliptic fix",
      worst["ellfix"]["src"] > 1e-4,
      f"(ellfix worst {worst['ellfix']['src']:.3e}, legacy {worst['legacy']['src']:.3e}, "
      f"ctx {worst['ctx']['src']:.3e})")

print("\nPart 3: legacy surface and global-state guard")
absent = [n for n in ("calc_offset", "calc_m_offset", "calc_PhiS_offset",
                      "calc_m_split", "calc_m_gold", "get_alpha", "_ctx")
          if not hasattr(legacy._ef, n)]
check("legacy exposes none of the post-refactor methods", len(absent) == 7,
      f"(absent: {len(absent)}/7)")

for mode in ("legacy", "legacy_elliptic"):
    try:
        EffectiveSource(mass=M, spin=A, mode=mode)
        check(f"second live {mode} instance is refused", False, "(no error raised)")
    except RuntimeError as exc:
        check(f"second live {mode} instance is refused", "global state" in str(exc))

del legacy, ellfix
gc.collect()
try:
    insp_legacy = Inspectre(spin=A, semilatus_rectum=P, eccentricity=ECC, mode="legacy")
    check("releasing a legacy instance clears the guard", True)
except RuntimeError as exc:
    check("releasing a legacy instance clears the guard", False, f"({exc})")
    insp_legacy = None

if insp_legacy is not None:
    for mode in ("legacy", "legacy_elliptic"):
        insp_legacy.es.mode = mode
        try:
            insp_legacy._ctx
            check(f"Inspectre(mode={mode!r})._ctx is refused", False, "(no error raised)")
        except RuntimeError as exc:
            check(f"Inspectre(mode={mode!r})._ctx is refused",
                  "no effsource_equatorial_ctx" in str(exc))
    insp_legacy.es.mode = "legacy"

    insp_legacy.set_particle(r_p, th_p, ph_p, ur)
    check("Inspectre(mode='legacy') still evaluates phi_s",
          np.isfinite(insp_legacy.phi_s(r_p + 1e-2, th_p, 0.0)))

print(f"\n{'ALL PASS' if failures == 0 else str(failures) + ' FAILURE(S)'}")
sys.exit(1 if failures else 0)
