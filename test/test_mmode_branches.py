"""Grade the two m-mode branches for PhiS_m and calc_m/src over bound orbits.

Truth is test/mmode_reference.modes: the same Laurent sum the "legendre" branch
computes, carried in mpmath from the ctx A####, dAdt#### and d2Adt2####
coefficients. Part 1 validates that construction against the C before it is
used as a reference.

Orbits are filtered by inspectre.separatrix.is_bound, so the sweep never enters
plunge. Offsets cover both signs of dr and a theta-dominated alpha, both of
which sit outside the geometry a dr-only sweep reaches.

Part 1  Laurent construction vs the C, pointwise and coefficient-wise.
Part 2  each branch vs truth, per (m, C1), for PhiS_m and src.
Part 3  threshold sweep, per-case oracle, and branch="auto".
Part 4  the m > 20 ceiling and the branch guards.

Run inside the spectre conda env:
    python test/test_mmode_branches.py
"""
import itertools
import math
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "effectivesource"))
from inspectre import EffectiveSource, Inspectre
from inspectre.mmode import EI_TABLE_MMAX, calc_m, phi_s_m, switch_c1, switch_c1_src
from inspectre.mmode_laurent import leval, numerators, src_numerator
from inspectre.separatrix import is_bound
from mmode_reference import modes

SPINS = (-0.9, -0.5, 1e-8, 0.5, 0.9, 0.99)
PS = (6.0, 12.0, 20.0, 50.0)
ES = (0.0, 0.3, 0.7)
FRACS = (0.0, 0.31, 0.77)
GEOM = ("dr+", "dr-", "theta")
C1S = (3e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0)
MS = (2, 5, 10, 16, 20)
MS_HIGH = (25, 40)
QTY = ("phis", "src")
SWITCH = {"phis": switch_c1, "src": switch_c1_src}

failures = 0


def check(name, cond, detail=""):
    """(name, bool, detail) -> None; prints a result row and counts failures."""
    global failures
    print(f"[{'ok ' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures += 1


def build_cases():
    """() -> list of (label, EffectiveSource, ctx, r_p, geom) over bound orbits."""
    out = []
    for a, p, e in itertools.product(SPINS, PS, ES):
        if not is_bound(a, p, e):
            continue
        insp = Inspectre(spin=a, semilatus_rectum=p, eccentricity=e)
        if not math.isfinite(insp.mino_period_r):
            continue
        for frac in FRACS:
            r_p = insp.r_from_lambda(frac * insp.mino_period_r)
            if not math.isfinite(r_p):
                continue
            es = EffectiveSource(spin=a)
            es.set_particle(r_p, math.pi / 2,
                            insp.phi_from_lambda(frac * insp.mino_period_r),
                            insp.energy, insp.angular_momentum,
                            insp.four_velocity_equatorial(frac * insp.mino_period_r))
            ctx = es._ef._ctx
            if not all(math.isfinite(v) for v in
                       (ctx.alpha20, ctx.alpha02, ctx.beta, ctx.c, ctx.xp.phi,
                        ctx.rt, ctx.rtt, ctx.phit, ctx.phitt, ctx.dcdt, ctx.d2cdt2)):
                continue
            for geom in GEOM:
                out.append((f"a={a:g} p={p:g} e={e:g} f={frac:g} {geom}",
                            es, ctx, r_p, geom))
    return out


def offset(ctx, r_p, a, C1, geom):
    """(ctx, r_p, spin, C1, geom) -> (dr, dtheta) hitting that C1, or None."""
    alpha = C1 * ctx.beta
    r_h = 1.0 + math.sqrt(max(1.0 - a * a, 0.0))
    if geom == "theta":
        dtheta = math.sqrt(0.9 * alpha / ctx.alpha02)
        if dtheta > 1.4:
            return None
        dr = math.sqrt(0.1 * alpha / ctx.alpha20)
    else:
        dtheta = 0.0
        dr = math.sqrt(alpha / ctx.alpha20) * (1.0 if geom == "dr+" else -1.0)
    return None if r_p + dr < 1.10 * r_h else (dr, dtheta)


def rel(v, ref):
    """((re, im), mpc) -> relative error."""
    return float(abs(mp.mpc(v[0], v[1]) - ref) / abs(ref))


CASES = build_cases()
SPIN_SET = sorted({c[0].split()[0] for c in CASES})
print(f"matrix: {len(CASES)} (bound orbit, phase, geometry) cases, "
      f"{len(SPIN_SET)} spins, C1 in [{C1S[0]:g}, {C1S[-1]:g}]")

print("\nPart 1: Laurent construction vs the C")
worst_block = worst_point = worst_coef = 0.0
for label, es, ctx, r_p, geom in CASES[::7]:
    off = offset(ctx, r_p, 0.0, 5e-2, geom)
    if off is None:
        continue
    dr, dtheta = off
    num, s2 = numerators(ctx, dr, dtheta)
    sn, _ = src_numerator(ctx, dr, dtheta)
    err = scale = 0.0
    for dphi in (0.3, 1.7, -2.4, 3.0):
        dphib = dphi - ctx.c * dr
        den = leval(s2, dphib).real ** 5.5
        ref = es._ef.calc_offset(dr, dtheta, dphi)
        worst_block = max(worst_block,
                          abs((leval(num["PhiS"], dphib) / den).real - ref[0])
                          / max(abs(ref[0]), 1e-300))
        err = max(err, abs((leval(sn, dphib) / den).real - ref[3]))
        scale = max(scale, abs(ref[3]))
    worst_point = max(worst_point, err / max(scale, 1e-300))
    N = 128
    f = np.array([es._ef.calc_offset(dr, dtheta, 2 * math.pi * j / N + ctx.c * dr)[3]
                  * leval(s2, 2 * math.pi * j / N).real ** 5.5 for j in range(N)])
    C = np.fft.fft(f) / N
    mine = {i // 2: v for i, v in sn.items() if i % 2 == 0}
    sc = max(abs(v) for v in mine.values())
    worst_coef = max(worst_coef, max(abs(mine.get(n, 0j) - (C[n] if n >= 0 else C[N + n]))
                                     for n in range(-12, 13)) / sc)
check("PhiS Laurent reproduces the C pointwise", worst_block < 1e-6,
      f"(worst rel {worst_block:.2e})")
check("src Laurent coefficients match an FFT of the C", worst_coef < 1e-11,
      f"(worst rel {worst_coef:.2e})")
check("src Laurent reproduces the C pointwise", worst_point < 1e-5,
      f"(worst rel {worst_point:.2e}; pointwise sums cancel, coefficients are the test)")

print("\nPart 2: worst relative error over the matrix, per (m, C1)")
bins = {}
percase = {}
for label, es, ctx, r_p, geom in CASES:
    a = float(label.split()[0][2:])
    for C1 in C1S:
        off = offset(ctx, r_p, a, C1, geom)
        if off is None:
            continue
        dr, dtheta = off
        RP, RS = modes(ctx, MS, dr, dtheta)
        for m in MS:
            for key, R in (("phis", RP), ("src", RS)):
                if abs(R[m]) == 0:
                    continue
                if key == "phis":
                    va = phi_s_m(es._ef, m, dr, dtheta, branch="ei_table")[:2]
                    vb = phi_s_m(es._ef, m, dr, dtheta, branch="legendre")[:2]
                else:
                    va = calc_m(es._ef, m, dr, dtheta, branch="ei_table")[3]
                    vb = calc_m(es._ef, m, dr, dtheta, branch="legendre")[3]
                ea, eb = rel(va, R[m]), rel(vb, R[m])
                for br, e in (("ei_table", ea), ("legendre", eb)):
                    bins[(key, br, m, C1)] = max(bins.get((key, br, m, C1), 0.0), e)
                percase.setdefault((key, label, m), []).append((C1, ea, eb))

for key in QTY:
    for br in ("ei_table", "legendre"):
        print(f"\n  {key} / {br}")
        print("      m " + " ".join(f"{c:>9.0e}" for c in C1S))
        for m in MS:
            print(f"  {m:5d} " + " ".join(
                f"{bins.get((key, br, m, c), float('nan')):9.1e}" for c in C1S))

print("\nPart 3: threshold sweep, per-case oracle, and branch='auto'")
for key in QTY:
    print(f"\n  {key}")
    print(f"    {'coeff':>7} {'exp':>5} " + " ".join(f"{'m=%d' % m:>9}" for m in MS)
          + f" {'worst':>9}")
    for coeff, expo in ((0.5, 1.2), (0.45, 1.1), (0.25, 1.1), (1.0, 1.2)):
        row, hi = [], 0.0
        for m in MS:
            thr = coeff / m ** expo
            e = max([bins.get((key, "ei_table", m, c), 0.0) for c in C1S if c <= thr]
                    + [bins.get((key, "legendre", m, c), 0.0) for c in C1S if c > thr])
            row.append(f"{e:9.1e}")
            hi = max(hi, e)
        print(f"    {coeff:7.3g} {expo:5.2f} " + " ".join(row) + f" {hi:9.1e}")
    fit = orc = 0.0
    for (k, label, m), pts in percase.items():
        if k != key or len(pts) < 2:
            continue
        thr = SWITCH[key](m)
        fit = max(fit, max(eb if c > thr else ea for c, ea, eb in pts))
        orc = max(orc, min(max(max(ea for c, ea, eb in pts if c <= t),
                               max([eb for c, ea, eb in pts if c > t] or [0.0]))
                           for t, _, _ in pts[:-1]))
    print(f"    shipped threshold worst {fit:.2e}; best per-case threshold {orc:.2e} "
          f"({'no gain' if orc > 0.5 * fit else 'gain available'}) -- the error is the "
          f"width of the window where neither branch is accurate, not the switch point")

worst_auto = {k: 0.0 for k in QTY}
worst_at = {k: "" for k in QTY}
counts = {"ei_table": 0, "legendre": 0}
for label, es, ctx, r_p, geom in CASES:
    a = float(label.split()[0][2:])
    for C1 in C1S:
        off = offset(ctx, r_p, a, C1, geom)
        if off is None:
            continue
        dr, dtheta = off
        RP, RS = modes(ctx, MS, dr, dtheta)
        for m in MS:
            p = phi_s_m(es._ef, m, dr, dtheta)
            s = calc_m(es._ef, m, dr, dtheta)
            counts[s[4]] += 1
            for key, v, R in (("phis", p[:2], RP), ("src", s[3], RS)):
                if abs(R[m]) == 0:
                    continue
                e = rel(v, R[m])
                if e > worst_auto[key]:
                    worst_auto[key] = e
                    worst_at[key] = f"m={m} C1={C1:.0e} {label}"
check("auto dispatch accurate for PhiS_m over the bound-orbit matrix",
      worst_auto["phis"] < 1e-6, f"(worst rel {worst_auto['phis']:.2e} at {worst_at['phis']})")
check("auto dispatch accurate for src over the bound-orbit matrix",
      worst_auto["src"] < 1e-2, f"(worst rel {worst_auto['src']:.2e} at {worst_at['src']})")
print(f"       dispatch counts: {counts}")

print("\nPart 4: ceiling and guards")
worst_high = worst_high_below = 0.0
for label, es, ctx, r_p, geom in CASES[::5]:
    a = float(label.split()[0][2:])
    for C1 in C1S:
        off = offset(ctx, r_p, a, C1, geom)
        if off is None:
            continue
        dr, dtheta = off
        RP, _ = modes(ctx, MS_HIGH, dr, dtheta)
        for m in MS_HIGH:
            if abs(RP[m]) == 0:
                continue
            re, im, used = phi_s_m(es._ef, m, dr, dtheta)
            if used != "legendre":
                check("m > 20 routes to legendre", False, f"(got {used})")
            e = rel((re, im), RP[m])
            if C1 > switch_c1(m):
                worst_high = max(worst_high, e)
            else:
                worst_high_below = max(worst_high_below, e)
check("m beyond the ei_table ceiling is supported above the switch",
      worst_high < 1e-6, f"(worst rel {worst_high:.2e} for m in {MS_HIGH})")
print(f"       below the switch m > {EI_TABLE_MMAX} has no ei_table alternative; "
      f"legendre there is worst {worst_high_below:.2e}")

es0 = CASES[0][1]
try:
    phi_s_m(es0._ef, 25, 1.0, 0.0, branch="ei_table")
    check("ei_table rejects m > 20", False, "(no error raised)")
except ValueError as exc:
    check("ei_table rejects m > 20", "supports m <= 20" in str(exc))
try:
    EffectiveSource(spin=0.5, impl="original").phi_s_m_offset(5, 1.0, 0.0)
    check("branch API refuses non-refactored impls", False, "(no error raised)")
except NotImplementedError:
    check("branch API refuses non-refactored impls", True)

print("\nswitch thresholds  PhiS: "
      + ", ".join(f"m={m}:{switch_c1(m):.4g}" for m in MS)
      + "\n                    src: "
      + ", ".join(f"m={m}:{switch_c1_src(m):.4g}" for m in MS))
print(f"\n{failures} failure(s)")
sys.exit(1 if failures else 0)
