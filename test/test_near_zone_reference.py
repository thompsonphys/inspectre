"""Validate the m-mode reference in the deep near zone and grade the C there.

The reference kernel is the closed-form minimal solution Q^q_{mu-1/2}(z), which
costs the same at every alpha; the downward recurrence it replaces needs a guard
of (dps + 15) ln 10 / -ln(z - sqrt(z^2 - 1)) terms, which is 1.3e8 at alpha =
1e-10 and so cannot reach the AMR regime.

Part 1 cross-checks that kernel against independent sinh-substituted quadrature,
and against the recurrence wherever the recurrence is still affordable.
Part 2 reports the contraction cancellation and fixes dps from it.
Part 3 grades calc_m_offset and calc_m_extended on a dr/dtheta grid reaching
alpha = 1e-10, each row carrying its own self-convergence figure.

Run inside the spectre conda env:
    python test/test_near_zone_reference.py
"""
import math
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inspectre.source import EffectiveSource
from mmode_reference import (cancellation, dps_for, kernel_guard, kernel_legendre,
                             kernel_miller, kernel_mp, kernel_quad, modes)

ALPHAS = ("1e-2", "1e-4", "1e-6", "1e-8", "1e-10")
MUS = (0, 1, 10, 34)
PTS = ((1e-1, 0.0), (1e-2, 0.0), (1e-3, 0.0), (1e-4, 0.0), (1e-5, 0.0),
       (0.0, 1e-3), (0.0, 1e-4), (0.0, 1e-5), (1e-5, 1e-5), (1e-6, 1e-6))

failures = 0


def check(name, cond, detail=""):
    global failures
    print(f"[{'ok ' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures += 1


es = EffectiveSource(spin=0.5)
es.set_particle(10.0, math.pi / 2, 0.0, 0.955, 3.6, 0.01)
ctx = es._ef._ctx
BETA = mp.mpf(ctx.beta)

print("closed-form kernel vs sinh-substituted quadrature\n")
print("%8s %5s %12s" % ("alpha", "mu", "rel"))
mp.mp.dps = 40
worst_kernel = 0.0
for a in ALPHAS:
    alpha = mp.mpf(a)
    S = kernel_mp(max(MUS), alpha, BETA)
    for mu in MUS:
        rel = abs(kernel_quad(mu, alpha, BETA) - S[mu]) / abs(S[mu])
        worst_kernel = max(worst_kernel, float(rel))
        print("%8s %5d %12.2e" % (a, mu, float(rel)))
check("closed-form kernel matches independent quadrature", worst_kernel < 1e-30,
      f"worst rel = {worst_kernel:.2e}")

print("\nclosed form vs downward recurrence where both are affordable\n")
print("%8s %8s %12s" % ("alpha", "guard", "rel"))
worst_route = 0.0
for a in ALPHAS:
    alpha = mp.mpf(a)
    guard = kernel_guard(alpha, BETA)
    if guard > 20000:
        print("%8s %8d %12s" % (a, guard, "recurrence unaffordable"))
        continue
    Sm = kernel_miller(max(MUS), alpha, BETA, 5, guard)
    Sl = kernel_legendre(max(MUS), alpha, BETA, 5)
    rel = max(abs(Sm[mu] - Sl[mu]) / abs(Sl[mu]) for mu in MUS)
    worst_route = max(worst_route, float(rel))
    print("%8s %8d %12.2e" % (a, guard, float(rel)))
check("the two kernel routes agree where they overlap", worst_route < 1e-35,
      f"worst rel = {worst_route:.2e}")

print("\ncontraction cancellation and the dps it forces\n")
print("%8s %8s %10s %12s %12s %6s" % ("dr", "dth", "alpha", "cancel PhiS", "cancel src", "dps"))
for dr, dth in PTS:
    alpha = ctx.alpha20 * dr * dr + ctx.alpha02 * dth * dth
    cp, cs = cancellation(ctx, 10, dr, dth)
    print("%8.0e %8.0e %10.3e %12.2e %12.2e %6d"
          % (dr, dth, alpha, float(cp), float(cs), dps_for(ctx, 10, dr, dth)))

print("\nC near-zone path graded against the reference\n")
print("%3s %8s %8s %10s %5s %9s | %10s %10s | %10s %10s"
      % ("m", "dr", "dth", "alpha", "dps", "selfconv",
         "PhiS dbl", "PhiS ext", "src dbl", "src ext"))
worst_conv = 0.0
worst_phis = 0.0
worst_src = 0.0
for m in (10, 20):
    for dr, dth in PTS:
        alpha = ctx.alpha20 * dr * dr + ctx.alpha02 * dth * dth
        d = dps_for(ctx, m, dr, dth)
        p1, s1 = modes(ctx, [m], dr, dth, dps=d)
        p2, s2 = modes(ctx, [m], dr, dth, dps=d + 25)
        conv = max(abs(p1[m] - p2[m]) / abs(p2[m]), abs(s1[m] - s2[m]) / abs(s2[m]))
        rp, rs = complex(p2[m]), complex(s2[m])
        o = es._ef.calc_m_offset(m, dr, dth)
        pd, sd = complex(*o[0]), complex(*o[3])
        pe, _, se = es._ef.calc_m_extended(m, dr, dth)
        pe, se = complex(*pe), complex(*se)
        ep, ee_, esd, ese = (abs(pd - rp) / abs(rp), abs(pe - rp) / abs(rp),
                             abs(sd - rs) / abs(rs), abs(se - rs) / abs(rs))
        worst_conv = max(worst_conv, float(conv))
        worst_phis = max(worst_phis, ep)
        worst_src = max(worst_src, esd)
        print("%3d %8.0e %8.0e %10.3e %5d %9.1e | %10.2e %10.2e | %10.2e %10.2e"
              % (m, dr, dth, alpha, d, float(conv), ep, ee_, esd, ese))

check("reference self-converges across the grid", worst_conv < 1e-25,
      f"worst = {worst_conv:.2e}")
check("double puncture holds to rounding at every alpha", worst_phis < 1e-13,
      f"worst rel = {worst_phis:.2e}")
check("double src holds its floor at every alpha", worst_src < 1e-4,
      f"worst rel = {worst_src:.2e}")

print(f"\n{failures} failure(s)")
sys.exit(1 if failures else 0)
