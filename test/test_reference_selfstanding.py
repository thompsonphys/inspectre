"""Validate the mpmath m-mode reference using only the pointwise C evaluator.

The reference is a Laurent numerator over a kernel. Neither part needs an
independent m-mode implementation to be checked:

Part 1 grades the kernel against two independent constructions -- the downward
recurrence and sinh-substituted quadrature -- which use no C at all.
Part 2 grades the Laurent numerators against effsource_equatorial_ctx_calc_offset,
the pointwise evaluator, by FFT in dphi.
Part 3 closes the loop, grading the assembled reference against a phi-projection
of that same pointwise evaluator.

Together these establish the reference without calc_m, calc_m_offset,
calc_m_split or calc_m_extended, so it survives their removal.

Run inside the spectre conda env:
    python test/test_reference_selfstanding.py
"""
import math
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inspectre.mmode_laurent import (Std, numerators, src_contract, src_numerator,
                                     src_weights)
from inspectre.source import EffectiveSource
from mmode_reference import (MP, kernel_guard, kernel_miller, kernel_mp, kernel_quad)

CASES = ((0.5, 10.0), (0.0, 12.0), (0.9, 8.0))
MUS = (0, 1, 10, 34)
ALPHAS = ("1e-2", "1e-4", "1e-6", "1e-8", "1e-10")
MS = (2, 10, 20)

failures = 0


def check(name, cond, detail=""):
    global failures
    print(f"[{'ok ' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures += 1


def half(poly):
    """Laurent in exp(i dphib/2) -> even part reindexed in exp(i dphib)."""
    return {i // 2: v for i, v in poly.items() if i % 2 == 0} or {0: 0}


def nstar(C1, m):
    """(C1, m) -> trapezoid nodes for the phi-projection."""
    return max(64, 2 * int(37.0 / math.sqrt(C1) / 2 + 1), 2 * int(1.4 * m) + 2)


es = EffectiveSource(spin=0.5)
es.set_particle(10.0, math.pi / 2, 0.0, 0.955, 3.6, 0.0)
BETA = mp.mpf(es._ef._ctx.beta)

print("Part 1: kernel against two independent constructions (no C)\n")
print("%8s %8s %13s %13s" % ("alpha", "guard", "vs quadrature", "vs recurrence"))
mp.mp.dps = 40
worst_quad = worst_rec = 0.0
for a in ALPHAS:
    alpha = mp.mpf(a)
    S = kernel_mp(max(MUS), alpha, BETA)
    eq = max(abs(kernel_quad(mu, alpha, BETA) - S[mu]) / abs(S[mu]) for mu in MUS)
    worst_quad = max(worst_quad, float(eq))
    guard = kernel_guard(alpha, BETA)
    if guard <= 20000:
        Sm = kernel_miller(max(MUS), alpha, BETA, 5, guard)
        er = max(abs(Sm[mu] - S[mu]) / abs(S[mu]) for mu in MUS)
        worst_rec = max(worst_rec, float(er))
        print("%8s %8d %13.2e %13.2e" % (a, guard, float(eq), float(er)))
    else:
        print("%8s %8d %13.2e %13s" % (a, guard, float(eq), "n/a"))
check("kernel matches sinh-substituted quadrature", worst_quad < 1e-30,
      f"worst rel = {worst_quad:.2e}")
check("kernel matches the downward recurrence where affordable", worst_rec < 1e-35,
      f"worst rel = {worst_rec:.2e}")

print("\nPart 2: Laurent numerators against the pointwise C, by FFT in dphi\n")
print("%4s %6s %8s %8s %13s %13s" % ("a", "r_p", "dr", "dth", "src coeffs", "PhiS coeffs"))
worst_src = worst_phis = 0.0
for a, r_p in CASES:
    e = EffectiveSource(spin=a)
    e.set_particle(r_p, math.pi / 2, 0.0, 0.955, 3.6, 0.0)
    ctx = e._ef._ctx
    for dr, dth in ((0.8, 0.10), (-1.5, 0.25), (2.5, 0.05)):
        num, s2 = numerators(ctx, dr, dth, Std)
        sn, _ = src_numerator(ctx, dr, dth, Std)
        N = 128
        idx = 2 * np.pi * np.arange(N) / N
        den = np.array([(ctx.alpha20 * dr * dr + ctx.alpha02 * dth * dth
                         + ctx.beta * math.sin(0.5 * x) ** 2) for x in idx])
        raw = np.array([e._ef.calc_offset(dr, dth, x + ctx.c * dr) for x in idx],
                       dtype=object)
        fs = np.array([r[3] for r in raw]) * den ** 5.5
        fp = np.array([r[0] for r in raw]) * den ** 5.5
        for poly, f, tag in ((half(sn), fs, "src"), (half(num["PhiS"]), fp, "PhiS")):
            C = np.fft.fft(f) / N
            sc = max(abs(v) for v in poly.values())
            w = max(abs(poly.get(n, 0j) - (C[n] if n >= 0 else C[N + n]))
                    for n in range(-12, 13)) / sc
            if tag == "src":
                worst_src = max(worst_src, w)
                ws = w
            else:
                worst_phis = max(worst_phis, w)
                wp = w
        print("%4.1f %6.1f %8.2f %8.2f %13.2e %13.2e" % (a, r_p, dr, dth, ws, wp))
check("src Laurent coefficients match the pointwise C", worst_src < 1e-11,
      f"worst rel = {worst_src:.2e}")
check("PhiS Laurent coefficients match the pointwise C", worst_phis < 1e-11,
      f"worst rel = {worst_phis:.2e}")

print("\nPart 3: assembled reference against a phi-projection of the pointwise C\n")
print("%4s %6s %8s %3s %10s %13s" % ("a", "r_p", "C1", "m", "N*", "rel"))
worst_close = 0.0
for a, r_p in CASES:
    e = EffectiveSource(spin=a)
    e.set_particle(r_p, math.pi / 2, 0.0, 0.955, 3.6, 0.0)
    ctx, ef = e._ef._ctx, e._ef
    for dr, dth in ((0.8, 0.10), (-1.5, 0.25)):
        C1 = (ctx.alpha20 * dr * dr + ctx.alpha02 * dth * dth) / ctx.beta
        mp.mp.dps = 70
        alpha = mp.mpf(ctx.alpha20) * mp.mpf(dr) ** 2 + mp.mpf(ctx.alpha02) * mp.mpf(dth) ** 2
        T = half(src_contract(numerators(ctx, dr, dth, MP)[0],
                              *src_weights(ctx, dr, dth, MP)))
        for m in MS:
            N = nstar(C1, m)
            psi = 2.0 * np.pi * np.arange(N) / N
            F = np.fft.fft(np.array([ef.calc_offset(dr, dth, x)[3] for x in psi])) / N
            proj = 2.0 * np.pi * F[m % N] * np.exp(-1j * m * ctx.xp.phi)
            S = kernel_mp(m + max(abs(n) for n in T), alpha, mp.mpf(ctx.beta))
            arg = m * (mp.mpf(ctx.xp.phi) + mp.mpf(ctx.c) * mp.mpf(dr))
            ref = sum(T[n] * S[abs(m - n)] for n in T) * (mp.cos(arg) - 1j * mp.sin(arg))
            rel = float(abs(mp.mpc(complex(proj)) - ref) / abs(ref))
            worst_close = max(worst_close, rel)
            print("%4.1f %6.1f %8.4g %3d %10d %13.2e" % (a, r_p, C1, m, N, rel))
check("assembled reference matches the pointwise C projected in phi", worst_close < 1e-8,
      f"worst rel = {worst_close:.2e}")

print(f"\n{failures} failure(s)")
sys.exit(1 if failures else 0)
