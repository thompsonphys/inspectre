"""Attribute the src m-mode double error to each branch's condition number.

The error is kappa * eps, where kappa = sum|term| / |sum| for whichever branch
is in use. No single arithmetic site owns it: the block polynomials, the Laurent
assembly and the wave-operator weighted sum are all accurate to rounding, and
kappa then amplifies each of them equally.

Part 1 checks kappa * eps predicts the measured double error of the legendre
branch across the C1 crossover.
Part 2 checks the near-branch seven-channel assembly, where the puncture has
kappa = 1 and src saturates.
Part 3 tabulates both branches at both precisions so the switch can be placed at
the crossing.

Run inside the spectre conda env:
    python test/test_conditioning.py
"""
import math
import os
import sys

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inspectre.mmode import switch_c1_src
from inspectre.mmode_laurent import Std, numerators, src_contract, src_weights
from inspectre.source import EffectiveSource
from mmode_reference import MP, kernel_mp

EPS = 2.220446049250313e-16
CLD = np.clongdouble
DIRECTION = (8.0, 0.5)
TS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0)
GEOM = (("apo", 14.2857), ("peri", 7.6923))

failures = 0


def check(name, cond, detail=""):
    global failures
    print(f"[{'ok ' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures += 1


class Ld:
    """Extended-precision numeric context backed by numpy long double."""

    mpc = staticmethod(lambda re, im=0: CLD(re) if isinstance(re, complex)
                       else CLD(complex(re, im)))
    mpf = staticmethod(np.longdouble)
    sin = staticmethod(np.sin)
    cos = staticmethod(np.cos)


def half(poly):
    """Laurent in exp(i dphib/2) -> the even part reindexed in exp(i dphib)."""
    return {i // 2: v for i, v in poly.items() if i % 2 == 0} or {0: 0}


def src_poly(ctx, dr, dth, N):
    """(ctx, dr, dth, numeric context) -> src Laurent coefficients in that precision."""
    return half(src_contract(numerators(ctx, dr, dth, N)[0], *src_weights(ctx, dr, dth, N)))


def legendre_row(ctx, m, dr, dth, dps=90):
    """(ctx, m, dr, dth) -> (kappa, double error, long double error) of the contraction."""
    mp.mp.dps = dps
    alpha = mp.mpf(ctx.alpha20) * mp.mpf(dr) ** 2 + mp.mpf(ctx.alpha02) * mp.mpf(dth) ** 2
    Tm, Td, Tl = (src_poly(ctx, dr, dth, N) for N in (MP, Std, Ld))
    S = kernel_mp(m + max(abs(n) for n in Tm), alpha, mp.mpf(ctx.beta))
    terms = [Tm[n] * S[abs(m - n)] for n in Tm]
    ref = sum(terms)
    kappa = float(sum(abs(t) for t in terms) / abs(ref))
    e = lambda v: float(abs(mp.mpc(complex(v)) - ref) / abs(ref))
    acc = CLD(0)
    for n in Tl:
        acc = acc + CLD(Tl[n]) * CLD(complex(S[abs(m - n)]))
    return kappa, e(sum(complex(Td[n]) * complex(S[abs(m - n)]) for n in Td)), e(acc)


def channel_kappa(ef, m, dr, dth, dps=60):
    """(wrapper, m, dr, dth) -> (src, PhiS) kappa of the seven-channel assembly."""
    mp.mp.dps = dps
    ctx = ef._ctx
    alpha = mp.mpf(ctx.alpha20) * mp.mpf(dr) ** 2 + mp.mpf(ctx.alpha02) * mp.mpf(dth) ** 2
    lg = mp.log(alpha)
    PhiS_s, _, _, src_s = ef.calc_m_split(m, dr, dth)
    out = []
    for blocks in (src_s, PhiS_s):
        terms = [mp.mpc(*blocks[0]), mp.mpc(*blocks[1]) * lg]
        terms += [mp.mpc(*blocks[q]) / alpha ** (q - 1) for q in range(2, 7)]
        out.append(float(sum(abs(t) for t in terms) / abs(sum(terms))))
    return out


es = EffectiveSource(spin=0.5)

print("legendre branch: kappa * eps against the measured double error\n")
print("%-5s %3s %9s %7s %11s %11s %11s %8s"
      % ("ph", "m", "C1", "C1/sw", "kappa", "kappa*eps", "measured", "ratio"))
worst_ratio = 0.0
for tag, r_p in GEOM:
    es.set_particle(r_p, math.pi / 2, 0.0, 0.955, 3.6, 0.0)
    ctx = es._ef._ctx
    for m in (10, 20):
        for t in TS:
            dr, dth = t * DIRECTION[0], t * DIRECTION[1]
            C1 = (ctx.alpha20 * dr * dr + ctx.alpha02 * dth * dth) / ctx.beta
            kappa, ed, _ = legendre_row(ctx, m, dr, dth)
            ratio = ed / (kappa * EPS)
            worst_ratio = max(worst_ratio, ratio)
            print("%-5s %3d %9.4g %7.2f %11.2e %11.2e %11.2e %8.1f"
                  % (tag, m, C1, C1 / switch_c1_src(m), kappa, kappa * EPS, ed, ratio))
check("kappa * eps predicts the legendre double error", worst_ratio < 100,
      f"worst measured / predicted = {worst_ratio:.1f}")

print("\nnear branch: seven-channel assembly conditioning\n")
es.set_particle(10.0, math.pi / 2, 0.0, 0.955, 3.6, 0.01)
print("%3s %8s %8s %10s %11s %11s" % ("m", "dr", "dth", "alpha", "kappa src", "kappa PhiS"))
worst_phis_kappa = 0.0
sat = []
for m in (10, 20):
    for dr, dth in ((1e-1, 0.0), (1e-2, 0.0), (1e-3, 0.0), (1e-4, 0.0), (1e-5, 0.0),
                    (0.0, 1e-3), (0.0, 1e-4), (0.0, 1e-5)):
        alpha = es._ef._ctx.alpha20 * dr * dr + es._ef._ctx.alpha02 * dth * dth
        ks, kp = channel_kappa(es._ef, m, dr, dth)
        worst_phis_kappa = max(worst_phis_kappa, kp)
        if alpha < 1e-5:
            sat.append(ks)
        print("%3d %8.0e %8.0e %10.3e %11.2e %11.2e" % (m, dr, dth, alpha, ks, kp))
check("near-branch puncture is perfectly conditioned", worst_phis_kappa < 2.0,
      f"worst kappa = {worst_phis_kappa:.2f}")
check("near-branch src conditioning saturates", max(sat) / min(sat) < 3.0,
      f"spread over alpha < 1e-5 = {max(sat) / min(sat):.2f}x, level = {max(sat):.2e}")

print("\nboth branches, both precisions, across the crossover\n")
worst_chosen = 0.0
for tag, r_p in GEOM:
    es.set_particle(r_p, math.pi / 2, 0.0, 0.955, 3.6, 0.0)
    ctx = es._ef._ctx
    for m in (10, 20):
        print(f"\n  m={m} {tag} r_p={r_p}  switch C1={switch_c1_src(m):.4g}\n")
        print("  %8s %8s %9s %7s %10s %10s %10s %10s %10s"
              % ("dr", "dth", "C1", "C1/sw", "leg dbl", "leg ld", "tab dbl", "tab ld", "chosen"))
        for t in TS:
            dr, dth = t * DIRECTION[0], t * DIRECTION[1]
            C1 = (ctx.alpha20 * dr * dr + ctx.alpha02 * dth * dth) / ctx.beta
            _, ld_, ll = legendre_row(ctx, m, dr, dth)
            mp.mp.dps = 90
            alpha = mp.mpf(ctx.alpha20) * mp.mpf(dr) ** 2 + mp.mpf(ctx.alpha02) * mp.mpf(dth) ** 2
            Tm = src_poly(ctx, dr, dth, MP)
            S = kernel_mp(m + max(abs(n) for n in Tm), alpha, mp.mpf(ctx.beta))
            arg = m * (mp.mpf(ctx.xp.phi) + mp.mpf(ctx.c) * mp.mpf(dr))
            ref = sum(Tm[n] * S[abs(m - n)] for n in Tm) * (mp.cos(arg) - 1j * mp.sin(arg))
            td = float(abs(mp.mpc(*es._ef.calc_m_offset(m, dr, dth)[3]) - ref) / abs(ref))
            tl = float(abs(mp.mpc(*es._ef.calc_m_extended(m, dr, dth)[2]) - ref) / abs(ref))
            chosen = ld_ if C1 > switch_c1_src(m) else td
            worst_chosen = max(worst_chosen, chosen)
            print("  %8.3f %8.3f %9.4g %7.2f %10.2e %10.2e %10.2e %10.2e %10.2e"
                  % (dr, dth, C1, C1 / switch_c1_src(m), ld_, ll, td, tl, chosen))

check("the branch the switch selects stays under 1e-5", worst_chosen < 1e-5,
      f"worst = {worst_chosen:.2e}")

print(f"\n{failures} failure(s)")
sys.exit(1 if failures else 0)
