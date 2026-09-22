"""Grade the projection branch of calc_m against the mpmath reference.

One sweep of the pointwise evaluator in the field-point azimuth yields every m at
once. The branch is accurate while m sqrt(C1) < 12; this checks that bound holds
across spin, orbit, offset direction and mode, for src and for every derivative.

Run inside the spectre conda env:
    python test/test_projection_branch.py
"""
import math
import os
import sys

import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inspectre.mmode import calc_m_projection, projection_nodes
from inspectre.mmode_laurent import numerators, src_contract, src_weights
from inspectre.source import EffectiveSource
from mmode_reference import MP, kernel_mp

CASES = ((0.5, 10.0), (0.0, 12.0), (0.9, 8.0))
MIXES = (("r", 0.01), ("bal", 1.0), ("th", 100.0))
C1S = (0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3)
MS = (2, 10, 20)
# (label, slot in the calc_m tuple, index, numerators() key)
COMPS = (("PhiS", 0, 0, "PhiS"), ("dt", 1, 0, "dt"), ("dr", 1, 1, "dr"),
         ("dth", 1, 2, "dth"), ("dph", 1, 3, "dph"), ("dr2", 2, 4, "dr2"),
         ("dth2", 2, 7, "dth2"), ("dph2", 2, 9, "dph2"))

failures = 0


def check(name, cond, detail=""):
    global failures
    print(f"[{'ok ' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures += 1


def half(poly):
    return {i // 2: v for i, v in poly.items() if i % 2 == 0} or {0: 0}


def offsets(ctx, C1, rho):
    alpha = C1 * ctx.beta
    dr = math.sqrt(alpha / ((1.0 + rho) * ctx.alpha20))
    dth = math.sqrt(alpha * rho / ((1.0 + rho) * ctx.alpha02))
    return None if dr > 10.88 or dth > 0.524 else (dr, dth)


print("projection branch vs the mpmath reference\n")
print("%4s %6s %-4s %8s %3s %6s %7s %11s %11s"
      % ("a", "r_p", "mix", "C1", "m", "nodes", "m*sqrtC1", "src", "worst comp"))
worst_src = worst_any = 0.0
worst_where = None
npts = 0
for a, r_p in CASES:
    es = EffectiveSource(spin=a)
    es.set_particle(r_p, math.pi / 2, 0.0, 0.955, 3.6, 0.0)
    ctx = es._ef._ctx
    for mix, rho in MIXES:
        for C1 in C1S:
            o = offsets(ctx, C1, rho)
            if o is None:
                continue
            dr, dth = o
            got = calc_m_projection(es._ef, MS, dr, dth)
            mp.mp.dps = 60 + int(12 * max(0, -math.log10(C1)))
            alpha = mp.mpf(ctx.alpha20) * mp.mpf(dr) ** 2 \
                + mp.mpf(ctx.alpha02) * mp.mpf(dth) ** 2
            nums, _ = numerators(ctx, dr, dth, MP)
            polys = {lab: half(nums[key]) for lab, _, _, key in COMPS}
            polys["src"] = half(src_contract(nums, *src_weights(ctx, dr, dth, MP)))
            for m in MS:
                errs = {}
                for lab in [c[0] for c in COMPS] + ["src"]:
                    T = polys[lab]
                    S = kernel_mp(m + max(abs(n) for n in T), alpha, mp.mpf(ctx.beta))
                    arg = m * (mp.mpf(ctx.xp.phi) + mp.mpf(ctx.c) * mp.mpf(dr))
                    ref = sum(T[n] * S[abs(m - n)] for n in T) \
                        * (mp.cos(arg) - 1j * mp.sin(arg))
                    if lab == "src":
                        v = complex(*got[m][3])
                    else:
                        slot, idx = next((c[1], c[2]) for c in COMPS if c[0] == lab)
                        blk = got[m][slot]
                        v = complex(blk[0], blk[1]) if slot == 0 \
                            else complex(blk[2 * idx], blk[2 * idx + 1])
                    errs[lab] = float(abs(mp.mpc(v) - ref) / abs(ref))
                npts += 1
                worst_src = max(worst_src, errs["src"])
                lab_w = max(errs, key=errs.get)
                if errs[lab_w] > worst_any:
                    worst_any, worst_where = errs[lab_w], (a, r_p, mix, C1, m, lab_w)
                print("%4.1f %6.1f %-4s %8.4g %3d %6d %7.1f %11.2e %11.2e (%s)"
                      % (a, r_p, mix, C1, m, projection_nodes(C1, m),
                         m * math.sqrt(C1), errs["src"], errs[lab_w], lab_w))

check("src through the projection branch", worst_src < 1e-7,
      f"worst = {worst_src:.2e} over {npts} (case, m)")
check("every component through the projection branch", worst_any < 1e-6,
      f"worst = {worst_any:.2e} at a={worst_where[0]} r_p={worst_where[1]} "
      f"{worst_where[2]} C1={worst_where[3]:.3g} m={worst_where[4]} ({worst_where[5]})")
check("node rule is a power of two and clears Nyquist",
      all(projection_nodes(c, m) >= 1.4 * m
          and projection_nodes(c, m) & (projection_nodes(c, m) - 1) == 0
          for c in C1S for m in MS), "")

print(f"\n{failures} failure(s)")
sys.exit(1 if failures else 0)
