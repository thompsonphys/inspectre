"""Regression test for the Chebyshev n-mode cache (inspectre/chebcache.py).

A_n(r, theta) is analytic in the field point off the sweep segment, so a
tensor-Chebyshev interpolant built from panel_nmodes_fast samples must
reproduce direct panel evaluation at arbitrary interior points.  Criterion
mirrors test_panel_gl.py:69 -- relative where the amplitude is alive, absolute
where the spectrum/nulls make relative meaningless.

Run inside the spectre conda env from the inspectre project root:
    PYTHONPATH=.:../effectivesource:../kerrgeodesic python test/test_chebcache.py
"""
import math
import sys

import numpy as np

from inspectre.inspectre import Inspectre
from inspectre.chebcache import (ChebCache, ChebCacheSet, nmodes_at,
                                 segment_distance)

A, P, E, M = 0.5, 10.0, 0.3, 2
N_LIST = [0, 16]
REL, ABS = 1e-8, 1e-10

failures = 0


def check(name, cond, detail=""):
    global failures
    print(f"[{'ok ' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures += 1


insp = Inspectre(spin=A, semilatus_rectum=P, eccentricity=E)
rmin, rmax = P / (1 + E), P / (1 - E)

# --- segment_distance geometry -----------------------------------------------
check("segment_distance inside range is 0",
      segment_distance(0.5 * (rmin + rmax), math.pi / 2, rmin, rmax) == 0.0)
check("segment_distance beyond apoapsis",
      abs(segment_distance(rmax + 1.5, math.pi / 2, rmin, rmax) - 1.5) < 1e-12)
check("segment_distance beyond periapsis",
      abs(segment_distance(rmin - 0.7, math.pi / 2, rmin, rmax) - 0.7) < 1e-12)
d_off = float(segment_distance(10.0, math.pi / 2 + 0.1, rmin, rmax))
check("segment_distance off-equator ~ r*dtheta",
      abs(d_off - 10.0 * math.sin(0.1)) < 5e-3, f"d={d_off:.4f}")

# --- cache vs direct panel on a boundary-like patch --------------------------
# patch beyond apoapsis, straddling the equator.  Sized so its METRIC extents
# (note theta counts as ~ r*dtheta) are comparable to the clearance: Chebyshev
# convergence is set by the singularity distance in units of the patch size,
# so patches must be sized ~ their clearance (the error-study notebook
# quantifies this; oversized patches converge slowly no matter how far most
# of the patch is from the segment).
patch = (rmax + 1.5, rmax + 3.0, math.pi / 2 - 0.05, math.pi / 2 + 0.05)
cch = ChebCache(insp, M, N_LIST, patch, Nr=14, Nth=14)
print(f"      patch clearance={cch.clearance:.3f}  "
      f"build={cch.build_seconds:.1f}s ({cch.Nr * cch.Nth} panel builds)")

decay = cch.coeff_decay()
check("coefficient decay >= 6 orders (all n)",
      bool((decay[:, 0] / np.maximum(decay[:, -1], 1e-300) > 1e6).all()),
      f"ratios={decay[:, 0] / np.maximum(decay[:, -1], 1e-300)}")

# accuracy: rel where the value is alive, else abs against
# max(1e-10, tol * sup-norm) -- the component's sup over the patch sets the
# recoverable scale (see ChebCache.sup_norm docstring: high-n components span
# decades across a patch), and 1e-10 is the panel reference's own abs floor.
sup = cch.sup_norm()
rng = np.random.default_rng(7)
worst = 0.0
npts = 12
for _ in range(npts):
    r = rng.uniform(patch[0], patch[1])
    th = rng.uniform(patch[2], patch[3])
    truth = insp.panel_nmodes_fast(M, N_LIST, r, th)
    ev = cch.eval(r, th)
    for q in range(len(N_LIST)):
        for blk, sl in ((0, [0]), (1, [1, 2, 3, 4]), (2, [5])):
            t = np.array(truth[q][blk])
            v = np.array(ev[q][blk])
            z_t = t[0::2] + 1j * t[1::2]
            z_v = v[0::2] + 1j * v[1::2]
            dabs = np.abs(z_v - z_t)
            rel = dabs / np.maximum(np.abs(z_t), 1e-300)
            floor = np.maximum(ABS, REL * sup[q, sl])
            worst = max(worst, float(np.where(rel < REL, 0.0,
                                              dabs / floor).max()))
check(f"cache vs panel at {npts} random points (all n, all components)",
      worst < 1.0,
      f"worst abs/floor={worst:.1e} (criterion rel<{REL} OR "
      f"abs<max({ABS}, {REL}*sup_norm))")

# --- switching logic ---------------------------------------------------------
cset = ChebCacheSet([cch])
pts = [(rmax + 1.7, math.pi / 2 + 0.02),   # in patch, clear  -> cheb
       (0.5 * (rmin + rmax), math.pi / 2 + 0.01),  # near segment -> panel
       (rmin - 1.0, math.pi / 2)]          # outside any patch -> panel
res, meth = nmodes_at(insp, M, N_LIST, pts, cset, d_switch=0.05)
check("switching tags", meth == ['cheb', 'panel', 'panel'], f"got {meth}")

direct = insp.panel_nmodes_fast(M, N_LIST, *pts[1])
same = all(np.array_equal(np.array(res[1][q][blk]), np.array(direct[q][blk]))
           for q in range(len(N_LIST)) for blk in range(3))
check("band point returns the exact panel result", same)

# no cacheset at all -> everything panel
_, meth2 = nmodes_at(insp, M, N_LIST, [pts[0]], cacheset=None)
check("no cache -> panel fallback", meth2 == ['panel'])

print(f"\n{failures} failure(s)")
sys.exit(1 if failures else 0)
