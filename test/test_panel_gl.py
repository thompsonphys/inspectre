"""Regression test for INSPECTRE_INTEG_PANEL_GL (panel Gauss-Legendre nodes
split at the particle's closest approach).

1. Off-orbit accuracy: panel amplitudes match tight-tolerance QAG_MINO at
   z-face / near-equatorial interior / r-face points, including high n.
2. Crossing point: puncture (PhiS) n-modes self-converge under order/level
   refinement. (Source n-modes are genuinely divergent there -- the m-mode
   effective source grows like 1/dr^2 at the particle -- and derivative
   n-modes are at best principal-value; neither is checked.)
3. The wrap fold: breakpoints rotate the integration window to
   [lambda_c, lambda_c + Vr]; agreement with QAG (which integrates [0, Vr])
   verifies the periodic fold of the wrapped nodes.

Run inside the spectre conda env from the inspectre project root:
    PYTHONPATH=.:../effectivesource:../kerrgeodesic python test/test_panel_gl.py
"""
import math
import sys

import inspectre_c as ic
from inspectre.inspectre import Inspectre

A, P, M = 0.5, 10.0, 2

failures = 0


def check(name, cond, detail=""):
    global failures
    print(f"[{'ok ' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures += 1


def cplx(pair):
    return complex(pair[0], pair[1])


# --- 1 + 3. off-orbit points vs QAG_MINO ------------------------------------
for e in (0.3, 0.5):
    insp = Inspectre(spin=A, semilatus_rectum=P, eccentricity=e)
    r_c = P / (1 + 0.5 * e)
    points = {
        "rface": (P / (1 + e) - 1.0, math.pi / 2),
        "zface": (r_c, math.pi / 2 + 0.08),
        "int_near": (r_c, math.pi / 2 + 0.01),
    }
    for pname, (r_f, th_f) in points.items():
        xF = insp.es.make_coordinate(0.0, r_f, th_f, 0.0)
        s = ic.panel_nodes_build(insp._ctx, M, xF, insp._orbpar, A, P, e,
                                 insp.omega_phi, insp.omega_r,
                                 order=16, maxLevels=40, nMax=48)
        check(f"e={e} {pname} finite samples", s.nNonFinite == 0,
              f"nonfinite={s.nNonFinite}")
        for n in (0, 4, 48):
            php, dpp, srp = ic.panel_nodes_integrate(s, M, n, insp.omega_phi,
                                                     insp.omega_r)
            phq, dpq, srq = insp.integrate_nmode_fast(
                M, n, r_f, th_f, mode=ic.INSPECTRE_INTEG_QAG_MINO,
                epsabs=1e-13, epsrel=1e-13)
            for lab, vp, vq in (("PhiS", cplx(php), cplx(phq)),
                                ("src", cplx(srp), cplx(srq))):
                # relative where the amplitude is alive; where the spectrum
                # has decayed to the reference's own noise floor (QAG runs at
                # epsabs=1e-13) only absolute agreement is meaningful
                dabs = abs(vp - vq)
                rel = dabs / max(abs(vq), 1e-300)
                check(f"e={e} {pname} n={n} {lab}",
                      rel < 1e-6 or dabs < 1e-12,
                      f"rel={rel:.1e} abs={dabs:.1e}")
        ic.panel_nodes_free(s)
    del insp

# --- 2. crossing-point puncture self-convergence -----------------------------
e = 0.3
insp = Inspectre(spin=A, semilatus_rectum=P, eccentricity=e)
r_f = P / (1 + 0.5 * e)
xF = insp.es.make_coordinate(0.0, r_f, math.pi / 2, 0.0)


def phis_modes(order, lev):
    s = ic.panel_nodes_build(insp._ctx, M, xF, insp._orbpar, A, P, e,
                             insp.omega_phi, insp.omega_r,
                             order=order, maxLevels=lev, nMax=48)
    out = {n: cplx(ic.panel_nodes_integrate(s, M, n, insp.omega_phi,
                                            insp.omega_r)[0])
           for n in (0, 4, 48)}
    ic.panel_nodes_free(s)
    return out


coarse = phis_modes(16, 36)
fine = phis_modes(24, 44)
for n in (0, 4, 48):
    rel = abs(coarse[n] - fine[n]) / abs(fine[n])
    check(f"crossing PhiS n={n} self-convergence", rel < 1e-10,
          f"rel={rel:.1e}")

print(f"\n{failures} failure(s)")
sys.exit(1 if failures else 0)
