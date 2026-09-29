"""Split the n-mode error into t-quadrature and calc_m noise, over the worldtube.

Field points are taken from the highest AMR level of a SpECTRE run and span C1
from the near-particle limit to the worldtube corner. For each point the C
src_m(t) and the mpmath reference are evaluated on a uniform grid over one
radial period, and the n-mode of the true series and of the error series are
formed separately.

Run inside the spectre conda env:
    python test/eps_spectrum_along_orbit.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inspectre.inspectre import Inspectre
from mmode_reference import dps_for, modes

SPIN, P, ECC = 0.9, 10.0, 0.2
MS = (2, 6)
NS = (2, 8, 16)
NFINE = 512
SUBS = (64, 128, 256, 512)

FIELD = (("near   ", 8.33859, 6.780454e-05),
         ("d=3e-3 ", 9.30734, 3.222749e-04),
         ("d=3e-2 ", 8.68029, -3.455536e-03),
         ("C1 mid ", 11.28871, 7.680605e-03),
         ("C1 p90 ", 8.03688, 3.015853e-01),
         ("C1 max ", 19.96312, -4.209789e-01))


def switch_c1(m):
    """m -> C1 above which inspectre's auto dispatch leaves the ei_table branch."""
    return 0.5 / m ** 1.2


def series(insp, r_f, th_f, dth, ts):
    """(insp, field point, times) -> (C series, mpmath series, C1 series) per m."""
    c = {m: np.empty(len(ts), dtype=complex) for m in MS}
    g = {m: np.empty(len(ts), dtype=complex) for m in MS}
    c1 = np.empty(len(ts))
    for i, t in enumerate(ts):
        lam = insp.lambda_from_t(t)
        r_p = insp.r_from_lambda(lam)
        insp.set_particle(r_p, math.pi / 2, insp.phi_from_lambda(lam),
                          insp.four_velocity_equatorial(lam))
        ctx = insp.es._ef._ctx
        dr = r_f - r_p
        c1[i] = (ctx.alpha20 * dr * dr + ctx.alpha02 * dth * dth) / ctx.beta
        dps = min(300, max(dps_for(ctx, m, dr, dth, work=30) for m in MS))
        _, ref = modes(ctx, list(MS), dr, dth, dps=dps)
        for m in MS:
            out = insp.es.calc_m(m, r_f, th_f)
            c[m][i] = complex(out["src"][0], out["src"][1])
            g[m][i] = complex(ref[m])
    return c, g, c1


def main():
    insp = Inspectre(spin=SPIN, semilatus_rectum=P, eccentricity=ECC)
    Tr = insp.t_from_lambda(insp.mino_period_r)
    om_r, om_ph = insp.omega_r, insp.omega_phi
    ts = np.arange(NFINE) * (Tr / NFINE)
    print(f"a={SPIN} p={P} e={ECC}  Tr={Tr:.5f}  NFINE={NFINE}\n")

    for tag, r_f, cos_th in FIELD:
        th_f = math.acos(cos_th)
        dth = th_f - math.pi / 2
        c, g, c1 = series(insp, r_f, th_f, dth, ts)
        print(f"=== {tag} r={r_f:.5f} cos th={cos_th:+.4e} "
              f"C1 {c1.min():.2e}..{c1.max():.2e} ===", flush=True)
        for m in MS:
            eps = c[m] - g[m]
            above = 100.0 * np.count_nonzero(c1 > switch_c1(m)) / len(c1)
            rel = np.abs(eps) / np.abs(g[m])
            print(f"  m={m}: |eps/g| median {np.median(rel):.2e} max {rel.max():.2e}"
                  f"   {above:.0f}% of the orbit above the m={m} switch")

            def mode(s, N, n):
                sub = s[:: NFINE // N]
                tt = ts[:: NFINE // N]
                return np.sum(sub * np.exp(1j * (m * om_ph + n * om_r) * tt)) / N

            for n in NS:
                ref = mode(g[m], NFINE, n)
                row = [f"n={n:<3d} |src_mn|={abs(ref):.3e} "]
                for N in SUBS:
                    e = abs(mode(eps, N, n)) / abs(ref)
                    q = abs(mode(g[m], N, n) - ref) / abs(ref)
                    row.append(f"N{N}: eps {e:.1e} quad {q:.1e}")
                print("     " + "  ".join(row))
        print(flush=True)


if __name__ == "__main__":
    main()
