"""Grade the C calc_m against the mpmath reference along the a=0.9 p=10 e=0.2 orbit."""
import math
import os
import sys

import mpmath as mp
import numpy as np

HERE = os.path.expanduser("~/projects/sf/inspectre")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "test"))

from inspectre.inspectre import Inspectre
from mmode_reference import dps_for, modes

SPIN, P, ECC = 0.9, 10.0, 0.2
FIELD = [(8.40733, 6.78045e-05, 5.70e-04),
         (9.30734, 3.22275e-04, 3.00e-03),
         (11.00516, -9.08758e-04, 1.00e-02),
         (12.48385, 8.01130e-03, 1.00e-01)]
MS = (2, 6)


def c_src(insp, m, r_f, th_f):
    out = insp.es.calc_m(m, r_f, th_f)
    return complex(out["src"][0], out["src"][1])


def ref_src(insp, m, dr, dth):
    ctx = insp.es._ef._ctx
    dps = min(400, dps_for(ctx, m, dr, dth, work=30))
    _, src = modes(ctx, [m], dr, dth, dps=dps)
    return complex(src[m]), dps


def main():
    insp = Inspectre(spin=SPIN, semilatus_rectum=P, eccentricity=ECC)
    Tr = insp.t_from_lambda(insp.mino_period_r)
    print(f"a={SPIN} p={P} e={ECC}   Tr = {Tr:.6f}\n")

    for r_f, cos_th, dtag in FIELD:
        th_f = math.acos(cos_th)
        dth = th_f - math.pi / 2

        scan = np.linspace(0.0, Tr, 257)[:-1]
        rs = np.array([insp.r_from_lambda(insp.lambda_from_t(t)) for t in scan])
        t0 = scan[int(np.argmin(np.abs(rs - r_f)))]
        dr_min = float(np.min(np.abs(rs - r_f)))

        width = Tr / 256.0
        near = t0 + np.linspace(-2.0 * width, 2.0 * width, 9)
        far = np.linspace(0.0, Tr, 9)[:-1]
        times = np.concatenate([near, far])

        print(f"=== field point r={r_f:.5f} dtheta={dth:+.3e} (d~{dtag:.0e}) ===")
        print(f"    closest approach t={t0:.4f}, min|dr| over scan = {dr_min:.3e}")
        print(f"    {'t':>10} {'dr':>11} {'m':>3} {'|src| C':>12} "
              f"{'rel err':>11} {'dps':>5} {'zone':>5}")
        worst = {m: 0.0 for m in MS}
        worst_near = {m: 0.0 for m in MS}
        for t in times:
            lam = insp.lambda_from_t(t)
            r_p = insp.r_from_lambda(lam)
            phi_p = insp.phi_from_lambda(lam)
            ur = insp.four_velocity_equatorial(lam)
            insp.set_particle(r_p, math.pi / 2, phi_p, ur)
            dr = r_f - r_p
            is_near = abs(t - t0) <= 2.0 * width + 1e-12
            for m in MS:
                c = c_src(insp, m, r_f, th_f)
                g, dps = ref_src(insp, m, dr, dth)
                rel = abs(c - g) / abs(g) if abs(g) > 0 else float("nan")
                worst[m] = max(worst[m], rel)
                if is_near:
                    worst_near[m] = max(worst_near[m], rel)
                print(f"    {t:10.4f} {dr:+11.3e} {m:>3} {abs(c):12.4e} "
                      f"{rel:11.3e} {dps:>5} {'PEAK' if is_near else 'far':>5}")
        for m in MS:
            print(f"    -> m={m}: worst over orbit {worst[m]:.3e}, "
                  f"worst in peak window {worst_near[m]:.3e}")
        print()


if __name__ == "__main__":
    main()
