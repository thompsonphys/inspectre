"""High-precision m-mode reference for grading the two branches.

Both quantities are (trigonometric polynomial)/s2^(11/2), so the reference is
the same sum the "legendre" branch computes, carried in mpmath: coefficients
from the ctx A####, dAdt#### and d2Adt2#### sets, kernel by the same downward
recurrence with elliptic seeds. Only the precision differs, which is the
calc_m_extended pattern. Checked against independent mpmath quadrature of the
integral to 1e-48 or better across spins, separations and offset geometries.
"""
import mpmath as mp

from inspectre.mmode_laurent import numerators, src_numerator


class MP:
    """mpmath numeric context for the Laurent construction."""

    mpc = staticmethod(mp.mpc)
    mpf = staticmethod(mp.mpf)
    sin = staticmethod(mp.sin)
    cos = staticmethod(mp.cos)


def kernel_mp(mumax, alpha, beta, q=5):
    """(mumax, alpha, beta, q) -> S_0..S_mumax in mpmath by downward recurrence."""
    ab = alpha + beta
    A = alpha + beta / 2
    D = alpha * ab
    I_prev = 4 * mp.sqrt(ab) * mp.ellipe(beta / ab)
    I_cur = 4 * mp.ellipk(beta / ab) / mp.sqrt(ab)
    p = mp.mpf(1) / 2
    while p < q:
        I_prev, I_cur = I_cur, ((2 * p - 1) * A * I_cur - (p - 1) * I_prev) / (p * D)
        p += 1
    z = 1 + 2 * alpha / beta
    ratio = z - mp.sqrt(z * z - 1)
    guard = int(min(200000, max(40, (mp.mp.dps + 15) * mp.log(10) / (-mp.log(ratio)))))
    top = mumax + guard
    s = [mp.mpf(0)] * (top + 2)
    s[top] = mp.mpf(1)
    for mu in range(top, 0, -1):
        s[mu - 1] = (2 * mu * z * s[mu] - (mu - q + mp.mpf(1) / 2) * s[mu + 1]) \
            / (mu + q - mp.mpf(1) / 2)
    scale = I_cur / s[0]
    return [v * scale for v in s[:mumax + 1]]


def modes(ctx, ms, dr, dtheta, dps=60):
    """(ctx, m list, dr, dtheta, dps) -> ({m: PhiS_m}, {m: src_m}) at that precision."""
    mp.mp.dps = dps
    num, _ = numerators(ctx, dr, dtheta, MP)
    sn, _ = src_numerator(ctx, dr, dtheta, MP)
    alpha = mp.mpf(ctx.alpha20) * mp.mpf(dr) ** 2 \
        + mp.mpf(ctx.alpha02) * mp.mpf(dtheta) ** 2
    beta = mp.mpf(ctx.beta)
    rot = {m: mp.e ** (-1j * m * (mp.mpf(ctx.xp.phi) + mp.mpf(ctx.c) * mp.mpf(dr)))
           for m in ms}
    out = []
    for poly in (num["PhiS"], sn):
        T = {i // 2: v for i, v in poly.items() if i % 2 == 0} or {0: mp.mpc(0)}
        S = kernel_mp(max(ms) + max(abs(n) for n in T), alpha, beta)
        out.append({m: sum(t * S[abs(m - n)] for n, t in T.items()) * rot[m] for m in ms})
    return out[0], out[1]
