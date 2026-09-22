"""High-precision m-mode reference for grading the two branches.

Both quantities are (trigonometric polynomial)/s2^(11/2), so the reference is
the same sum the "legendre" branch computes, carried in mpmath: coefficients
from the ctx A####, dAdt#### and d2Adt2#### sets, kernel from the closed-form
minimal solution of the same recurrence. Only the precision differs, which is
the calc_m_extended pattern. kernel_quad integrates the kernel independently
and agrees with kernel_mp to 1e-38 at dps 40 for alpha down to 1e-10.

The contraction sum(T_n S_|m-n|) cancels: PhiS loses about 10 decimal digits
per decade of alpha and src about 8, so dps must be set from cancellation()
rather than left at a fixed value.
"""
import mpmath as mp

from inspectre.mmode_laurent import numerators, src_numerator


class MP:
    """mpmath numeric context for the Laurent construction."""

    mpc = staticmethod(mp.mpc)
    mpf = staticmethod(mp.mpf)
    sin = staticmethod(mp.sin)
    cos = staticmethod(mp.cos)


def _seed(alpha, beta, q):
    """(alpha, beta, q) -> I_{q+1/2}, the mu = 0 kernel value, by the elliptic ladder."""
    ab = alpha + beta
    A = alpha + beta / 2
    D = alpha * ab
    I_prev = 4 * mp.sqrt(ab) * mp.ellipe(beta / ab)
    I_cur = 4 * mp.ellipk(beta / ab) / mp.sqrt(ab)
    p = mp.mpf(1) / 2
    while p < q:
        I_prev, I_cur = I_cur, ((2 * p - 1) * A * I_cur - (p - 1) * I_prev) / (p * D)
        p += 1
    return I_cur


GUARD_MAX = 20000


def kernel_miller(mumax, alpha, beta, q, guard):
    """(mumax, alpha, beta, q, guard) -> S_0..S_mumax by downward recurrence."""
    z = 1 + 2 * alpha / beta
    top = mumax + guard
    s = [mp.mpf(0)] * (top + 2)
    s[top] = mp.mpf(1)
    for mu in range(top, 0, -1):
        s[mu - 1] = (2 * mu * z * s[mu] - (mu - q + mp.mpf(1) / 2) * s[mu + 1]) \
            / (mu + q - mp.mpf(1) / 2)
    scale = _seed(alpha, beta, q) / s[0]
    return [v * scale for v in s[:mumax + 1]]


def kernel_legendre(mumax, alpha, beta, q):
    """(mumax, alpha, beta, q) -> S_0..S_mumax from Q^q_{mu-1/2}(1 + 2 alpha/beta)."""
    z = 1 + 2 * alpha / beta
    Q = [mp.legenq(mu - mp.mpf(1) / 2, q, z, type=3) for mu in range(mumax + 1)]
    scale = _seed(alpha, beta, q) / Q[0]
    return [v * scale for v in Q]


def kernel_guard(alpha, beta):
    """(alpha, beta) -> downward-recurrence terms needed at the working precision."""
    z = 1 + 2 * alpha / beta
    ratio = z - mp.sqrt(z * z - 1)
    return int(max(40, (mp.mp.dps + 15) * mp.log(10) / (-mp.log(ratio))))


def kernel_mp(mumax, alpha, beta, q=5):
    """(mumax, alpha, beta, q) -> S_0..S_mumax, by recurrence when its guard is short.

    The two routes agree to 1e-61 at dps 60; the recurrence is 3 to 70 times
    faster while the guard is short and unusable once alpha makes it long.
    """
    guard = kernel_guard(alpha, beta)
    if guard <= GUARD_MAX:
        return kernel_miller(mumax, alpha, beta, q, guard)
    return kernel_legendre(mumax, alpha, beta, q)


def kernel_quad(mu, alpha, beta, q=5):
    """(mu, alpha, beta, q) -> S_mu by quadrature under sin(psi/2) = sqrt(alpha/beta) sinh u."""
    w = mp.sqrt(alpha / beta)
    umax = mp.asinh(1 / w)

    def f(u):
        s = w * mp.sinh(u)
        return mp.cos(mu * 2 * mp.asin(s)) / (mp.cosh(u) ** (2 * q) * mp.sqrt(1 - s * s))

    pts = [mp.mpf(0)]
    for k in range(1, mu + 1):
        t = mp.sin(mp.pi * k / (2 * mu)) / w
        if t < mp.sinh(umax):
            pts.append(mp.asinh(t))
    pts.append(umax)
    return 4 * w / alpha ** (q + mp.mpf(1) / 2) * mp.quad(f, pts, method='tanh-sinh')


def cancellation(ctx, m, dr, dtheta, dps=120):
    """(ctx, m, dr, dtheta, dps) -> (PhiS, src) |sum| / sum |term| in the mode contraction."""
    mp.mp.dps = dps
    num, _ = numerators(ctx, dr, dtheta, MP)
    sn, _ = src_numerator(ctx, dr, dtheta, MP)
    alpha = mp.mpf(ctx.alpha20) * mp.mpf(dr) ** 2 \
        + mp.mpf(ctx.alpha02) * mp.mpf(dtheta) ** 2
    beta = mp.mpf(ctx.beta)
    out = []
    for poly in (num["PhiS"], sn):
        T = {i // 2: v for i, v in poly.items() if i % 2 == 0} or {0: mp.mpc(0)}
        S = kernel_mp(m + max(abs(n) for n in T), alpha, beta)
        terms = [t * S[abs(m - n)] for n, t in T.items()]
        out.append(abs(sum(terms)) / sum(abs(t) for t in terms))
    return out[0], out[1]


def dps_for(ctx, m, dr, dtheta, work=30):
    """(ctx, m, dr, dtheta, work) -> dps that leaves work digits after the contraction cancels."""
    p, s = cancellation(ctx, m, dr, dtheta)
    return int(max(-mp.log10(p), -mp.log10(s))) + work


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
