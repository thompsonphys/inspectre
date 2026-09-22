"""Two evaluators of the same m-mode puncture integral, split by separation.

The m-mode is
    PhiSb_m = int_0^2pi N(dphib) (alpha + beta sin^2(dphib/2))^(-7/2)
              exp(-i m dphib) d dphib
    N(dphib) = sum_j ReA[j] dQ^2j + sin(dphib) sum_j ImA[j] dQ^2j
    dQ = sin(dphib/2),  alpha = alpha20 dr^2 + alpha02 dtheta^2
and PhiS_m = PhiSb_m exp(-i m (phi_p + c dr)).

Branch "ei_table" is the shipped closed form: a degree-(m+6) polynomial in
C1 = alpha/beta whose coefficients are the ReEI/ImEI tables. It is the degree
recurrence for the kernel run upward m times, so it loses one digit per decade
of C1 per unit m and is unusable once C1 > ~11/m^2.

Branch "legendre" evaluates the kernel
    S_mu = int_0^2pi cos(mu dphib) (alpha + beta sin^2(dphib/2))^(-7/2) d dphib
        = Q^3_{mu-1/2}(1 + 2 C1) up to normalisation
directly. S_mu is the minimal solution of
    (mu - 5/2) S_{mu+1} - 2 mu z S_mu + (mu + 5/2) S_{mu-1} = 0,  z = 1 + 2 C1
so it is generated downward from mu = m + 5 + guard and normalised on S_0,
which comes from complete elliptic integrals. Then
    PhiSb_m = sum_{n=-5}^{5} N_n S_|m-n|
with N_n the Laurent coefficients of N in exp(i n dphib). No coefficient
table and no m ceiling, and the cancellation runs the other way: this branch
degrades as C1 -> 0, where "ei_table" is exact. Above the crossing both branches
are available for m <= 20 and only "legendre" for m > 20; below it, m > 20 has
no accurate evaluator.

The switch is C1 > 0.5 m^-1.2 for PhiS_m and 0.45 m^-1.1 for calc_m. Those are
median fits: over 549 bound-orbit cases (6 spins, both signs of dr and a
theta-dominated alpha) the measured crossing spans a factor of 3 to 20 at fixed
m. That spread costs nothing, because a threshold chosen per case with hindsight
gives the same worst error as the fitted law -- the residual is the width of the
window where neither branch is accurate, not the placement of the switch inside
it. Worst dispatched error over that matrix is 8.7e-8 for PhiS_m and 6.1e-4 for
src, both at m = 20 in that window; away from it "legendre" holds ~1e-14 while
"ei_table" diverges without bound.
"""

import math

import numpy as np
from scipy.special import ellipe, ellipk

from .mmode_laurent import numerators, src_numerator

BRANCHES = ("auto", "ei_table", "legendre")
EI_TABLE_MMAX = 20
SWITCH_C1_COEFF = 0.5
SWITCH_C1_SRC_COEFF = 0.45
SWITCH_C1_SRC_EXP = 1.1
SWITCH_C1_EXP = 1.2
GUARD_DIGITS = 40.0
GUARD_MIN = 20
GUARD_MAX = 300000

_LAURENT_DQ2 = {-1: -0.25, 0: 0.5, 1: -0.25}
_LAURENT_SIN = {-1: 0.5j, 1: -0.5j}


def blocks(ctx, dr, dtheta):
    """(ctx, dr, dtheta) -> (ReA[5], ImA[5]), the dQ^2j coefficients of N."""
    dr2 = dr * dr
    dr4 = dr2 * dr2
    dr6 = dr4 * dr2
    dr8 = dr4 * dr4
    dt2 = dtheta * dtheta
    dt4 = dt2 * dt2
    dt8 = dt4 * dt4
    k = ctx

    ReA = [0.0] * 5
    ImA = [0.0] * 5

    ReA[0] = ((k.A6000 + k.A7000 * dr) * dr6 + (k.A8000 + k.A9000 * dr) * dr8
              + (k.A4200 + dr * (k.A5200 + dr * (k.A6200 + k.A7200 * dr))) * dr4 * dt2
              + ((k.A2400 + k.A3400 * dr) * dr2 + (k.A4400 + k.A5400 * dr) * dr4
                 + (k.A0600 + dr * (k.A1600 + dr * (k.A2600 + k.A3600 * dr))) * dt2) * dt4
              + (k.A0800 + k.A1800 * dr) * dt8)
    ImA[0] = ((k.A6001 + k.A7001 * dr) * dr6 + k.A8001 * dr8
              + (k.A4201 + dr * (k.A5201 + k.A6201 * dr)) * dr4 * dt2
              + ((k.A2401 + k.A3401 * dr) * dr2 + k.A4401 * dr4
                 + (k.A0601 + dr * (k.A1601 + k.A2601 * dr)) * dt2) * dt4
              + k.A0801 * dt8)
    ReA[1] = ((k.A4020 + dr * (k.A5020 + dr * (k.A6020 + k.A7020 * dr))) * dr4
              + (k.A2220 + dr * (k.A3220 + dr * (k.A4220 + k.A5220 * dr))) * dr2 * dt2
              + (k.A0420 + k.A1420 * dr + (k.A2420 + k.A3420 * dr) * dr2
                 + (k.A0620 + k.A1620 * dr) * dt2) * dt4)
    ImA[1] = ((k.A4021 + dr * (k.A5021 + k.A6021 * dr)) * dr4
              + (k.A2221 + dr * (k.A3221 + k.A4221 * dr)) * dr2 * dt2
              + (k.A0421 + k.A1421 * dr + k.A2421 * dr2 + k.A0621 * dt2) * dt4)
    ReA[2] = ((k.A2040 + k.A3040 * dr) * dr2 + (k.A4040 + k.A5040 * dr) * dr4
              + (k.A0240 + dr * (k.A1240 + dr * (k.A2240 + k.A3240 * dr))) * dt2
              + (k.A0440 + k.A1440 * dr) * dt4)
    ImA[2] = ((k.A2041 + k.A3041 * dr) * dr2 + k.A4041 * dr4
              + (k.A0241 + dr * (k.A1241 + k.A2241 * dr)) * dt2 + k.A0441 * dt4)
    ReA[3] = (k.A0060 + k.A1060 * dr + (k.A2060 + k.A3060 * dr) * dr2
              + (k.A0260 + k.A1260 * dr) * dt2)
    ImA[3] = k.A0061 + k.A1061 * dr + k.A2061 * dr2 + k.A0261 * dt2
    ReA[4] = k.A0080 + k.A1080 * dr
    ImA[4] = k.A0081
    return ReA, ImA


def numerator_laurent(ReA, ImA):
    """(ReA[5], ImA[5]) -> {n: N_n} complex coefficients of N in exp(i n dphib)."""
    def mul(p, q):
        out = {}
        for a, x in p.items():
            for b, y in q.items():
                out[a + b] = out.get(a + b, 0.0 + 0.0j) + x * y
        return out

    pows = [{0: 1.0 + 0.0j}]
    for _ in range(4):
        pows.append(mul(pows[-1], _LAURENT_DQ2))

    out = {}
    for j in range(5):
        for n, v in pows[j].items():
            out[n] = out.get(n, 0.0 + 0.0j) + ReA[j] * v
        for n, v in mul(pows[j], _LAURENT_SIN).items():
            out[n] = out.get(n, 0.0 + 0.0j) + ImA[j] * v
    return out


def kernel_seed(alpha, beta, q=3):
    """(alpha, beta, q) -> S_0 = int_0^2pi (alpha + beta sin^2(t/2))^(-q-1/2) dt.

    Built from I_-1/2 = 4 sqrt(alpha+beta) E(k) and I_1/2 = 4 K(k)/sqrt(alpha+beta),
    k^2 = beta/(alpha+beta), raised by p (A^2-B^2) I_{p+1} = (2p-1) A I_p - (p-1) I_{p-1}.
    """
    A = alpha + 0.5 * beta
    ab = alpha + beta
    k2 = beta / ab
    D = alpha * ab
    I_prev = 4.0 * math.sqrt(ab) * float(ellipe(k2))
    I_cur = 4.0 * float(ellipk(k2)) / math.sqrt(ab)
    p = 0.5
    while p < q:
        I_prev, I_cur = I_cur, ((2.0 * p - 1.0) * A * I_cur - (p - 1.0) * I_prev) / (p * D)
        p += 1.0
    return I_cur


def kernel_guard(C1):
    """C1 -> number of extra downward steps needed for a converged minimal solution."""
    z = 1.0 + 2.0 * C1
    ratio = z - math.sqrt(z * z - 1.0)
    decay = -math.log(ratio) if ratio < 1.0 else float("inf")
    if decay <= 0.0 or not math.isfinite(decay):
        return GUARD_MAX
    return int(min(GUARD_MAX, max(GUARD_MIN, GUARD_DIGITS / decay)))


def kernel(mumax, alpha, beta, guard=None, q=3):
    """(mumax, alpha, beta, q) -> S_0..S_mumax by downward recurrence, normalised on S_0."""
    C1 = alpha / beta
    z = 1.0 + 2.0 * C1
    g = kernel_guard(C1) if guard is None else guard
    lo, hi = q - 0.5, q - 0.5
    top = mumax + g
    s = np.zeros(top + 2)
    s[top] = 1e-300
    for mu in range(top, 0, -1):
        s[mu - 1] = (2.0 * mu * z * s[mu] - (mu - lo) * s[mu + 1]) / (mu + hi)
        if abs(s[mu - 1]) > 1e250:
            s[:mu + 2] *= 1e-250
    return s[:mumax + 1] * (kernel_seed(alpha, beta, q) / s[0])


def switch_c1(m):
    """m -> C1 above which the legendre branch is used."""
    return SWITCH_C1_COEFF / max(m, 1) ** SWITCH_C1_EXP


def choose_branch(ctx, m, dr, dtheta):
    """(ctx, m, dr, dtheta) -> "ei_table" or "legendre"."""
    if m > EI_TABLE_MMAX:
        return "legendre"
    alpha = ctx.alpha20 * dr * dr + ctx.alpha02 * dtheta * dtheta
    return "legendre" if alpha / ctx.beta > switch_c1(m) else "ei_table"


def phi_s_m_legendre(ctx, m, dr, dtheta):
    """(ctx, m, dr, dtheta) -> (Re, Im) m-mode puncture by the kernel recurrence."""
    alpha = ctx.alpha20 * dr * dr + ctx.alpha02 * dtheta * dtheta
    beta = ctx.beta
    N = numerator_laurent(*blocks(ctx, dr, dtheta))
    S = kernel(m + 5, alpha, beta)
    phiSb = sum(N[n] * S[abs(m - n)] for n in N)
    rot = complex(math.cos(m * (ctx.xp.phi + ctx.c * dr)),
                  -math.sin(m * (ctx.xp.phi + ctx.c * dr)))
    z = phiSb * rot
    return z.real, z.imag


def calc_m_legendre(ctx, m, dr, dtheta):
    """(ctx, m, dr, dtheta) -> (PhiS[2], dPhiS[8], d2PhiS[20], src[2]) by recurrence.

    Mixed second derivatives are NAN, matching effsource_equatorial_ctx_calc_m.
    """
    num, _ = numerators(ctx, dr, dtheta)
    sn, _ = src_numerator(ctx, dr, dtheta)
    alpha = ctx.alpha20 * dr * dr + ctx.alpha02 * dtheta * dtheta
    polys = dict(num)
    polys["src"] = sn
    w_polys = {k: ({i // 2: v for i, v in poly.items() if i % 2 == 0} or {0: 0j})
               for k, poly in polys.items()}
    deg = max(max(abs(n) for n in w) for w in w_polys.values())
    S = kernel(m + deg, alpha, ctx.beta, q=5)
    arg = m * (ctx.xp.phi + ctx.c * dr)
    rot = complex(math.cos(arg), -math.sin(arg))

    def mode(key):
        acc = 0j
        for n, t in w_polys[key].items():
            acc += t * S[abs(m - n)]
        return acc * rot

    PhiS = mode("PhiS")
    d1 = [mode(k) for k in ("dt", "dr", "dth", "dph")]
    src = mode("src")
    d2 = [float("nan")] * 20
    for slot, key in ((0, "dt2"), (3, "dtph"), (4, "dr2"), (7, "dth2"), (9, "dph2")):
        v = mode(key)
        d2[2 * slot], d2[2 * slot + 1] = v.real, v.imag
    dPhiS = []
    for v in d1:
        dPhiS += [v.real, v.imag]
    return ([PhiS.real, PhiS.imag], dPhiS, d2, [src.real, src.imag])


def calc_m_ei_table(ctx_obj, m, dr, dtheta):
    """(EffsourceEquatorialContext, m, dr, dtheta) -> the shipped calc_m_offset."""
    return ctx_obj.calc_m_offset(m, dr, dtheta)


def switch_c1_src(m):
    """m -> C1 above which the legendre branch is used for calc_m / src."""
    return SWITCH_C1_SRC_COEFF / max(m, 1) ** SWITCH_C1_SRC_EXP


def choose_branch_src(ctx, m, dr, dtheta):
    """(ctx, m, dr, dtheta) -> "ei_table" or "legendre" for calc_m."""
    if m > EI_TABLE_MMAX:
        return "legendre"
    alpha = ctx.alpha20 * dr * dr + ctx.alpha02 * dtheta * dtheta
    return "legendre" if alpha / ctx.beta > switch_c1_src(m) else "ei_table"


def calc_m(ctx_obj, m, dr, dtheta, branch="auto"):
    """(EffsourceEquatorialContext, m, dr, dtheta, branch) -> (PhiS, dPhiS, d2PhiS, src, branch)."""
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}, got {branch!r}")
    ctx = ctx_obj._ctx
    used = choose_branch_src(ctx, m, dr, dtheta) if branch == "auto" else branch
    if used == "ei_table":
        if m > EI_TABLE_MMAX:
            raise ValueError(f"branch='ei_table' supports m <= {EI_TABLE_MMAX}, got {m}")
        out = calc_m_ei_table(ctx_obj, m, dr, dtheta)
    else:
        out = calc_m_legendre(ctx, m, dr, dtheta)
    return out + (used,)


def phi_s_m_ei_table(ctx_obj, m, dr, dtheta):
    """(EffsourceEquatorialContext, m, dr, dtheta) -> (Re, Im) by the shipped closed form."""
    return ctx_obj.calc_PhiS_m_offset(m, dr, dtheta)


def phi_s_m(ctx_obj, m, dr, dtheta, branch="auto"):
    """(EffsourceEquatorialContext, m, dr, dtheta, branch) -> (Re, Im, branch used)."""
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}, got {branch!r}")
    ctx = ctx_obj._ctx
    used = choose_branch(ctx, m, dr, dtheta) if branch == "auto" else branch
    if used == "ei_table":
        if m > EI_TABLE_MMAX:
            raise ValueError(f"branch='ei_table' supports m <= {EI_TABLE_MMAX}, got {m}")
        re, im = phi_s_m_ei_table(ctx_obj, m, dr, dtheta)
    else:
        re, im = phi_s_m_legendre(ctx, m, dr, dtheta)
    return re, im, used
