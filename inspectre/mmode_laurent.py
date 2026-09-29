"""Laurent representation of the puncture and effective source in u = exp(i dphib/2).

Every component of (PhiS, dPhiS, d2PhiS, src) that
effsource_equatorial_ctx_calc produces has the form

    X = T_X(dphib) / s2^p,   s2 = alpha + beta sin^2(dphib/2),   p in {7/2, 9/2, 11/2}

with T_X a trigonometric polynomial, and is a Laurent polynomial in u carrying
even powers alone. Over the common denominator s2^(11/2),

    X_m = sum_n T_n S^(11/2)_|m-n| * exp(-i m (phi_p + c dr))

with T_n the coefficient of u^(2n) and S^(11/2) the q = 5 kernel family.

A and its Q, R derivatives follow from the two blocks

    A = sum_j ReA[j] dQ^2j + dR sum_j ImA[j] dQ^2j

carried through second-order jets in (dr, dtheta) for the r and theta
derivatives, and over the ctx dAdt#### and d2Adt2#### coefficient sets for the t
derivatives.
"""

import math

import numpy as np


class Std:
    """Double-precision numeric context."""

    mpc = complex
    mpf = float
    sin = staticmethod(math.sin)
    cos = staticmethod(math.cos)


class Ld:
    """Long-double numeric context, 64-bit mantissa on x86."""

    mpc = staticmethod(lambda re, im=0.0:
                       np.clongdouble(re) + np.clongdouble(im) * np.clongdouble(1j))
    mpf = staticmethod(np.longdouble)
    sin = staticmethod(np.sin)
    cos = staticmethod(np.cos)


def bases(N):
    """numeric context -> {name: Laurent} for dQ, cos(dphib/2), dR, cos(dphib)."""
    half_i = N.mpc(0, 1) / 2
    half = N.mpf(1) / 2
    quarter = N.mpf(1) / 4
    dQ = {1: -half_i, -1: half_i}
    cQ = {1: N.mpc(half), -1: N.mpc(half)}
    dR = {2: -half_i, -2: half_i}
    cR = {2: N.mpc(half), -2: N.mpc(half)}
    return {"dQ": dQ, "cQ": cQ, "dR": dR, "cR": cR,
            "dQ2": lmul(dQ, dQ), "one": {0: N.mpc(1)},
            "quarter": quarter, "half": half}


class Jet:
    """Value and first and second partials with respect to (dr, dtheta)."""

    __slots__ = ("v", "r", "t", "rr", "rt", "tt")

    def __init__(self, v, r=0.0, t=0.0, rr=0.0, rt=0.0, tt=0.0):
        self.v, self.r, self.t, self.rr, self.rt, self.tt = v, r, t, rr, rt, tt

    def __add__(self, o):
        if not isinstance(o, Jet):
            return Jet(self.v + o, self.r, self.t, self.rr, self.rt, self.tt)
        return Jet(self.v + o.v, self.r + o.r, self.t + o.t,
                   self.rr + o.rr, self.rt + o.rt, self.tt + o.tt)

    __radd__ = __add__

    def __mul__(self, o):
        if not isinstance(o, Jet):
            return Jet(self.v * o, self.r * o, self.t * o,
                       self.rr * o, self.rt * o, self.tt * o)
        return Jet(self.v * o.v,
                   self.r * o.v + self.v * o.r,
                   self.t * o.v + self.v * o.t,
                   self.rr * o.v + 2.0 * self.r * o.r + self.v * o.rr,
                   self.rt * o.v + self.r * o.t + self.t * o.r + self.v * o.rt,
                   self.tt * o.v + 2.0 * self.t * o.t + self.v * o.tt)

    __rmul__ = __mul__


class Coeffs:
    """ctx view that answers A#### with the coefficient set named by prefix."""

    def __init__(self, ctx, prefix=""):
        self._ctx, self._prefix = ctx, prefix

    def __getattr__(self, name):
        return getattr(self._ctx, self._prefix + name[1:] if self._prefix else name)


def lmul(a, b):
    """(Laurent, Laurent) -> product."""
    out = {}
    for i, x in a.items():
        if x == 0:
            continue
        for j, y in b.items():
            k = i + j
            out[k] = out[k] + x * y if k in out else x * y
    return out


def ladd(*ps):
    """(Laurent, ...) -> sum."""
    out = {}
    for p in ps:
        for i, x in p.items():
            out[i] = out[i] + x if i in out else x
    return out


def lscale(a, s):
    """(Laurent, scalar) -> scaled copy."""
    return {i: x * s for i, x in a.items()}


def lpow(a, n, one):
    """(Laurent, n >= 0, unit Laurent) -> a**n."""
    out = dict(one)
    for _ in range(n):
        out = lmul(out, a)
    return out


def leval(a, dphib):
    """(Laurent, dphib) -> value at that dphib."""
    return sum(x * complex(math.cos(0.5 * i * dphib), math.sin(0.5 * i * dphib))
               for i, x in a.items())


def _blocks(k, dr, dtheta):
    """(coefficient view, dr, dtheta) -> (ReA[5], ImA[5]); dr/dtheta may be Jets."""
    dr2 = dr * dr
    dr4 = dr2 * dr2
    dr6 = dr4 * dr2
    dr8 = dr4 * dr4
    dt2 = dtheta * dtheta
    dt4 = dt2 * dt2
    dt8 = dt4 * dt4

    ReA = [None] * 5
    ImA = [None] * 5
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


def _assemble(B, P, Q, order):
    """(bases, ReA[5], ImA[5], 0|1|2) -> Laurent of A, dA_dQ or d2A_dQ2."""
    out = {}
    for j in range(5):
        e = 2 * j - order
        if e < 0:
            continue
        w = 1
        for s in range(order):
            w *= 2 * j - s
        if w == 0:
            continue
        dqe = lpow(B["dQ"], e, B["one"])
        block = ladd(lscale(dqe, P[j]), lmul(B["dR"], lscale(dqe, Q[j])))
        out = ladd(out, lscale(block, w))
    return out


def _assemble_R(B, Q, order):
    """(bases, ImA[5], 0|1) -> Laurent of dA_dR or d2A_dQR."""
    out = {}
    for j in range(5):
        e = 2 * j - order
        if e < 0:
            continue
        w = 2 * j if order else 1
        if w == 0:
            continue
        out = ladd(out, lscale(lpow(B["dQ"], e, B["one"]), Q[j] * w))
    return out


def numerators(ctx, dr, dtheta, N=Std):
    """(ctx, dr, dtheta) -> (dict of Laurent numerators over s2^(11/2), s2 Laurent).

    Keys: "PhiS", "dt", "dr", "dth", "dph", "dt2", "dr2", "dth2", "dph2", "dtph".
    """
    B = bases(N)
    Z, U = N.mpf(0), N.mpf(1)
    dr, dtheta = N.mpf(dr), N.mpf(dtheta)
    drj = Jet(dr, U, Z, Z, Z, Z)
    dtj = Jet(dtheta, Z, U, Z, Z, Z)
    P0, Q0 = _blocks(Coeffs(ctx), drj, dtj)
    Pt, Qt = _blocks(Coeffs(ctx, "dAdt"), drj, dtj)
    Ptt, Qtt = _blocks(Coeffs(ctx, "d2Adt2"), drj, dtj)

    def pick(B, attr):
        return [getattr(b, attr) if isinstance(b, Jet)
                else (N.mpf(b) if attr == "v" else Z) for b in B]

    A = _assemble(B, pick(P0, "v"), pick(Q0, "v"), 0)
    dA_dQ = _assemble(B, pick(P0, "v"), pick(Q0, "v"), 1)
    dA_dR = _assemble_R(B, pick(Q0, "v"), 0)
    d2A_dQ2 = _assemble(B, pick(P0, "v"), pick(Q0, "v"), 2)
    d2A_dQR = _assemble_R(B, pick(Q0, "v"), 1)
    dA_dr0 = _assemble(B, pick(P0, "r"), pick(Q0, "r"), 0)
    dA_dth = _assemble(B, pick(P0, "t"), pick(Q0, "t"), 0)
    d2A_dr2_0 = _assemble(B, pick(P0, "rr"), pick(Q0, "rr"), 0)
    d2A_dth2 = _assemble(B, pick(P0, "tt"), pick(Q0, "tt"), 0)
    d2A_dQr = _assemble(B, pick(P0, "r"), pick(Q0, "r"), 1)
    d2A_dRr = _assemble_R(B, pick(Q0, "r"), 0)
    dA_dt0 = _assemble(B, pick(Pt, "v"), pick(Qt, "v"), 0)
    d2A_dtQ = _assemble(B, pick(Pt, "v"), pick(Qt, "v"), 1)
    d2A_dtR = _assemble_R(B, pick(Qt, "v"), 0)
    d2A_dtr = _assemble(B, pick(Pt, "r"), pick(Qt, "r"), 0)
    d2A_dt2_0 = _assemble(B, pick(Ptt, "v"), pick(Qtt, "v"), 0)

    c, rt, rtt = N.mpf(ctx.c), N.mpf(ctx.rt), N.mpf(ctx.rtt)
    phit, phitt = N.mpf(ctx.phit), N.mpf(ctx.phitt)
    dcdt, d2cdt2 = N.mpf(ctx.dcdt), N.mpf(ctx.d2cdt2)
    V = dr * dcdt + phit - c * rt
    W = -2.0 * dcdt * rt + dr * d2cdt2 + phitt - c * rtt

    dQ_dph = lscale(B["cQ"], B["half"])
    dQ_dr = lscale(dQ_dph, -c)
    dQ_dt = lscale(dQ_dph, -V)
    dR_dph = dict(B["cR"])
    dR_dr = lscale(dR_dph, -c)
    dR_dt = lscale(dR_dph, -V)
    d2Q_dph2 = lscale(B["dQ"], -B["quarter"])
    d2Q_dr2 = lscale(d2Q_dph2, c * c)
    d2Q_dt2 = ladd(lscale(d2Q_dph2, V * V), lscale(dQ_dph, -W))
    d2Q_dtph = lscale(d2Q_dph2, -V)
    d2R_dph2 = lscale(B["dR"], -U)
    d2R_dr2 = lscale(d2R_dph2, c * c)
    d2R_dt2 = ladd(lscale(d2R_dph2, V * V), lscale(dR_dph, -W))
    d2R_dtph = lscale(d2R_dph2, -V)

    dA_dt = ladd(dA_dt0, lmul(dA_dQ, dQ_dt), lmul(dA_dR, dR_dt), lscale(dA_dr0, -rt))
    dA_dr = ladd(dA_dr0, lmul(dA_dQ, dQ_dr), lmul(dA_dR, dR_dr))
    dA_dph = ladd(lmul(dA_dQ, dQ_dph), lmul(dA_dR, dR_dph))
    d2A_dph2 = ladd(lmul(dA_dR, d2R_dph2), lmul(dA_dQ, d2Q_dph2),
                    lscale(lmul(lmul(dQ_dph, dR_dph), d2A_dQR), 2.0),
                    lmul(lmul(dQ_dph, dQ_dph), d2A_dQ2))
    d2A_dr2 = ladd(d2A_dr2_0, lmul(d2Q_dr2, dA_dQ), lmul(d2R_dr2, dA_dR),
                   lscale(lmul(d2A_dQr, dQ_dr), 2.0),
                   lmul(d2A_dQ2, lmul(dQ_dr, dQ_dr)),
                   lscale(lmul(ladd(d2A_dRr, lmul(d2A_dQR, dQ_dr)), dR_dr), 2.0))
    d2A_dtph = ladd(lmul(dA_dR, d2R_dtph), lmul(dA_dQ, d2Q_dtph),
                    lmul(d2A_dQR, lmul(dR_dph, dQ_dt)),
                    lmul(d2A_dQR, lmul(dQ_dph, dR_dt)),
                    lmul(d2A_dQ2, lmul(dQ_dph, dQ_dt)),
                    lscale(lmul(d2A_dRr, dR_dph), -rt),
                    lscale(lmul(d2A_dQr, dQ_dph), -rt),
                    lmul(d2A_dtR, dR_dph), lmul(d2A_dtQ, dQ_dph))
    d2A_dt2 = ladd(d2A_dt2_0, lmul(dA_dR, d2R_dt2), lmul(dA_dQ, d2Q_dt2),
                   lscale(lmul(d2A_dQR, lmul(dQ_dt, dR_dt)), 2.0),
                   lmul(d2A_dQ2, lmul(dQ_dt, dQ_dt)),
                   lscale(dA_dr0, -rtt),
                   lscale(lmul(d2A_dRr, dR_dt), -2.0 * rt),
                   lscale(lmul(d2A_dQr, dQ_dt), -2.0 * rt),
                   lscale(d2A_dr2_0, rt * rt),
                   lscale(lmul(d2A_dtR, dR_dt), 2.0),
                   lscale(lmul(d2A_dtQ, dQ_dt), 2.0),
                   lscale(d2A_dtr, -2.0 * rt))

    a20, a02, beta = N.mpf(ctx.alpha20), N.mpf(ctx.alpha02), N.mpf(ctx.beta)
    alpha = a20 * dr * dr + a02 * dtheta * dtheta
    s2 = ladd({0: N.mpc(alpha)}, lscale(B["dQ2"], beta))
    ds2_dr0 = 2 * a20 * dr
    ds2_dth = 2 * a02 * dtheta
    ds2_dQ = lscale(B["dQ"], 2 * beta)
    ds2_dt0 = ladd({0: N.mpc(N.mpf(ctx.dalphadt20) * dr * dr
                            + N.mpf(ctx.dalphadt02) * dtheta * dtheta)},
                   lscale(B["dQ2"], N.mpf(ctx.dbetadt)))
    d2s2_dr2_0 = 2 * a20
    d2s2_dth2 = 2 * a02
    d2s2_dQ2 = 2 * beta
    d2s2_dtQ = lscale(B["dQ"], 2 * N.mpf(ctx.dbetadt))
    d2s2_dtr = 2 * N.mpf(ctx.dalphadt20) * dr
    d2s2_dt2_0 = ladd(
        {0: N.mpc(N.mpf(ctx.d2alphadt220) * dr * dr
                  + N.mpf(ctx.d2alphadt202) * dtheta * dtheta)},
        lscale(B["dQ2"], N.mpf(ctx.d2betadt2)))

    ds2_dt = ladd(ds2_dt0, lmul(ds2_dQ, dQ_dt), {0: N.mpc(-ds2_dr0 * rt)})
    ds2_dr = ladd({0: N.mpc(ds2_dr0)}, lmul(ds2_dQ, dQ_dr))
    ds2_dph = lmul(ds2_dQ, dQ_dph)
    d2s2_dph2 = ladd(lmul(ds2_dQ, d2Q_dph2),
                     lscale(lmul(dQ_dph, dQ_dph), d2s2_dQ2))
    d2s2_dr2 = ladd({0: N.mpc(d2s2_dr2_0)},
                    lscale(lmul(dQ_dr, dQ_dr), d2s2_dQ2), lmul(ds2_dQ, d2Q_dr2))
    d2s2_dtph = ladd(lmul(ds2_dQ, d2Q_dtph),
                     lscale(lmul(dQ_dph, dQ_dt), d2s2_dQ2),
                     lmul(d2s2_dtQ, dQ_dph))
    d2s2_dt2 = ladd(d2s2_dt2_0, lmul(ds2_dQ, d2Q_dt2),
                    lscale(lmul(dQ_dt, dQ_dt), d2s2_dQ2),
                    {0: N.mpc(-ds2_dr0 * rtt + d2s2_dr2_0 * rt * rt - 2 * d2s2_dtr * rt)},
                    lscale(lmul(d2s2_dtQ, dQ_dt), 2.0))

    s2sq = lmul(s2, s2)
    dA = {"dt": dA_dt, "dr": dA_dr, "dth": dA_dth, "dph": dA_dph}
    ds2 = {"dt": ds2_dt, "dr": ds2_dr, "dth": {0: N.mpc(ds2_dth)}, "dph": ds2_dph}
    d2A = {"dt2": d2A_dt2, "dr2": d2A_dr2, "dth2": d2A_dth2, "dph2": d2A_dph2,
           "dtph": d2A_dtph}
    d2s2 = {"dt2": d2s2_dt2, "dr2": d2s2_dr2, "dth2": {0: N.mpc(d2s2_dth2)},
            "dph2": d2s2_dph2, "dtph": d2s2_dtph}
    pair = {"dt2": ("dt", "dt"), "dr2": ("dr", "dr"), "dth2": ("dth", "dth"),
            "dph2": ("dph", "dph"), "dtph": ("dph", "dt")}

    num = {"PhiS": lmul(A, s2sq)}
    for key in ("dt", "dr", "dth", "dph"):
        num[key] = lscale(lmul(ladd(lscale(lmul(ds2[key], A), -7),
                                    lscale(lmul(dA[key], s2), 2)), s2), B["half"])
    for key, (x, y) in pair.items():
        num[key] = lscale(ladd(
            lscale(lmul(lmul(ds2[x], ds2[y]), A), 63),
            lscale(lmul(s2, ladd(lmul(dA[x], ds2[y]), lmul(dA[y], ds2[x]),
                                 lmul(d2s2[key], A))), -14),
            lscale(lmul(d2A[key], s2sq), 4)), B["quarter"])
    return num, s2


def src_weights(ctx, dr, dtheta, N=Std):
    """(ctx, dr, dtheta) -> ({component: weight}, denominator) of the wave operator."""
    r = N.mpf(ctx.xp.r) + N.mpf(dr)
    theta = N.mpf(ctx.xp.theta) + N.mpf(dtheta)
    a = N.mpf(ctx.a)
    sinth = N.sin(theta)
    sinth2 = sinth * sinth
    sin2th = N.sin(2 * theta)
    cos2th = N.cos(2 * theta)
    r2, a2 = r * r, a * a
    r3, r4, a4 = r2 * r, r2 * r2, a2 * a2

    w = {
        "dr": (2 * a2 - 4 * r - 2 * a2 * r + 6 * r2 - 2 * r3
               + cos2th * (2 * (-2 + r) * r * (-1 + r) + 2 * a2 * (-1 + r))),
        "dph2": (-a2 + 4 * r - 2 * r2 + cos2th * (-a2)),
        "dr2": (-a4 + 4 * a2 * r - 4 * r2 - 2 * a2 * r2 + 4 * r3 - r4
                + cos2th * (a4 + (-2 + r) * r * (-2 + r) * r
                            + a2 * (-4 * r + 2 * r2))),
        "dth2": (-a2 + 2 * r - r2 + cos2th * ((-2 + r) * r + a2)),
        "dtph": (2 * a * r + cos2th * (-2 * a * r) + 4 * a * r * sinth2),
        "dt2": (sinth2 * (a4 + 2 * r4 + a2 * r * (2 + 3 * r))
                + cos2th * a2 * (a2 + (-2 + r) * r) * sinth2),
        "dth": (-a2 * sin2th + 2 * r * sin2th - r2 * sin2th),
    }
    denom = sinth2 * (a2 + (-2 + r) * r) * (a2 + 2 * r2 + a2 * cos2th)
    return w, denom


def src_contract(num, w, denom):
    """({component: Laurent}, weights, denominator) -> Laurent numerator of src."""
    total = {}
    for key, cf in w.items():
        if cf != 0:
            total = ladd(total, lscale(num[key], cf))
    return lscale(total, -1 / denom)


def src_numerator(ctx, dr, dtheta, N=Std):
    """(ctx, dr, dtheta) -> (Laurent numerator of src over s2^(11/2), s2 Laurent)."""
    num, s2 = numerators(ctx, dr, dtheta, N)
    w, denom = src_weights(ctx, dr, dtheta, N)
    return src_contract(num, w, denom), s2
