"""Whether branch conditioning predicts branch error, and where the two cross.

Each branch carries a condition number formed from quantities it already
computes:

    kappa_ei  = sum |EI[m][i][j][k] ellip[i] A[j] C1^k| / |sum of the same|
    kappa_leg = sum_n |T_n S_|m-n||                     / |sum_n T_n S_|m-n||

kappa * eps saturates at 1, above which the measured relative error is unbounded,
so predictiveness is graded where the bound is below USABLE.

Part 1  kappa against measured error, per branch.
Part 2  scaling of each kappa in (m, C1).
Part 3  the kappa crossing against the fitted law and against a measured oracle.
Part 4  min(kappa_ei, kappa_lg) * eps, the dispatch floor.
Part 5  the same for src, with ei_table conditioning proxied by the PhiS one.
Part 6  the legendre src error with the Laurent assembly in long double.
Part 7  the shipped rule, legendre when kappa_leg * eps <= tol, when
        C1 > switch_c1_src(m), or when m > EI_TABLE_MMAX.

Run inside the spectre conda env:
    python test/mmode_transition.py [stride]
"""
import itertools
import math
import os
import re
import sys
import time

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "effectivesource"))
from inspectre import EffectiveSource, Inspectre
from inspectre.mmode import (EI_TABLE_MMAX, SWITCH_TOL, blocks, kernel,
                            numerator_laurent, switch_c1, switch_c1_src)
from inspectre.mmode_laurent import src_numerator
from inspectre.separatrix import is_bound
from mmode_reference import MP, kernel_mp, modes
from scipy.special import ellipe, ellipk

EPS = np.finfo(float).eps
DPS = 80
USABLE = 0.1
LD_DPS = 19
CTX_SOURCE = os.path.join(os.path.dirname(__file__), "..", "..",
                          "effectivesource", "kerr-equatorial-ctx.c")

SPINS = (-0.9, 1e-8, 0.5, 0.9, 0.99)
PS = (6.0, 12.0, 20.0, 50.0)
ES = (0.0, 0.3, 0.7)
FRACS = (0.0, 0.31, 0.77)
GEOM = ("dr+", "dr-", "theta")
C1S = tuple(10.0 ** np.linspace(-3, 0.3, 18))
MS = (1, 2, 5, 10, 16, 20)


def ei_tables(path=CTX_SOURCE):
    """C source path -> (ReEI, ImEI) as (21, 2, 5, 27) arrays."""
    src = open(path).read()
    out = []
    for name in ("effsource_ReEI", "effsource_ImEI"):
        i = src.index(f"const double {name}[21][2][5][27] =")
        body = src[src.index("{", i):src.index("};", i)]
        a = np.array([float(x) for x in
                      re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", body)])
        out.append(a.reshape(21, 2, 5, 27))
    return out


ReEI, ImEI = ei_tables()


def ei_parts(ctx, m, dr, dtheta):
    """(ctx, m, dr, dtheta) -> (PhiSb, kappa) of the ei_table numerator sums."""
    ReA, ImA = blocks(ctx, dr, dtheta)
    alpha = ctx.alpha20 * dr * dr + ctx.alpha02 * dtheta * dtheta
    C1 = alpha / ctx.beta
    C = C1 ** np.arange(27)
    ab = alpha + ctx.beta
    ellip = (float(ellipk(ctx.beta / ab)), float(ellipe(ctx.beta / ab)))
    sr = ar = si = ai = 0.0
    for i in range(2):
        for j in range(5):
            for k in range(max(j - i, 0), m + 3 + j):
                t = ReEI[m, i, j, k] * ellip[i] * ReA[j] * C[k]
                sr += t
                ar += abs(t)
            for k in range(max(j - i - 1, 0), m + 2 + j):
                t = ImEI[m, i, j, k] * ellip[i] * ImA[j] * C[k]
                si += t
                ai += abs(t)
    re = 4.0 * sr / (ctx.beta * C[3] * ab ** 2.5)
    im = -32.0 * si / (ctx.beta ** 2 * C[2] * ab ** 1.5)
    val = complex(re, im)
    err = abs(re / sr) * ar if sr else 0.0
    err += abs(im / si) * ai if si else 0.0
    return val, err / abs(val)


def leg_parts(ctx, m, dr, dtheta):
    """(ctx, m, dr, dtheta) -> (PhiSb, kappa) of the legendre kernel contraction."""
    alpha = ctx.alpha20 * dr * dr + ctx.alpha02 * dtheta * dtheta
    N = numerator_laurent(*blocks(ctx, dr, dtheta))
    S = kernel(m + 5, alpha, ctx.beta)
    terms = [N[n] * S[abs(m - n)] for n in N]
    tot = sum(terms)
    return tot, sum(abs(t) for t in terms) / abs(tot)


def leg_src_parts(ctx, m, dr, dtheta):
    """(ctx, m, dr, dtheta) -> (src_m, kappa) of the legendre src contraction."""
    alpha = ctx.alpha20 * dr * dr + ctx.alpha02 * dtheta * dtheta
    sn, _ = src_numerator(ctx, dr, dtheta)
    T = {i // 2: v for i, v in sn.items() if i % 2 == 0} or {0: 0j}
    S = kernel(m + max(abs(n) for n in T), alpha, ctx.beta, q=5)
    terms = [t * S[abs(m - n)] for n, t in T.items()]
    tot = sum(terms)
    arg = m * (ctx.xp.phi + ctx.c * dr)
    rot = complex(math.cos(arg), -math.sin(arg))
    return tot * rot, sum(abs(t) for t in terms) / abs(tot)


def reference(ctx, m, dr, dtheta):
    """(ctx, m, dr, dtheta) -> (PhiSb_m without the rotation, src_m with it)."""
    rot = mp.e ** (-1j * m * (mp.mpf(ctx.xp.phi) + mp.mpf(ctx.c) * mp.mpf(dr)))
    phis, src = modes(ctx, [m], dr, dtheta, dps=DPS)
    return phis[m] / rot, src[m]


def build_cases():
    """() -> list of (label, es, ctx, r_p, spin, geom) over bound orbits."""
    out = []
    for a, p, e in itertools.product(SPINS, PS, ES):
        if not is_bound(a, p, e):
            continue
        insp = Inspectre(spin=a, semilatus_rectum=p, eccentricity=e)
        if not math.isfinite(insp.mino_period_r):
            continue
        for frac in FRACS:
            lam = frac * insp.mino_period_r
            r_p = insp.r_from_lambda(lam)
            if not math.isfinite(r_p):
                continue
            es = EffectiveSource(spin=a)
            es.set_particle(r_p, math.pi / 2, insp.phi_from_lambda(lam),
                            insp.energy, insp.angular_momentum,
                            insp.four_velocity_equatorial(lam))
            ctx = es._ef._ctx
            if not all(math.isfinite(v) for v in
                       (ctx.alpha20, ctx.alpha02, ctx.beta, ctx.c, ctx.xp.phi)):
                continue
            for geom in GEOM:
                out.append((f"a={a:g} p={p:g} e={e:g} f={frac:g} {geom}",
                            es, ctx, r_p, a, geom))
    return out


def offset(ctx, r_p, a, C1, geom):
    """(ctx, r_p, spin, C1, geom) -> (dr, dtheta) hitting that C1, or None."""
    alpha = C1 * ctx.beta
    r_h = 1.0 + math.sqrt(max(1.0 - a * a, 0.0))
    if geom == "theta":
        dtheta = math.sqrt(0.9 * alpha / ctx.alpha02)
        if dtheta > 1.4:
            return None
        dr = math.sqrt(0.1 * alpha / ctx.alpha20)
    else:
        dtheta = 0.0
        dr = math.sqrt(alpha / ctx.alpha20) * (1.0 if geom == "dr+" else -1.0)
    return None if r_p + dr < 1.10 * r_h else (dr, dtheta)


def quantiles(v, qs=(50, 90, 99, 100)):
    """(values, percentiles) -> formatted row."""
    v = np.asarray([x for x in v if np.isfinite(x)])
    if v.size == 0:
        return "no finite samples"
    return "  ".join(f"p{q}={np.percentile(v, q):.2e}" for q in qs)


DTYPE = [("case", int), ("geom", "U8"), ("C1", float), ("m", int),
         ("err_ei", float), ("kap_ei", float),
         ("err_lg", float), ("kap_lg", float),
         ("serr_ei", float), ("serr_lg", float), ("skap_lg", float)]


def collect(cases, stride=1):
    """(cases, stride) -> record array over (case, C1, m) with errors and kappas."""
    rows = []
    for idx, (label, es, ctx, r_p, a, geom) in enumerate(cases[::stride]):
        for C1 in C1S:
            off = offset(ctx, r_p, a, C1, geom)
            if off is None:
                continue
            dr, dtheta = off
            for m in MS:
                ref, sref = reference(ctx, m, dr, dtheta)
                if ref == 0 or sref == 0:
                    continue
                v_ei, k_ei = ei_parts(ctx, m, dr, dtheta)
                v_lg, k_lg = leg_parts(ctx, m, dr, dtheta)
                s_ei = es._ef.calc_m_offset(m, dr, dtheta)[3]
                s_lg, sk_lg = leg_src_parts(ctx, m, dr, dtheta)
                rows.append((idx, geom, C1, m,
                             float(abs(mp.mpc(v_ei) - ref) / abs(ref)), k_ei,
                             float(abs(mp.mpc(v_lg) - ref) / abs(ref)), k_lg,
                             float(abs(mp.mpc(s_ei[0], s_ei[1]) - sref) / abs(sref)),
                             float(abs(mp.mpc(s_lg) - sref) / abs(sref)), sk_lg))
    return np.array(rows, dtype=DTYPE)


def part1(rec):
    """record array -> None; prints kappa predictiveness per branch."""
    print("Part 1: measured error / (kappa * eps), where the bound is below "
          f"{USABLE:g}")
    for tag, e, k in (("ei_table", rec["err_ei"], rec["kap_ei"]),
                      ("legendre", rec["err_lg"], rec["kap_lg"])):
        b = k * EPS
        use = b < USABLE
        r = e[use] / b[use]
        print(f"  {tag}: {use.sum():5d}/{len(rec)} usable   "
              f"{quantiles(r, (1, 50, 90, 99, 100))}")
        bad = use & (e > 100.0 * b) & (b < 1e-8)
        print(f"            false confidence (err > 100x a bound under 1e-8): "
              f"{bad.sum()}")
    print("\n  p99 of the ratio by m")
    for m in MS:
        s = rec[rec["m"] == m]
        out = []
        for tag, e, k in (("ei", s["err_ei"], s["kap_ei"]),
                          ("leg", s["err_lg"], s["kap_lg"])):
            b = k * EPS
            use = b < USABLE
            out.append(f"{tag} {np.percentile(e[use] / b[use], 99):.2e} "
                       f"({use.sum():4d})" if use.sum() else f"{tag} -")
        print(f"   m={m:<3d} " + "   ".join(out))
    print()


def part2(rec):
    """record array -> None; prints the scaling of each kappa in (m, C1)."""
    print("Part 2: scaling")
    z = 1.0 + 2.0 * rec["C1"]
    rho = z - np.sqrt(z * z - 1.0)
    scaled = rec["kap_ei"] * rho ** rec["m"]
    print("  ei_table, kappa * rho^m by m (flat => kappa ~ rho^-m):")
    for m in MS:
        s = scaled[rec["m"] == m]
        print(f"   m={m:<3d} {quantiles(s, (10, 50, 90)):s}")
    print("  legendre, d log kappa / d log C1 by m (expect a negative constant):")
    for m in MS:
        s = rec[rec["m"] == m]
        ok = (s["kap_lg"] > 2.0) & (s["C1"] < 0.2)
        if ok.sum() < 3:
            print(f"   m={m:<3d} too few samples")
            continue
        p = np.polyfit(np.log(s["C1"][ok]), np.log(s["kap_lg"][ok]), 1)
        print(f"   m={m:<3d} slope {p[0]:+.3f}   intercept {math.exp(p[1]):.3e} "
              f"  n={ok.sum()}")
    print()


def crossing(x, ya, yb):
    """(x, ya, yb) -> x where ya - yb changes sign, log-interpolated, or nan."""
    d = np.log(ya) - np.log(yb)
    s = np.where(np.diff(np.sign(d)) != 0)[0]
    if s.size == 0:
        return float("nan")
    i = s[-1]
    t = d[i] / (d[i] - d[i + 1])
    return float(math.exp(math.log(x[i]) + t * (math.log(x[i + 1]) - math.log(x[i]))))


def part3(rec):
    """record array -> None; prints kappa crossing vs the fit and the oracle."""
    print("Part 3: crossing, kappa vs measured oracle vs the fitted law")
    print(f"  {'m':>3} {'kappa cross':>26} {'oracle cross':>26} "
          f"{'fit':>9} {'kappa/fit':>10}")
    for m in MS:
        kx, ox = [], []
        for c in np.unique(rec["case"]):
            s = rec[(rec["m"] == m) & (rec["case"] == c)]
            if s.size < 4:
                continue
            o = np.argsort(s["C1"])
            s = s[o]
            kx.append(crossing(s["C1"], s["kap_ei"], s["kap_lg"]))
            ox.append(crossing(s["C1"],
                               np.maximum(s["err_ei"], 1e-300),
                               np.maximum(s["err_lg"], 1e-300)))
        kx = np.array([v for v in kx if np.isfinite(v)])
        ox = np.array([v for v in ox if np.isfinite(v)])
        fit = switch_c1(m)
        km = np.median(kx) if kx.size else float("nan")
        print(f"  {m:>3} {np.min(kx):.3e}-{np.max(kx):.3e} ({kx.size:3d})  "
              f"{np.min(ox):.3e}-{np.max(ox):.3e} ({ox.size:3d})  "
              f"{fit:9.3e} {km / fit:10.2f}")
    print()


def part4(rec):
    """record array -> None; prints the best achievable error over the C1 range."""
    print("Part 4: min(kappa_ei, kappa_lg) * eps, the dispatch floor")
    print(f"  {'m':>3} " + " ".join(f"{c:>9.2e}" for c in C1S[::3]))
    for m in MS:
        row = []
        for c in C1S[::3]:
            s = rec[(rec["m"] == m) & (np.abs(rec["C1"] / c - 1) < 1e-9)]
            if s.size == 0:
                row.append(f"{'-':>9}")
                continue
            best = np.minimum(s["kap_ei"], s["kap_lg"]) * EPS
            row.append(f"{np.max(best):9.2e}")
        print(f"  {m:>3} " + " ".join(row))
    print("\n  worst dispatch floor over all C1, by m")
    for m in MS:
        s = rec[rec["m"] == m]
        best = np.minimum(s["kap_ei"], s["kap_lg"]) * EPS
        print(f"   m={m:<3d} {np.max(best):.3e} at C1={s['C1'][np.argmax(best)]:.3e}")
    print()


def part5(rec):
    """record array -> None; prints the src study, ei_table conditioning proxied
    by the PhiS one."""
    print("Part 5: src")
    b = rec["skap_lg"] * EPS
    use = b < USABLE
    print(f"  legendre: err / (kappa_lg_src eps), {use.sum()}/{len(rec)} usable   "
          f"{quantiles(rec['serr_lg'][use] / b[use], (50, 90, 99, 100))}")
    b = rec["kap_ei"] * EPS
    use = b < USABLE
    print(f"  ei_table: err / (kappa_ei_PhiS eps), {use.sum()}/{len(rec)} usable   "
          f"{quantiles(rec['serr_ei'][use] / b[use], (50, 90, 99, 100))}")
    print("\n  inflation of the PhiS proxy on src, p99 by m")
    for m in MS:
        s = rec[rec["m"] == m]
        b = s["kap_ei"] * EPS
        use = b < USABLE
        if not use.sum():
            print(f"   m={m:<3d} -")
            continue
        print(f"   m={m:<3d} {np.percentile(s['serr_ei'][use] / b[use], 99):.2e}  "
              f"max {np.max(s['serr_ei'][use] / b[use]):.2e}  ({use.sum():4d})")
    print("\n  crossing for src")
    print(f"  {'m':>3} {'kappa cross':>26} {'oracle cross':>26} "
          f"{'fit':>9} {'kappa/fit':>10}")
    for m in MS:
        kx, ox = [], []
        for c in np.unique(rec["case"]):
            s = rec[(rec["m"] == m) & (rec["case"] == c)]
            if s.size < 4:
                continue
            s = s[np.argsort(s["C1"])]
            kx.append(crossing(s["C1"], s["kap_ei"], s["skap_lg"]))
            ox.append(crossing(s["C1"], np.maximum(s["serr_ei"], 1e-300),
                               np.maximum(s["serr_lg"], 1e-300)))
        kx = np.array([v for v in kx if np.isfinite(v)])
        ox = np.array([v for v in ox if np.isfinite(v)])
        fit = switch_c1_src(m)
        km = np.median(kx) if kx.size else float("nan")
        print(f"  {m:>3} {np.min(kx):.3e}-{np.max(kx):.3e} ({kx.size:3d})  "
              f"{np.min(ox):.3e}-{np.max(ox):.3e} ({ox.size:3d})  "
              f"{fit:9.3e} {km / fit:10.2f}")
    print("\n  src dispatch floor, worst over C1 by m")
    for m in MS:
        s = rec[rec["m"] == m]
        best = np.minimum(s["kap_ei"], s["skap_lg"]) * EPS
        print(f"   m={m:<3d} {np.max(best):.3e} at C1={s['C1'][np.argmax(best)]:.3e}")
    print()


def part6(cases, stride, c1s=(1e-2, 3e-2, 1e-1, 5e-1), ms=(2, 10, 20)):
    """(cases, stride, C1 list, m list) -> None; the legendre src error with the
    Laurent assembly in long double, against the double conditioning."""
    print("Part 6: legendre src with a long double assembly")
    print(f"  {'C1':>8} {'m':>3} {'err double':>11} {'err ld':>11} "
          f"{'kappa_ctr':>11} {'err_ld/(k eps)':>15}")
    rows = {}
    for label, es, ctx, r_p, a, geom in cases[::stride]:
        for C1 in c1s:
            off = offset(ctx, r_p, a, C1, geom)
            if off is None:
                continue
            dr, dtheta = off
            alpha = ctx.alpha20 * dr * dr + ctx.alpha02 * dtheta * dtheta
            mp.mp.dps = DPS
            Tm = {i // 2: v for i, v in src_numerator(ctx, dr, dtheta, MP)[0].items()
                  if i % 2 == 0}
            Tf = {i // 2: v for i, v in src_numerator(ctx, dr, dtheta)[0].items()
                  if i % 2 == 0}
            mp.mp.dps = LD_DPS
            Tl = {i // 2: v for i, v in src_numerator(ctx, dr, dtheta, MP)[0].items()
                  if i % 2 == 0}
            deg = max(abs(n) for n in Tf)
            for m in ms:
                mp.mp.dps = DPS
                Sm = kernel_mp(m + deg, mp.mpf(alpha), mp.mpf(ctx.beta), q=5)
                ref = sum(Tm[n] * Sm[abs(m - n)] for n in Tm)
                Sf = kernel(m + deg, alpha, ctx.beta, q=5)
                v_d = sum(Tf[n] * Sf[abs(m - n)] for n in Tf)
                mp.mp.dps = LD_DPS
                Sl = kernel_mp(m + deg, mp.mpf(alpha), mp.mpf(ctx.beta), q=5)
                v_l = sum(Tl[n] * Sl[abs(m - n)] for n in Tl)
                mp.mp.dps = DPS
                k = sum(abs(Tf[n] * Sf[abs(m - n)]) for n in Tf) / abs(v_d)
                rows.setdefault((C1, m), []).append(
                    (float(abs(mp.mpc(v_d) - ref) / abs(ref)),
                     float(abs(mp.mpc(v_l) - ref) / abs(ref)), k))
    for (C1, m), v in sorted(rows.items()):
        ed = np.array([x[0] for x in v])
        el = np.array([x[1] for x in v])
        kk = np.array([x[2] for x in v])
        print(f"  {C1:>8.2g} {m:>3} {np.max(ed):>11.2e} {np.max(el):>11.2e} "
              f"{np.max(kk):>11.2e} {np.max(el / (kk * EPS)):>15.2e}")
    print()


RULE_DTYPE = [("case", int), ("C1", float), ("m", int), ("kap", float),
              ("err_ei", float), ("err_lg", float), ("err_ld", float)]


def collect_rule(cases, stride=1):
    """(cases, stride) -> record array carrying the shipped rule's inputs.

    kap is the legendre src contraction condition number, computed in double;
    err_ld is the legendre src error with the Laurent assembly and the kernel
    carried at long double precision.
    """
    rows = []
    for idx, (label, es, ctx, r_p, a, geom) in enumerate(cases[::stride]):
        for C1 in C1S:
            off = offset(ctx, r_p, a, C1, geom)
            if off is None:
                continue
            dr, dtheta = off
            alpha = ctx.alpha20 * dr * dr + ctx.alpha02 * dtheta * dtheta
            mp.mp.dps = DPS
            Tm = {i // 2: v for i, v in src_numerator(ctx, dr, dtheta, MP)[0].items()
                  if i % 2 == 0}
            Tf = {i // 2: v for i, v in src_numerator(ctx, dr, dtheta)[0].items()
                  if i % 2 == 0}
            mp.mp.dps = LD_DPS
            Tl = {i // 2: v for i, v in src_numerator(ctx, dr, dtheta, MP)[0].items()
                  if i % 2 == 0}
            deg = max(abs(n) for n in Tf)
            for m in MS:
                mp.mp.dps = DPS
                Sm = kernel_mp(m + deg, mp.mpf(alpha), mp.mpf(ctx.beta), q=5)
                ref = sum(Tm[n] * Sm[abs(m - n)] for n in Tm)
                if ref == 0:
                    continue
                Sf = kernel(m + deg, alpha, ctx.beta, q=5)
                v_d = sum(Tf[n] * Sf[abs(m - n)] for n in Tf)
                mp.mp.dps = LD_DPS
                Sl = kernel_mp(m + deg, mp.mpf(alpha), mp.mpf(ctx.beta), q=5)
                v_l = sum(Tl[n] * Sl[abs(m - n)] for n in Tl)
                mp.mp.dps = DPS
                arg = m * (ctx.xp.phi + ctx.c * dr)
                rot = mp.e ** (-1j * m * (mp.mpf(ctx.xp.phi) + mp.mpf(ctx.c) * mp.mpf(dr)))
                s_ei = es._ef.calc_m_offset(m, dr, dtheta)[3]
                e_ei = float(abs(mp.mpc(s_ei[0], s_ei[1]) / rot - ref) / abs(ref))
                rows.append((idx, C1, m,
                             sum(abs(Tf[n] * Sf[abs(m - n)]) for n in Tf) / abs(v_d),
                             e_ei,
                             float(abs(mp.mpc(v_d) - ref) / abs(ref)),
                             float(abs(mp.mpc(v_l) - ref) / abs(ref))))
    return np.array(rows, dtype=RULE_DTYPE)


def shipped_rule(rec, tol):
    """(record array, tol) -> bool mask, True where calc_m keeps the legendre branch."""
    return ((rec["kap"] * EPS <= tol)
            | (rec["C1"] > np.array([switch_c1_src(m) for m in rec["m"]]))
            | (rec["m"] > EI_TABLE_MMAX))


def part7(rec, tols=(1e-10, 1e-8, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2)):
    """record array -> None; grades the shipped rule against the fit and an oracle."""
    print("Part 7: the shipped rule, legendre when kappa_ctr * eps <= tol, when "
          "C1 > switch_c1_src(m), or when m > EI_TABLE_MMAX")
    fit = rec["C1"] > np.array([switch_c1_src(m) for m in rec["m"]])
    bound = rec["kap"] * EPS
    oracle = np.minimum(rec["err_ei"], rec["err_ld"])
    pick = lambda use: np.where(use, rec["err_ld"], rec["err_ei"])
    print(f"  {'rule':>16} {'legendre share':>15} {'p99 err':>10} {'worst err':>10} "
          f"{'worst/oracle':>13} {'bound leg viol':>15}")

    def row(tag, use, viol="n/a"):
        e = pick(use)
        print(f"  {tag:>16} {100 * np.mean(use):14.1f}% {np.percentile(e, 99):10.2e} "
              f"{np.max(e):10.2e} {np.max(e / np.maximum(oracle, 1e-300)):13.2e} "
              f"{viol:>15}")

    row("fitted law", fit)
    row(f"kappa only {SWITCH_TOL:.0e}", bound <= SWITCH_TOL,
        str(int(((bound <= SWITCH_TOL) & (rec["err_ld"] > SWITCH_TOL)).sum())))
    for tol in tols:
        use = shipped_rule(rec, tol)
        row(f"shipped {tol:.0e}", use,
            str(int(((bound <= tol) & (rec["err_ld"] > tol)).sum())))
    row("oracle", rec["err_ld"] < rec["err_ei"])

    print("\n  where the shipped rule falls back to ei_table, is ei_table any good?")
    for tol in tols:
        fb = ~shipped_rule(rec, tol)
        if not fb.sum():
            continue
        bad = fb & (rec["err_ei"] > tol)
        print(f"   tol {tol:.0e}: fallback {fb.sum():5d}/{len(rec)}, "
              f"of which ei_table also above tol: {bad.sum():5d} "
              f"({100 * bad.sum() / max(fb.sum(), 1):.1f}%), "
              f"worst there {np.max(rec['err_ei'][bad]) if bad.sum() else 0:.2e}")

    print("\n  best single tolerance per m, and what the fit achieves there")
    print(f"   {'m':>3} {'best tol':>10} {'worst err':>10} {'fit worst':>10} "
          f"{'oracle':>10} {'kappa*eps at the crossing':>26}")
    for m in MS:
        s_ = rec[rec["m"] == m]
        b = s_["kap"] * EPS
        best, bw = None, np.inf
        for tol in 10.0 ** np.arange(-14, 2, 0.25):
            w = np.max(np.where(shipped_rule(s_, tol), s_["err_ld"], s_["err_ei"]))
            if w < bw:
                best, bw = tol, w
        fw = np.max(np.where(s_["C1"] > switch_c1_src(m), s_["err_ld"], s_["err_ei"]))
        cross = b[np.argsort(np.abs(np.log(np.maximum(s_["err_ei"], 1e-300))
                                    - np.log(np.maximum(s_["err_ld"], 1e-300))))[:3]]
        print(f"   {m:>3} {best:10.1e} {bw:10.2e} {fw:10.2e} "
              f"{np.max(np.minimum(s_['err_ei'], s_['err_ld'])):10.2e} "
              f"{np.median(cross):26.2e}")

    print(f"\n  per m at the shipped tolerance {SWITCH_TOL:.0e}")
    print(f"   {'m':>3} {'shipped':>11} {'kappa only':>11} {'fit':>11} {'oracle':>11} "
          f"{'legendre share':>15} {'fit C1':>9}")
    for m in MS:
        s_ = rec[rec["m"] == m]
        use = shipped_rule(s_, SWITCH_TOL)
        kap = s_["kap"] * EPS <= SWITCH_TOL
        print(f"   {m:>3} {np.max(np.where(use, s_['err_ld'], s_['err_ei'])):11.2e} "
              f"{np.max(np.where(kap, s_['err_ld'], s_['err_ei'])):11.2e} "
              f"{np.max(np.where(s_['C1'] > switch_c1_src(m), s_['err_ld'], s_['err_ei'])):11.2e} "
              f"{np.max(np.minimum(s_['err_ei'], s_['err_ld'])):11.2e} "
              f"{100 * use.mean():14.1f}% {switch_c1_src(m):9.3e}")

    print("\n  long double assembly against a double one, worst error by m")
    for m in MS:
        s = rec[rec["m"] == m]
        print(f"   m={m:<3d} double {np.max(s['err_lg']):.2e}   "
              f"long double {np.max(s['err_ld']):.2e}   "
              f"gain {np.max(s['err_lg']) / max(np.max(s['err_ld']), 1e-300):.0f}x")
    print()


if __name__ == "__main__":
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    cases = build_cases()
    print(f"matrix: {len(cases)} cases, stride {stride} -> {len(cases[::stride])} used, "
          f"{len(C1S)} C1 in [{C1S[0]:.3g}, {C1S[-1]:.3g}], m in {MS}, "
          f"reference at dps {DPS}")
    t0 = time.time()
    rec = collect(cases, stride)
    print(f"{len(rec)} samples in {time.time() - t0:.0f} s\n")
    part1(rec)
    part2(rec)
    part3(rec)
    part4(rec)
    part5(rec)
    part6(cases, stride)
    t0 = time.time()
    rrec = collect_rule(cases, stride)
    print(f'rule matrix: {len(rrec)} samples in {time.time() - t0:.0f} s')
    np.save(os.path.join(os.path.dirname(__file__),
                         f'mmode_transition_rule_{stride}.npy'), rrec)
    part7(rrec)
