"""Validate the long-double kernel reassembly used by the gold reference.

The seven analytic channels of calc_m_split reassemble to the m-mode outputs as
    X = A + L ln(alpha) + sum_{q=1..5} Pq / alpha^q,
channel q+1 carrying the 1/alpha^q term. The channels cancel against each other:
the puncture reaches only 1/alpha^3 and stays well conditioned, while src
reaches 1/alpha^5 and its conditioning grows without bound as alpha -> 0. The
gold path therefore takes the channels unnarrowed (calc_m_split_l) and fuses the
dominant P1/P2 pair as (P1 alpha + P2)/alpha^2.

Part 1 grades four evaluators on the puncture against the BigFloat truth table.
Only PhiS is graded there: the table's src column follows a different
normalization, reported but not used.
Part 2 grades src by convention-free +-ulp jitter instead.
Part 3 checks inspectre's orbit-parameterized gold agrees with effectivesource's
offset gold, which the truth table cannot cover since it fixes ur = 0.

Run inside the spectre conda env:
    python test/test_gold_reference.py
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "effectivesource"))
import effsource_equatorial as ee
import inspectre as I
import inspectre_c as c
import kerrgeodesics as kg

LD = np.longdouble
M, A = 1.0, 0.5
RP, THP, PHP = 10.0, math.pi / 2, math.pi / 3

failures = 0


def check(name, cond, detail=""):
    global failures
    print(f"[{'ok ' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures += 1


def assemble(blocks, ncomp, alpha, fused, dtype):
    """Reassemble `ncomp` components from the 7 channel blocks.

    Channel index q holds the 1/alpha^(q-1) term for q >= 2. fused=True pays the
    P1/P2 cancellation as one (P1 alpha + P2)/alpha^2 subtraction.
    """
    al = dtype(alpha)
    lg = np.log(al) if dtype is LD else math.log(float(al))
    out = []
    for k in range(ncomp):
        ch = [dtype(blocks[q][k]) for q in range(7)]
        v = ch[0] + ch[1] * lg
        if fused:
            v += (ch[2] * al + ch[3]) / (al * al)
            q0 = 4
        else:
            v += ch[2] / al
            q0 = 3
        ap = al ** (q0 - 1)
        for q in range(q0, 7):
            v += ch[q] / ap
            ap *= al
        out.append(v)
    return out


def alpha_of(ctx, dr, dth):
    al = ctx.get_alpha()
    return LD(al[0]) * LD(dr) ** 2 + LD(al[1]) * LD(dth) ** 2


# --- part 1: puncture vs the BigFloat truth ---------------------------------
E = math.sqrt(1 - 2 * M / RP + A**2 / RP**2 - 2 * M * A**2 / RP**3) / math.sqrt(
    1 - 3 * M / RP + 2 * A * math.sqrt(M / RP**3)
)
L = (RP**2 - 2 * M * RP + A * math.sqrt(M * RP)) / (
    RP * math.sqrt(RP - 3 * M + 2 * A * math.sqrt(M / RP))
)
ee.disable_gsl_error_handler()
ctx = ee.EffsourceEquatorialContext(M, A)
ctx.set_particle(ee.make_coordinate(t=0.0, r=RP, theta=THP, phi=PHP), E, L, 0.0)

truth_csv = os.path.join(os.path.dirname(__file__), "..", "..", "effectivesource",
                         "julia", "examples", "truth_circular.csv")
if not os.path.exists(truth_csv):
    print(f"truth file missing ({truth_csv}); run julia dump_truth.jl first")
    sys.exit(1)
rows = [r for r in csv.DictReader(open(truth_csv))]

print("m-mode puncture: relative error vs BigFloat truth (near zone only)\n")
print(f"{'m':>3} {'dr':>9} {'dtheta':>9} {'|PhiS|':>10} "
      f"{'calc_m':>9} {'offset':>9} {'split/f64':>9} {'gold':>9}")

worst = {k: 0.0 for k in ("calc_m", "offset", "split", "gold")}
ratio_src = []
nrows = 0
for row in rows:
    m = int(row["m"])
    dr, dth = float(row["dr"]), float(row["dtheta"])
    truth = complex(float(row["RePhiS"]), float(row["ImPhiS"]))
    if truth == 0:
        continue
    nrows += 1

    x = ee.make_coordinate(t=0.0, r=RP + dr, theta=THP + dth, phi=PHP)
    p_cm = complex(*ctx.calc_PhiS_m(m, x))
    p_off = complex(*ctx.calc_PhiS_m_offset(m, dr, dth))

    PhiS_s, _, _, src_s = ctx.calc_m_split(m, dr, dth)
    alpha = alpha_of(ctx, dr, dth)
    p_split = complex(*[float(v) for v in assemble(PhiS_s, 2, alpha, False, float)])
    p_gold = complex(*ctx.calc_m_gold(m, dr, dth)[0])

    errs = {"calc_m": abs(p_cm - truth) / abs(truth),
            "offset": abs(p_off - truth) / abs(truth),
            "split": abs(p_split - truth) / abs(truth),
            "gold": abs(p_gold - truth) / abs(truth)}
    for k, v in errs.items():
        worst[k] = max(worst[k], v)

    ts = float(row["Resrc"])
    if ts and m == 0 and dth == 0.0 and dr > 1e-6:
        ratio_src.append(ctx.calc_m_offset(m, dr, dth)[3][0] / ts)

    if math.hypot(dr, dth) < 3e-7:
        print(f"{m:>3} {dr:>9.1e} {dth:>9.1e} {abs(truth):>10.3e} "
              f"{errs['calc_m']:>9.1e} {errs['offset']:>9.1e} "
              f"{errs['split']:>9.1e} {errs['gold']:>9.1e}")

print(f"\nrows compared: {nrows}")
for k in ("calc_m", "offset", "split", "gold"):
    print(f"  worst {k:>9}: {worst[k]:.2e}")
rs = np.array(ratio_src)
print(f"\n  src column is on another normalization; over the m=0, dtheta=0 family "
      f"C/truth = {rs.mean():.6f} to {(rs.max() - rs.min()) / abs(rs.mean()):.0e}. "
      f"Not used below.")

check("split reassembly reproduces the puncture", worst["split"] < 1e-8,
      f"worst={worst['split']:.2e}")
check("gold beats the double offset path", worst["gold"] < worst["offset"],
      f"gold={worst['gold']:.2e} vs offset={worst['offset']:.2e}")
check("gold beats the double reassembly", worst["gold"] < worst["split"],
      f"gold={worst['gold']:.2e} vs split={worst['split']:.2e}")

# --- part 2: src graded by convention-free ulp jitter ------------------------
# alpha = a20 dr^2 + a02 dtheta^2, so perturbing the SUBDOMINANT offset moves the
# result by nothing the double core can even represent -- its spread then reads 0
# for insensitivity, not accuracy. Always jitter the offset that dominates alpha.
print("\neffective source: +-3-ulp jitter of the alpha-dominant offset\n")
print(f"{'dtheta':>9} {'dr':>9} {'jittered':>9} "
      f"{'offset spread':>14} {'gold spread':>12} {'ratio':>8}")


def jitter(fn, dr0, dth0, n=3):
    """Spread of fn over +-n ulp of whichever offset dominates alpha."""
    al = ctx.get_alpha()
    dom_r = al[0] * dr0 ** 2 >= al[1] * dth0 ** 2
    x0 = dr0 if dom_r else dth0
    vals = []
    x = x0
    for _ in range(n):
        x = math.nextafter(x, -math.inf)
    for _ in range(2 * n + 1):
        vals.append(fn(x, dth0) if dom_r else fn(dr0, x))
        x = math.nextafter(x, math.inf)
    mean = sum(vals) / len(vals)
    spread = (max(vals) - min(vals)) / abs(mean) if mean else float("nan")
    return spread, ("dr" if dom_r else "dtheta")


def src_offset(dr, dth):
    return math.hypot(*ctx.calc_m_offset(2, dr, dth)[3])


def src_gold(dr, dth):
    return math.hypot(*ctx.calc_m_gold(2, dr, dth)[2])


gains = []
for dth in (1e-2, 1e-4, 1e-6):
    for dr in (1e-2, 1e-5, 1e-8):
        ctx.calc_m_offset(2, dr, dth)
        so, which = jitter(src_offset, dr, dth)
        sg, _ = jitter(src_gold, dr, dth)
        if so > 0 and sg > 0:
            gains.append(so / sg)
        print(f"{dth:>9.0e} {dr:>9.0e} {which:>9} "
              f"{so:>14.2e} {sg:>12.2e} {so / sg if sg > 0 else float('inf'):>8.1f}x")

med = float(np.median(gains))
check("gold is quieter than the double core on src", med > 10.0,
      f"median jitter ratio = {med:.1f}x")

# --- part 3: C gold entry point vs the long-double assembly on an orbit ------
a, p, e, m = 0.5, 10.0, 0.3, 2
insp = I.Inspectre(spin=a, semilatus_rectum=p, eccentricity=e, x=1.0)
orbpar = insp._orbpar
ctx2 = ee.EffsourceEquatorialContext(1.0, a)
raw2 = ctx2._ctx
Vr = c.inspectre_radial_mino_period(orbpar)

def rel(got, ref):
    n = math.hypot(got[0] - float(ref[0]), got[1] - float(ref[1]))
    d = math.hypot(float(ref[0]), float(ref[1]))
    return n / d if d else n


def assemble_at(dr, dth, psi, lam):
    """calc_m_gold with the particle seated as insp_seat_particle does.

    korb_rfrompsi supplies the absolute r_p that the smooth channel coefficients
    depend on; dr is passed separately so the two arms share the same offset.
    """
    ctx2.set_particle(
        ee.make_coordinate(t=0.0, r=kg.korb_rfrompsi(psi, orbpar),
                           theta=math.pi / 2,
                           phi=kg.korb_phifromla(lam, orbpar)),
        orbpar.E, orbpar.Lz, c.fourVel(psi, a, p, e, orbpar.E))
    gp, _, gs = ctx2.calc_m_gold(m, dr, dth)
    return gp, gs


# Both arms now call calc_m_gold, so any residual is the dr they were handed:
# C forms it with glibc cosl, this arm with numpy's longdouble cos, and the two
# need not agree in the last extended-precision bit. src amplifies a 1-ulp dr
# difference by the channel conditioning, so the honest bar is that sensitivity,
# not zero: `sens` measures it per row.
print("\ninspectre orbit gold vs effectivesource offset gold\n")
print(f"{'lam/Vr':>8} {'dtheta':>9} {'|src| gold':>13} "
      f"{'PhiS rel':>10} {'src rel':>10} {'src sens':>10}")
worst_phis = 0.0
over_sens = 0
for frac in (0.05, 0.25, 0.5, 0.75):
    for dth in (1e-2, 1e-4, 1e-6):
        lam = frac * Vr
        fp = c.make_field_point(p, dth)
        gp, _, gs = c.eval_gold_at_lambda(raw2, m, fp, lam, orbpar, a, p, e)

        psi = kg.korb_psifromla(lam, orbpar)
        r_p_l = LD(p) / (LD(1) + LD(e) * np.cos(LD(psi)))
        dr = float(LD(p) - r_p_l)

        rp_, rs_ = assemble_at(dr, dth, psi, lam)
        rp2, rs2 = assemble_at(math.nextafter(dr, math.inf), dth, psi, lam)
        sens = rel([float(v) for v in rs2], rs_)

        ep, es = rel(gp, rp_), rel(gs, rs_)
        worst_phis = max(worst_phis, ep)
        if es > max(sens, 1e-15):
            over_sens += 1
        print(f"{frac:>8.2f} {dth:>9.0e} {math.hypot(*gs):>13.5e} "
              f"{ep:>10.1e} {es:>10.1e} {sens:>10.1e}")

check("orbit gold reproduces the puncture to rounding", worst_phis < 1e-15,
      f"worst rel = {worst_phis:.2e}")
check("orbit gold src agrees within the 1-ulp-dr sensitivity", over_sens == 0,
      f"{over_sens} row(s) exceeded their own sensitivity")

# --- part 5: gold-driven panel quadrature reproduces the double one ----------
# A gold reference is built by re-evaluating the panel nodes, which means folding
# lambda into [0, Vr) exactly as the build does (inspectre_lib.c:892). Skipping
# that fold rotates the m-mode carrier exp(-i m phi_p) on the trailing nodes --
# right modulus, wrong phase -- and quietly corrupts every amplitude graded
# against it. PhiS is well conditioned, so gold and double must agree there to
# rounding; any disagreement means the re-evaluation is off the node.
print("\ngold-driven panel quadrature vs the C panel integrate\n")
print(f"{'dtheta':>9} {'n':>4} {'nodes':>6} {'PhiS rel':>10} {'src rel':>10}")
worst_q = 0.0
for dth in (1e-2, 1e-4):
    fpq = c.make_field_point(p, dth)
    sp = c.panel_nodes_build(raw2, m, fpq, orbpar, a, p, e,
                             insp.omega_phi, insp.omega_r,
                             order=24, maxLevels=40, nMax=8)
    try:
        N = sp.n
        lamA = c.doubleArray.frompointer(sp.lam)
        tA = c.doubleArray.frompointer(sp.t)
        JA = c.doubleArray.frompointer(sp.J)
        wA = c.doubleArray.frompointer(sp.w)
        gph = np.empty(N, dtype=complex)
        gsr = np.empty(N, dtype=complex)
        for k in range(N):
            le = lamA[k] - sp.Vr if lamA[k] >= sp.Vr else lamA[k]
            q, _, r_ = c.eval_gold_at_lambda(raw2, m, fpq, le, orbpar, a, p, e)
            gph[k] = complex(q[0], q[1])
            gsr[k] = complex(r_[0], r_[1])
        tv = np.array([tA[k] for k in range(N)])
        W = np.array([wA[k] * JA[k] for k in range(N)])
        for n in (0, 8):
            ph, _, sr = c.panel_nodes_integrate(sp, m, n, insp.omega_phi,
                                                insp.omega_r)
            car = np.exp(1j * (m * insp.omega_phi + n * insp.omega_r) * tv)
            gp_ = np.sum(W * gph * car) / sp.Tr
            gs_ = np.sum(W * gsr * car) / sp.Tr
            ep = abs(gp_ - complex(*ph)) / abs(gp_)
            es = abs(gs_ - complex(*sr)) / abs(gs_)
            worst_q = max(worst_q, ep)
            print(f"{dth:>9.0e} {n:>4} {N:>6} {ep:>10.1e} {es:>10.1e}")
    finally:
        c.panel_nodes_free(sp)

check("gold panel quadrature matches the double panel on PhiS", worst_q < 1e-13,
      f"worst rel = {worst_q:.2e}")

# --- part 4: eval counter ---------------------------------------------------
c.inspectre_eval_count_reset()
fp = c.make_field_point(p, 1e-2)
for k in range(7):
    c.eval_at_lambda(raw2, m, fp, 0.1 * k * Vr, orbpar, a, p, e)
got = c.inspectre_eval_count()
check("eval counter counts each evaluation", got == 7, f"got {got}, want 7")

print(f"\n{failures} failure(s)")
sys.exit(1 if failures else 0)
