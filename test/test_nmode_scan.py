#!/usr/bin/env python
"""mn-mode amplitude scan over a field-point grid: evaluator precision x quadrature.

Given (m, n, a, p, e, Ntimes), computes the mn-mode amplitudes of

    Phi        <- PhiS[0:2]
    drPhi      <- dPhiS[2:4]
    dthetaPhi  <- dPhiS[4:6]
    src        <- src[0:2]

at every point of a two-dimensional field-point grid. The radial grid is either
linear over [r_min - pad, r_max + pad] or, with --r-scheme anchored, the six radii
a worldtube cares about: r_min - pad, r_min, the two interior thirds of the
libration range, r_max, r_max + pad. z = cos(theta) is log spaced.

Each cell is run once per arm, where an arm is an evaluator crossed with a
field-point route.

    evaluator  f64  effsource calc_m_offset, kernel channels reassembled in double
               ld   effsource calc_m_gold, the same channels in long double
    route      dtheta  theta - pi/2 handed to C directly, full relative accuracy
               theta   absolute theta = pi/2 + dtheta, differenced inside C

Arm labels compose as f64, f64+theta, ld, ld+theta. Node geometry is identical
across arms, so a difference between them is the evaluator or the route and
nothing else.

Quadratures, per arm:

    qag_mino     adaptive GSL QAG over Mino time
    panel        panels split at closest approach + Gauss-Legendre
    mino_spline  graded Mino mesh of Ntimes nodes + spline quadrature
    unif_spline  uniform-t grid of Ntimes points + spline quadrature
    kind@Nt      swept shared uniform grid, kind in {unif, fft}, via --nt-sweep

The kind@Nt rules are the ones whose cost amortizes over a domain: they sample on
one uniform t grid, so a single particle seating serves every field point, which
panel cannot do because its breakpoints move with r. fft is one FFTW transform
yielding every n at once and produces src only.

Errors are quoted against a high-order panel build in the ld arm on the dtheta
route -- deliberately the best available, so both routes are graded against it. A
decaying spectrum makes relative error meaningless once |A_n| sinks to the
roundoff of |A_0|, so every row carries |A_0| for the same output and cell and the
report suppresses rel err where |A_n| / |A_0| < --dynrange.

Rows are cached to CSV keyed on grid indices: rerun to fill gaps, --force to
recompute just the requested arms and methods, --report-only to reprint from
cache. Only the requested arms and methods are reported, but rows for any other
configuration sharing the file are preserved, never truncated.

--classes restricts the scan by field-point class (inside, peri, crossing, apo,
outside), the turning points named separately because they are the worst cells.
Adaptive quadrature costs three orders of magnitude more evaluations at a crossing
than off it, so an arm that can afford one class often cannot afford another.

--aggregate reads many caches and reports, per configuration, the Nt from which a
tolerance holds for every higher rung, against the carrier |m| wphi/wr + |n|.

    python test/test_nmode_scan.py --m 2 --n 8 --a 0.9 --p 10 --e 0.3 --ntimes 257
    python test/test_nmode_scan.py --r-scheme anchored --pad 2 --nz 4 \
        --zmin 1e-2 --zmax 0.866 --routes dtheta,theta --arms f64 \
        --nt-sweep fft --nt-min 129 --ntimes 8193
"""
import argparse
import csv
import glob
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import inspectre_c
from inspectre import Inspectre, eval_precision

# (name, block index in the (PhiS, dPhiS, src) result tuple, real-part offset)
OUTPUTS = (("Phi", 0, 0), ("drPhi", 1, 2), ("dthetaPhi", 1, 4), ("src", 2, 0))
OUTPUT_NAMES = tuple(o[0] for o in OUTPUTS)
ARMS = ("f64", "ld")
METHODS = ("qag_mino", "panel", "mino_spline", "unif_spline")
R_CLASSES = ("inside", "peri", "crossing", "apo", "outside")
# How the polar offset reaches C: "dtheta" hands over theta - pi/2 directly via
# inspectre_field_point, "theta" hands over an absolute theta that C differences,
# capping the offset's relative accuracy at eps/2 dtheta.
ROUTES = ("dtheta", "theta")
# swept shared-time-grid rules, named kind@Nt: both sample the source on one
# uniform t grid, so their cost amortizes over field points as panel's cannot.
# unif = gsl spline quadrature per n; fft = one FFTW transform giving every n at
# once (src only, which is all the FFT path produces).
GRID_KINDS = ("unif", "fft")

FIELDS = ("m", "n", "a", "p", "e", "ntimes", "order", "max_levels",
          "ref_order", "ref_levels", "epsabs", "epsrel",
          "ir", "r", "r_class", "iz", "z", "dtheta",
          "output", "arm", "method",
          "re", "im", "amp", "ref_re", "ref_im", "ref_amp", "ref_amp_n0",
          "ref_selfconv", "abs_err", "rel_err", "dyn", "seconds", "evals",
          "qag_limit", "finite")

SCALAR_KEYS = ("m", "n", "a", "p", "e", "ntimes", "order", "max_levels",
               "ref_order", "ref_levels", "epsabs", "epsrel")


def amps(res):
    """Complex amplitude per output from a (PhiS[2], dPhiS[8], src[2]) result."""
    return {name: complex(res[blk][off], res[blk][off + 1])
            for name, blk, off in OUTPUTS}


def parse_grid_method(method):
    """(kind, Nt) for a swept uniform-grid method like unif@513, else None."""
    if "@" not in method:
        return None
    kind, _, nt = method.partition("@")
    if kind not in GRID_KINDS or not nt.isdigit():
        return None
    return kind, int(nt)


def nt_ladder(nt_min, nt_max):
    """Doubling ladder of closed-grid sizes 2^k + 1 spanning [nt_min, nt_max].

    nt -> 2(nt - 1) + 1 has a fixed point at 1 and decreases below it, so a start
    of 1 or less never terminates; the floor of 3 is also the smallest grid that
    spans a period.
    """
    if nt_min < 3:
        raise ValueError(f"nt_min = {nt_min} must be at least 3")
    out, nt = [], nt_min
    while nt <= nt_max:
        out.append(nt)
        nt = 2 * (nt - 1) + 1
    return out


def nan_amps():
    return {name: complex("nan") for name in OUTPUT_NAMES}


def nan_floats():
    return {name: float("nan") for name in OUTPUT_NAMES}


def dedupe(seq):
    """Order-preserving deduplication of a comma-split option list."""
    out = []
    for s in seq:
        if s and s not in out:
            out.append(s)
    return out


def horizon(a):
    return 1.0 + math.sqrt(max(0.0, 1.0 - a * a))


def arm_base(label):
    """Evaluator half of an arm label: f64 or ld."""
    return label.split("+", 1)[0]


def arm_route(label):
    """Field-point route half of an arm label."""
    return label.split("+", 1)[1] if "+" in label else "dtheta"


def arm_label(base, route):
    return base if route == "dtheta" else f"{base}+{route}"


def classify_r(r, rmin, rmax):
    """Field-point radius relative to the libration range, turning points named.

    The turning points are called out because the integrand is worst there: the
    particle lingers at closest approach, so the peak the quadrature has to
    resolve is both tallest and least like the smooth interior case.
    """
    tol = 1e-9 * max(abs(rmax), 1.0)
    if abs(r - rmin) <= tol:
        return "peri"
    if abs(r - rmax) <= tol:
        return "apo"
    if r < rmin:
        return "inside"
    if r > rmax:
        return "outside"
    return "crossing"


def build_grid(cfg):
    """(r values, z values, dtheta values, r_class per r) for the scan.

    The anchored scheme places the six radii the worldtube actually cares about:
    pad inside periapsis, periapsis, two interior thirds of the libration range,
    apoapsis, pad outside apoapsis.
    """
    rmin = cfg.p / (1.0 + cfg.e)
    rmax = cfg.p / (1.0 - cfg.e) if cfg.e < 1.0 else float("inf")
    floor_r = 0.5 * (rmin + horizon(cfg.a))
    if cfg.r_scheme == "anchored":
        span = rmax - rmin
        if span <= 0.0:
            raise SystemExit("anchored r grid needs e > 0; the libration range "
                             "is empty at e = 0")
        rs = np.array([max(rmin - cfg.pad, floor_r), rmin,
                       rmin + span / 3.0, rmin + 2.0 * span / 3.0,
                       rmax, rmax + cfg.pad])
    else:
        rs = np.linspace(max(rmin - cfg.pad, floor_r), rmax + cfg.pad, cfg.nr)
    zs = np.logspace(math.log10(cfg.zmin), math.log10(cfg.zmax), cfg.nz)
    dths = np.array([-math.asin(z) for z in zs])
    cls = [classify_r(r, rmin, rmax) for r in rs]
    return rs, zs, dths, cls


def measure(fn):
    """(result, seconds, source evals, QAG limit hits) for one integration."""
    inspectre_c.inspectre_eval_count_reset()
    inspectre_c.inspectre_qag_limit_reset()
    t0 = time.perf_counter()
    try:
        res = fn()
    except Exception:
        res = None
    dt = time.perf_counter() - t0
    return (res, dt, inspectre_c.inspectre_eval_count(),
            inspectre_c.inspectre_qag_limit_count())


def fft_amps(out, n, N):
    """src amplitude from an fft_source_nmodes result; the FFT path yields no others."""
    if out is None:
        return nan_amps()
    re, im = out
    k = ((n % N) + N) % N
    got = nan_amps()
    got["src"] = complex(re[k], im[k])
    return got


def route_kwargs(route, dth):
    """theta_field / dtheta keywords selecting how the polar offset reaches C."""
    if route == "dtheta":
        return dict(theta_field=None, dtheta=dth)
    return dict(theta_field=math.pi / 2.0 + dth, dtheta=None)


def run_method(insp, cfg, method, r, dth, route="dtheta"):
    """Amplitudes + cost for one quadrature at one field point."""
    m, n = cfg.m, cfg.n
    rk = route_kwargs(route, dth)

    grid = parse_grid_method(method)
    if grid is not None:
        kind, nt = grid
        if kind == "fft":
            res, dt, evals, qlim = measure(
                lambda: insp.fft_source_nmodes_fast(m, r, N=nt, **rk))
            return fft_amps(res, n, nt), dt, evals, qlim
        res, dt, evals, qlim = measure(
            lambda: insp.integrate_nmode_fast(
                m, n, r, mode=inspectre_c.INSPECTRE_INTEG_SPLINE, nSamples=nt,
                epsabs=cfg.epsabs, epsrel=cfg.epsrel, **rk))
        return (nan_amps() if res is None else amps(res)), dt, evals, qlim

    if method == "qag_mino":
        def fn():
            return insp.integrate_nmode_fast(
                m, n, r, mode=inspectre_c.INSPECTRE_INTEG_QAG_MINO,
                epsabs=cfg.epsabs, epsrel=cfg.epsrel, **rk)
    elif method == "mino_spline":
        def fn():
            return insp.integrate_nmode_fast(
                m, n, r, mode=inspectre_c.INSPECTRE_INTEG_MINO_SPLINE,
                nSamples=cfg.ntimes, epsabs=cfg.epsabs, epsrel=cfg.epsrel, **rk)
    elif method == "unif_spline":
        def fn():
            return insp.integrate_nmode_fast(
                m, n, r, mode=inspectre_c.INSPECTRE_INTEG_SPLINE,
                nSamples=cfg.ntimes, epsabs=cfg.epsabs, epsrel=cfg.epsrel, **rk)
    elif method == "panel":
        def fn():
            return insp.panel_nmodes_fast(
                m, [n], r, order=cfg.order, max_levels=cfg.max_levels, **rk)[0]
    else:
        raise ValueError(method)

    res, dt, evals, qlim = measure(fn)
    return (nan_amps() if res is None else amps(res)), dt, evals, qlim


def run_reference(insp, cfg, r, dth):
    """(A_n, A_0, self-convergence, seconds, evals) from panel builds in the ld arm.

    A_0 comes from the same build and only sets the dynamic-range guard; n = 0 is
    the largest amplitude of a decaying radial spectrum, so |A_n| / |A_0| says
    whether the requested mode is above the reference's own roundoff.

    Self-convergence is the reference against a coarser build, order - 8 and
    max_levels - 4 together: a deliberately conservative floor, since the panel
    error has both a Gauss-Legendre-order part and a refinement-depth part and a
    floor that moves only one knob can miss the other. A quoted method error at
    or under this floor says only that the two builds agree.
    """
    n_list = [cfg.n] if cfg.n == 0 else [cfg.n, 0]

    def build(order, levels):
        return lambda: insp.panel_nmodes_fast(
            cfg.m, n_list, r, None, dtheta=dth, order=order, max_levels=levels)

    with eval_precision(True):
        res, dt, evals, _ = measure(build(cfg.ref_order, cfg.ref_levels))
        lo = None
        if cfg.ref_check:
            lo, dt2, ev2, _ = measure(
                build(max(4, cfg.ref_order - 8), max(8, cfg.ref_levels - 4)))
            dt += dt2
            evals += ev2
    if res is None:
        return nan_amps(), nan_amps(), nan_floats(), dt, evals
    a_n = amps(res[0])
    a_0 = a_n if cfg.n == 0 else amps(res[-1])
    if lo is None:
        sc = nan_floats()
    else:
        b_n = amps(lo[0])
        sc = {name: (abs(a_n[name] - b_n[name]) / abs(a_n[name])
                     if abs(a_n[name]) > 0.0 else float("nan"))
              for name in OUTPUT_NAMES}
    return a_n, a_0, sc, dt, evals


def ref_from_rows(cell_rows, ref_check):
    """Reference triple recovered from cached rows for a cell, else None.

    The reference is method- and arm-independent, so an incremental fill that
    adds one more quadrature must not pay for it again. Rows written under
    --no-ref-check carry no self-convergence floor, so they are refused once the
    check is asked for -- otherwise the NaN propagates to every row added later
    and the floor can never be filled in without discarding the whole cache.
    """
    if not cell_rows:
        return None
    ref, ref0, refsc = {}, {}, {}
    for name in OUTPUT_NAMES:
        hit = next((r for r in cell_rows if r["output"] == name), None)
        if hit is None:
            return None
        if ref_check and not np.isfinite(hit["ref_selfconv"]):
            return None
        ref[name] = complex(hit["ref_re"], hit["ref_im"])
        ref0[name] = complex(hit["ref_amp_n0"], 0.0)
        refsc[name] = hit["ref_selfconv"]
    return ref, ref0, refsc


REF_COLS = ("ref_re", "ref_im", "ref_amp", "ref_amp_n0", "ref_selfconv")


def restamp_reference(rows, ir, iz, fresh):
    """Carry a recomputed reference onto the cell's already-cached rows.

    A cell whose reference was rebuilt -- because the cached one lacked the
    self-convergence floor -- would otherwise hold two references at once, old
    rows graded against one and new rows against the other.
    """
    by_output = {}
    for row in fresh:
        by_output.setdefault(row["output"], row)
    for row in rows:
        if int(row["ir"]) != ir or int(row["iz"]) != iz:
            continue
        src = by_output.get(row["output"])
        if src is None:
            continue
        for col in REF_COLS:
            row[col] = src[col]
        ra = row["ref_amp"]
        row["abs_err"] = abs(complex(row["re"], row["im"])
                             - complex(row["ref_re"], row["ref_im"]))
        row["rel_err"] = row["abs_err"] / ra if ra > 0.0 else float("nan")
        r0a = row["ref_amp_n0"]
        row["dyn"] = ra / r0a if r0a > 0.0 else float("nan")


def run_cell(insp, cfg, ir, r, r_class, iz, z, dth, want, cached_ref=None):
    """Rows for one field point; `want` is the set of (arm, method) still needed."""
    if cached_ref is not None:
        ref, ref0, refsc = cached_ref
        ref_dt, ref_evals = 0.0, 0
    else:
        ref, ref0, refsc, ref_dt, ref_evals = run_reference(insp, cfg, r, dth)
    rows = []
    for arm in cfg.arms:
        with eval_precision(arm_base(arm) == "ld"):
            for method in cfg.methods:
                if (arm, method) not in want:
                    continue
                got, dt, evals, qlim = run_method(insp, cfg, method, r, dth,
                                                  route=arm_route(arm))
                for name in OUTPUT_NAMES:
                    v, rv, r0 = got[name], ref[name], ref0[name]
                    ra, r0a = abs(rv), abs(r0)
                    ae = abs(v - rv)
                    rows.append(dict(
                        m=cfg.m, n=cfg.n, a=cfg.a, p=cfg.p, e=cfg.e,
                        ntimes=cfg.ntimes, order=cfg.order,
                        max_levels=cfg.max_levels, ref_order=cfg.ref_order,
                        ref_levels=cfg.ref_levels, epsabs=cfg.epsabs,
                        epsrel=cfg.epsrel,
                        ir=ir, r=r, r_class=r_class, iz=iz, z=z, dtheta=dth,
                        output=name, arm=arm, method=method,
                        re=v.real, im=v.imag, amp=abs(v),
                        ref_re=rv.real, ref_im=rv.imag, ref_amp=ra,
                        ref_amp_n0=r0a, ref_selfconv=refsc[name],
                        abs_err=ae,
                        rel_err=(ae / ra if ra > 0.0 else float("nan")),
                        dyn=(ra / r0a if r0a > 0.0 else float("nan")),
                        seconds=dt, evals=evals, qag_limit=qlim,
                        finite=int(np.isfinite(v.real) and np.isfinite(v.imag)),
                    ))
    if cfg.verbose:
        print(f"  ref {ref_dt:7.3f}s {ref_evals:7d} evals   "
              f"|A_n|/|A_0| src = {abs(ref['src']) / max(abs(ref0['src']), 1e-300):.2e}",
              file=sys.stderr)
    return rows


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

STR_FIELDS = ("r_class", "output", "arm", "method")


def row_matches(raw, cfg, rs, zs):
    """True when a raw CSV row belongs to this configuration and grid."""
    if any(float(raw[k]) != float(getattr(cfg, k)) for k in SCALAR_KEYS):
        return False
    ir, iz = int(raw["ir"]), int(raw["iz"])
    if not (0 <= ir < len(rs) and 0 <= iz < len(zs)):
        return False
    return (abs(float(raw["r"]) - rs[ir]) <= 1e-12 * rs[ir]
            and abs(float(raw["z"]) - zs[iz]) <= 1e-12 * zs[iz])


def load_cache(path, cfg, rs, zs):
    """(rows for this configuration, foreign rows to carry through unchanged).

    The cache path is keyed on (m, n, a, p, e, Ntimes) only, so a change of
    --order, --ref-order or the grid lands in the same file while invalidating
    every row in it. Those rows are kept and written back verbatim: they answer a
    different question, they are not garbage, and a scan costs hours.

    A header that is not exactly FIELDS is from an older revision of this script;
    it is moved aside rather than parsed or truncated.
    """
    if not os.path.exists(path):
        return [], []
    mine, others, corrupt = [], [], 0
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        if rd.fieldnames != list(FIELDS):
            os.replace(path, path + ".bak")
            print(f"cache header mismatch; previous file moved to "
                  f"{os.path.basename(path)}.bak", file=sys.stderr)
            return [], []
        for raw in rd:
            row = dict(raw)
            try:
                if not row_matches(row, cfg, rs, zs):
                    others.append(row)
                    continue
                for k in FIELDS:
                    if k not in STR_FIELDS:
                        row[k] = float(row[k])
                row["ir"], row["iz"] = int(row["ir"]), int(row["iz"])
            except (TypeError, ValueError):
                corrupt += 1
                continue
            mine.append(row)
    if corrupt:
        print(f"dropped {corrupt} unparseable cache rows", file=sys.stderr)
    seen, dedup = set(), []
    for row in reversed(mine):
        k = (row["ir"], row["iz"], row["output"], row["arm"], row["method"])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(row)
    dedup.reverse()
    if len(dedup) != len(mine):
        print(f"dropped {len(mine) - len(dedup)} duplicate cache rows",
              file=sys.stderr)
    return dedup, others


def save_cache(path, rows, others):
    """Rewrite the cache atomically; a scan is long enough to be interrupted."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for row in list(rows) + list(others):
            w.writerow({k: row[k] for k in FIELDS})
    os.replace(tmp, path)


def scan(cfg, path):
    """Fill the cache for every (cell, arm, method) not already present."""
    rs, zs, dths, cls = build_grid(cfg)
    rows, others = load_cache(path, cfg, rs, zs)
    if cfg.force:
        req = {(arm, meth) for arm in cfg.arms for meth in cfg.methods}
        rows = [r for r in rows if (r["arm"], r["method"]) not in req]
    dead = [r for r in rows if not np.isfinite(r["ref_amp"])]
    if dead:
        rows = [r for r in rows if np.isfinite(r["ref_amp"])]
        print(f"discarding {len(dead)} rows whose reference build failed; "
              f"those cells will be retried", file=sys.stderr)
    have = {(int(r["ir"]), int(r["iz"]), r["arm"], r["method"]) for r in rows}

    if cfg.report_only:
        return rows, rs, zs, dths, cls

    insp = Inspectre(spin=cfg.a, semilatus_rectum=cfg.p, eccentricity=cfg.e)
    total = len(rs) * len(zs)
    done = 0
    for ir, r in enumerate(rs):
        for iz, z in enumerate(zs):
            done += 1
            if cls[ir] not in cfg.classes:
                continue
            want = {(arm, meth) for arm in cfg.arms for meth in cfg.methods
                    if (ir, iz, arm, meth) not in have}
            if not want:
                continue
            if cfg.verbose:
                print(f"[{done}/{total}] r = {r:.6f} ({cls[ir]})  z = {z:.3e}  "
                      f"dtheta = {dths[iz]:.3e}  {len(want)} runs",
                      file=sys.stderr)
            cached_ref = ref_from_rows(
                [x for x in rows if int(x["ir"]) == ir and int(x["iz"]) == iz],
                cfg.ref_check)
            fresh = run_cell(insp, cfg, ir, r, cls[ir], iz, z, dths[iz], want,
                             cached_ref=cached_ref)
            if cached_ref is None:
                restamp_reference(rows, ir, iz, fresh)
            rows += fresh
            save_cache(path, rows, others)
    save_cache(path, rows, others)
    return rows, rs, zs, dths, cls


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def stat(vals):
    """(median, p90, max) of the finite entries, or NaNs when there are none."""
    v = np.array([x for x in vals if np.isfinite(x)])
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(v)), float(np.percentile(v, 90)), float(v.max())


def paired_gain(good, name, meth, route="dtheta"):
    """(median f64 / median ld, shared cell count) over cells both arms reached.

    An arm that could only afford part of the grid -- adaptive quadrature at a
    crossing costs three orders more evaluations than off it -- would otherwise
    have its median compared against the other arm's median over a different set
    of field points, which is not a ratio of anything.
    """
    def by_cell(arm):
        return {(r["ir"], r["iz"]): r["rel_err"] for r in good
                if r["output"] == name and r["method"] == meth
                and r["arm"] == arm}

    f = by_cell(arm_label("f64", route))
    g = by_cell(arm_label("ld", route))
    keys = [k for k in f if k in g]
    if not keys:
        return float("nan"), 0
    mf = float(np.median([f[k] for k in keys]))
    mg = float(np.median([g[k] for k in keys]))
    return (mf / mg if mg > 0.0 else float("nan")), len(keys)


def nt_required(good, kind, name, cell, target, arm):
    """Smallest ladder Nt from which `target` holds for every higher rung, else None.

    The convergence is not monotone in Nt -- a rung can meet a target by
    cancellation and the next rung miss it -- so the test is trailing rather than
    first-meeting: Nt qualifies only if it and every larger rung in the ladder
    meet the target. The top rung is therefore taken on trust, which is the one
    thing a finite ladder cannot check.

    `arm` is required, not optional. Pooling arms would return the minimum over
    evaluators and routes, which reads as the requirement for whichever arm the
    caller had in mind and is not.
    """
    rungs = sorted(
        (parse_grid_method(r["method"])[1], r["rel_err"]) for r in good
        if r["output"] == name and (int(r["ir"]), int(r["iz"])) == cell
        and r["arm"] == arm and parse_grid_method(r["method"])
        and parse_grid_method(r["method"])[0] == kind)
    if not rungs:
        return None
    qualifies = None
    for nt, err in reversed(rungs):
        if err <= target:
            qualifies = nt
        else:
            break
    return qualifies


def nt_knee(good, kind, name, cell, arm, factor=4.0, tail=3):
    """(knee Nt, floor) where extra sampling stops paying at one cell.

    The floor is the median error over the top `tail` rungs and the knee is the
    first rung within `factor` of it. The median, not the minimum: the error is
    non-monotone at the bottom of the ladder, so a minimum picks whichever rung
    dipped lowest by luck and no earlier rung can come within a factor of it,
    which drives every knee to the top of the ladder.

    This is the number to size a production grid by. Past the knee the error is set
    by the evaluator, not the sampling, and a tolerance below the floor is
    unreachable at any Nt -- asking which Nt meets 1e-6 of a cell whose floor is
    7e-7 returns whichever rung dipped under by luck, which reads as a sampling
    requirement and is not one.
    """
    rungs = sorted(
        (parse_grid_method(r["method"])[1], r["rel_err"]) for r in good
        if r["output"] == name and (int(r["ir"]), int(r["iz"])) == cell
        and r["arm"] == arm and parse_grid_method(r["method"])
        and parse_grid_method(r["method"])[0] == kind)
    if len(rungs) < tail:
        return None, float("nan")
    floor = float(np.median([e for _nt, e in rungs[-tail:]]))
    for nt, err in rungs:
        if err <= factor * floor:
            return nt, floor
    return rungs[-1][0], floor


def cells_for(good, kind, name, arm):
    """Cells that actually carry a swept row for this (kind, output, arm).

    Scoping to the (kind, output, arm) under discussion matters: `dyn` is computed
    per output, so a cell can clear the dynamic-range guard for one output and not
    another. A cell union taken over all outputs would count those as unmet.
    """
    return sorted({(int(r["ir"]), int(r["iz"])) for r in good
                   if r["output"] == name and r["arm"] == arm
                   and parse_grid_method(r["method"])
                   and parse_grid_method(r["method"])[0] == kind})


def report_nt_required(good, rows, cfg, kinds):
    """Uniform shared-grid point count needed per cell, and the region maximum."""
    ladder_top = max((parse_grid_method(m)[1] for m in cfg.methods
                      if parse_grid_method(m)), default=cfg.ntimes)
    print()
    print(f"-- shared grid: Nt from which rel err <= {cfg.nt_target:.0e} holds "
          f"for all higher rungs (top {ladder_top}) --")
    print(f"  {'kind':<6} {'output':<10} {'arm':<12} {'Nt_max':>7} {'Nt_med':>7} "
          f"{'cells':>9}  worst cell")
    for kind in kinds:
        names = ("src",) if kind == "fft" else OUTPUT_NAMES
        for name in names:
            for arm in cfg.arms:
                cells = cells_for(good, kind, name, arm)
                if not cells:
                    continue
                got = {c: nt_required(good, kind, name, c, cfg.nt_target, arm)
                       for c in cells}
                hit = {c: v for c, v in got.items() if v is not None}
                miss = [c for c, v in got.items() if v is None]
                pre = f"  {kind:<6} {name:<10} {arm:<12}"
                if hit:
                    worst = max(hit, key=lambda c: hit[c])
                    wrow = next(r for r in good
                                if (int(r["ir"]), int(r["iz"])) == worst
                                and r["output"] == name and r["arm"] == arm)
                    print(f"{pre} {max(hit.values()):>7d} "
                          f"{int(np.median(list(hit.values()))):>7d} "
                          f"{len(hit):4d}/{len(cells):<4d} "
                          f" r = {wrow['r']:.4f} ({wrow['r_class']}) "
                          f"z = {wrow['z']:.2e}")
                else:
                    print(f"{pre} {'unmet':>7} {'':>7} {0:4d}/{len(cells):<4d}")
                if miss:
                    ex = next((r for r in good
                               if (int(r["ir"]), int(r["iz"])) == miss[0]
                               and r["output"] == name and r["arm"] == arm), None)
                    where = (f"r = {ex['r']:.4f} ({ex['r_class']}) "
                             f"z = {ex['z']:.2e}") if ex else "unknown cell"
                    print(f"  {'':<6} {'':<10} {'':<12} {len(miss)} cells never "
                          f"settle below the target, e.g. {where}")


def usable(rows, cfg):
    return [r for r in rows
            if r["finite"] and np.isfinite(r["rel_err"])
            and np.isfinite(r["dyn"]) and r["dyn"] >= cfg.dynrange]


def report(rows, cfg, rs, zs, cls):
    rows = [r for r in rows
            if r["method"] in cfg.methods and r["arm"] in cfg.arms
            and r["r_class"] in cfg.classes]
    if not rows:
        print("no rows; nothing to report")
        return 0

    rmin = cfg.p / (1.0 + cfg.e)
    rmax = cfg.p / (1.0 - cfg.e)
    print("=" * 78)
    print(f"mn-mode scan   m = {cfg.m}  n = {cfg.n}   "
          f"a = {cfg.a}  p = {cfg.p}  e = {cfg.e}   Ntimes = {cfg.ntimes}")
    print(f"  r in [{rs[0]:.4f}, {rs[-1]:.4f}] over {len(rs)} points   "
          f"(r_min = {rmin:.4f}, r_max = {rmax:.4f}, pad = {cfg.pad})")
    print(f"  z = cos(theta) in [{zs[0]:.2e}, {zs[-1]:.2e}] over {len(zs)} "
          f"points, log spaced, entered as dtheta = -arcsin(z)")
    print(f"  reference: panel order = {cfg.ref_order}, "
          f"max_levels = {cfg.ref_levels}, ld arm"
          + ("" if cfg.ref_check else "  (self-convergence check off)"))
    print(f"  rel err suppressed where |A_n| / |A_0| < {cfg.dynrange:.0e}")
    print(f"  {len(rows)} rows, {len(rows) // (len(OUTPUT_NAMES) or 1)} "
          f"(cell, arm, method) runs")

    good = usable(rows, cfg)

    print()
    print("-- reference self-convergence (order vs order - 8, ld arm) ------------")
    print(f"{'output':<10} {'median':>10} {'p90':>10} {'max':>10}")
    for name in OUTPUT_NAMES:
        med, p90, mx = stat([r["ref_selfconv"] for r in good
                             if r["output"] == name and r["method"] == cfg.methods[0]
                             and r["arm"] == cfg.arms[0]])
        print(f"{name:<10} {med:10.2e} {p90:10.2e} {mx:10.2e}")
    print("  a method error at or below this is a floor, not a measurement")

    print()
    print("-- accuracy vs reference, relative -------------------------------------")
    print(f"{'output':<10} {'method':<12} {'arm':<12} "
          f"{'median':>10} {'p90':>10} {'max':>10} {'cells':>6} {'supp':>5} "
          f"{'<=ref':>6}")
    for name in OUTPUT_NAMES:
        for meth in cfg.methods:
            for arm in cfg.arms:
                sel = [r for r in good if r["output"] == name
                       and r["method"] == meth and r["arm"] == arm]
                allsel = [r for r in rows if r["output"] == name
                          and r["method"] == meth and r["arm"] == arm]
                med, p90, mx = stat([r["rel_err"] for r in sel])
                floored = sum(1 for r in sel
                              if np.isfinite(r["ref_selfconv"])
                              and r["rel_err"] <= r["ref_selfconv"])
                print(f"{name:<10} {meth:<12} {arm:<12} "
                      f"{med:10.2e} {p90:10.2e} {mx:10.2e} "
                      f"{len(sel):6d} {len(allsel) - len(sel):5d} {floored:6d}")
        print()

    bases = {arm_base(a) for a in cfg.arms}
    if len(bases) < 2:
        print(f"-- extended precision gain: only the "
              f"{sorted(bases)[0]} evaluator was run, no ratio ---")
    else:
        report_gain(good, cfg)

    if len(cfg.routes) > 1:
        report_routes(good, cfg)

    print()
    print("-- cost per amplitude -------------------------------------------------")
    print(f"{'method':<12} {'arm':<12} {'seconds':>10} {'evals':>9} {'qag_lim':>8}")
    for meth in cfg.methods:
        for arm in cfg.arms:
            sel = [r for r in rows if r["method"] == meth and r["arm"] == arm
                   and r["output"] == "src"]
            if not sel:
                continue
            print(f"{meth:<12} {arm:<12} "
                  f"{np.mean([r['seconds'] for r in sel]):10.4f} "
                  f"{np.mean([r['evals'] for r in sel]):9.0f} "
                  f"{int(sum(r['qag_limit'] for r in sel)):8d}")

    swept0 = sorted({parse_grid_method(m)[0] for m in cfg.methods
                     if parse_grid_method(m)})
    if swept0:
        report_nt_required(good, rows, cfg, swept0)
    report_trends(good, rows, cfg, rs, zs, cls)
    return report_tail(good, rows, cfg)


def route_offset_error(z):
    """(|delta dtheta| the absolute-theta route introduces, relative size) at this z.

    The theta route hands C fl(pi/2 + dtheta) and C returns fl(theta - pi/2). This
    measures that round trip rather than quoting eps/2 dtheta, because the round
    trip is frequently exact and the estimate is then badly wrong: by Sterbenz it
    is exact for every |dtheta| >= pi/4, i.e. every z >= sin(pi/4) = 0.7071, and
    exact by coincidence for about 7.5% of z below that. A route table that prints
    an expected scale next to a measured 0.00e+00 invites the reader to conclude
    the route was tested and found harmless where it was not exercised at all.
    """
    dth = -math.asin(z)
    back = (math.pi / 2.0 + dth) - math.pi / 2.0
    delta = abs(back - dth)
    return delta, (delta / abs(dth) if dth != 0.0 else float("nan"))


def report_routes(good, cfg):
    """Amplitude difference between the dtheta and absolute-theta field points.

    Measured as |A(theta) - A(dtheta)| / |A(dtheta)| on the same cell and rule, so
    it isolates the route from the quadrature and from the reference. Rows where
    the two routes are the same computation bit for bit are labelled, not left to
    look like a measured null result.
    """
    print()
    print("-- field-point route: |A(theta) - A(dtheta)| / |A(dtheta)| ------------")
    print(f"{'output':<10} {'evaluator':<10} {'z':>10} "
          f"{'median':>11} {'max':>11} {'d(dth)/dth':>11} {'cells':>6}  note")
    for name in OUTPUT_NAMES:
        for base in sorted({arm_base(a) for a in cfg.arms}):
            la, lb = arm_label(base, "dtheta"), arm_label(base, "theta")
            byz, zof = {}, {}
            for r in good:
                if r["output"] != name or r["arm"] not in (la, lb):
                    continue
                iz = int(r["iz"])
                byz.setdefault(iz, {})[(int(r["ir"]), r["method"], r["arm"])] = r
                zof[iz] = r["z"]
            for iz in sorted(byz):
                z = zof[iz]
                rel = []
                for (ir, meth, arm), row in byz[iz].items():
                    if arm != la:
                        continue
                    other = byz[iz].get((ir, meth, lb))
                    if other is None:
                        continue
                    a0 = complex(row["re"], row["im"])
                    a1 = complex(other["re"], other["im"])
                    if abs(a0) > 0.0:
                        rel.append(abs(a1 - a0) / abs(a0))
                if not rel:
                    continue
                med, _p90, mx = stat(rel)
                delta, relative = route_offset_error(z)
                note = ("routes identical: pi/2 + dtheta round trips exactly, "
                        "so this row tests nothing"
                        if delta == 0.0 else "")
                print(f"{name:<10} {base:<10} {z:10.2e} "
                      f"{med:11.2e} {mx:11.2e} {relative:11.2e} "
                      f"{len(rel):6d}  {note}")


def report_gain(good, cfg):
    print("-- extended precision gain (median rel err f64 / ld) ------------------")
    print(f"{'output':<10} {'route':<8} " + "".join(f"{m:>18}"
                                                    for m in cfg.methods))
    for name in OUTPUT_NAMES:
        for route in cfg.routes:
            line = f"{name:<10} {route:<8} "
            for meth in cfg.methods:
                ratio, ncell = paired_gain(good, name, meth, route)
                line += f"{ratio:>13.2f}[{ncell:>3d}]"
            print(line)
    print("  [n] is the cell count the two arms share; a ratio over cells only "
          "one arm reached would compare different grids")


def report_trends(good, rows, cfg, rs, zs, cls):
    print()
    print("-- rel err vs z (median over r) ---------------------------------------")
    for name in cfg.trend_outputs:
        print(f"  {name}")
        head = f"{'z':>10} " + "".join(f"{m + '/' + a:>16}"
                                      for m in cfg.methods for a in cfg.arms)
        print(head)
        for iz, z in enumerate(zs):
            line = f"{z:10.2e} "
            for meth in cfg.methods:
                for arm in cfg.arms:
                    sel = [r for r in good if r["output"] == name
                           and r["method"] == meth and r["arm"] == arm
                           and int(r["iz"]) == iz]
                    line += f"{stat([r['rel_err'] for r in sel])[0]:>16.2e}"
            print(line)
        print()

    print("-- rel err vs field-point class (median over grid) --------------------")
    classes = [c for c in R_CLASSES if c in cls]
    print(f"{'output':<10} {'method':<12} {'arm':<12} "
          + "".join(f"{c:>12}" for c in classes))
    for name in cfg.trend_outputs:
        for meth in cfg.methods:
            for arm in cfg.arms:
                line = f"{name:<10} {meth:<12} {arm:<12} "
                for c in classes:
                    sel = [r for r in good if r["output"] == name
                           and r["method"] == meth and r["arm"] == arm
                           and r["r_class"] == c]
                    line += f"{stat([r['rel_err'] for r in sel])[0]:>12.2e}"
                print(line)


def report_tail(good, rows, cfg):
    print()
    print("-- worst cell per output ----------------------------------------------")
    for name in OUTPUT_NAMES:
        sel = [r for r in good if r["output"] == name]
        if not sel:
            print(f"{name:<10} no usable cell above the dynamic-range guard")
            continue
        w = max(sel, key=lambda r: r["rel_err"])
        print(f"{name:<10} {w['rel_err']:.2e}  {w['method']}/{w['arm']}  "
              f"r = {w['r']:.6f} ({w['r_class']})  z = {w['z']:.2e}  "
              f"|A_n| = {w['ref_amp']:.3e}  |A_n|/|A_0| = {w['dyn']:.2e}")

    nonfin = [r for r in rows if not r["finite"]]
    if nonfin:
        print()
        print(f"-- {len(nonfin)} non-finite amplitudes ---------------------------")
        seen = set()
        for r in nonfin:
            k = (r["method"], r["arm"])
            if k in seen:
                continue
            seen.add(k)
            print(f"  {r['method']}/{r['arm']} first at r = {r['r']:.6f} "
                  f"z = {r['z']:.2e} ({r['output']})")

    if cfg.assert_rel is not None:
        bad = [r for r in good if r["rel_err"] > cfg.assert_rel]
        print()
        print(f"assert rel_err <= {cfg.assert_rel:.1e}: "
              f"{len(bad)} of {len(good)} usable rows fail")
        return 1 if bad else 0
    return 0


def aggregate(pattern, target, kind, name, arm):
    """Nt required across every cache matching `pattern`, against the carrier.

    The uniform grid must resolve the phase exp(i(m wphi + n wr) t) over one
    radial period, which is 2 pi (|m| wphi/wr + |n|) of phase, so that combination
    and not m or n alone is the natural abscissa. Prints the per-configuration
    requirement and the points-per-carrier-cycle it implies.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"no caches match {pattern}")
        return 1
    print(f"-- Nt for {kind} {name} on arm {arm}, rel err <= {target:.0e} over "
          f"{len(paths)} configurations --")
    print(f"{'a':>6} {'p':>6} {'e':>6} {'m':>4} {'n':>4} {'carrier':>8} "
          f"{'Nt_max':>7} {'Nt_med':>7} {'cells':>8} {'knee':>7} "
          f"{'floor_med':>10} {'floor_max':>10}")
    fits = []
    for path in paths:
        with open(path, newline="") as fh:
            rows = [r for r in csv.DictReader(fh)]
        if not rows:
            continue
        for k in FIELDS:
            if k not in STR_FIELDS:
                for r in rows:
                    r[k] = float(r[k])
        head = rows[0]
        a, p, e = head["a"], head["p"], head["e"]
        m, n = int(head["m"]), int(head["n"])
        keys = {(r["a"], r["p"], r["e"], r["m"], r["n"], r["ntimes"],
                 r["ref_order"]) for r in rows}
        if len(keys) > 1:
            rows = [r for r in rows
                    if (r["a"], r["p"], r["e"], r["m"], r["n"], r["ntimes"],
                        r["ref_order"]) == (head["a"], head["p"], head["e"],
                                            head["m"], head["n"],
                                            head["ntimes"], head["ref_order"])]
            print(f"  note {os.path.basename(path)} holds {len(keys)} "
                  f"configurations; keeping the first only")
        insp = Inspectre(spin=a, semilatus_rectum=p, eccentricity=e)
        ratio = insp.omega_phi / insp.omega_r
        carrier = abs(m) * ratio + abs(n)
        good = [r for r in rows
                if r["finite"] and np.isfinite(r["rel_err"])
                and np.isfinite(r["dyn"]) and r["dyn"] >= 1e-13]
        cells = cells_for(good, kind, name, arm)
        if not cells:
            print(f"{a:6.2f} {p:6.2f} {e:6.2f} {m:4d} {n:4d} {ratio:9.3f} "
                  f"{carrier:9.2f} {'no rows':>7}")
            continue
        got = [nt_required(good, kind, name, c, target, arm) for c in cells]
        hit = [v for v in got if v is not None]
        knees, floors = [], []
        for c in cells:
            k, fl = nt_knee(good, kind, name, c, arm)
            if k is not None:
                knees.append(k)
                floors.append(fl)
        knee = max(knees) if knees else 0
        fmed = float(np.median(floors)) if floors else float("nan")
        fmax = float(np.max(floors)) if floors else float("nan")
        pre = (f"{a:6.2f} {p:6.2f} {e:6.2f} {m:4d} {n:4d} {carrier:8.2f} ")
        tail = (f"{knee:7d} {fmed:10.1e} {fmax:10.1e}")
        if not hit:
            print(f"{pre}{'unmet':>7} {'':>7} {0:4d}/{len(cells):<3d} {tail}")
            continue
        ntmax = max(hit)
        print(f"{pre}{ntmax:7d} {int(np.median(hit)):7d} "
              f"{len(hit):4d}/{len(cells):<3d} {tail}")
        if len(hit) == len(cells):
            fits.append((carrier, ntmax))
    if len(fits) < 3:
        print("  fewer than 3 configurations met the target on every cell; "
              "no fit")
        return 0
    print(f"  fit uses the {len(fits)} configurations that met the target on "
          f"every cell; a configuration with unmet cells has an Nt_max "
          f"conditioned on the cells that happened to converge")
    c = np.array([f[0] for f in fits])
    nt = np.array([f[1] for f in fits])
    print(f"  peak-resolution floor (min Nt_max over configurations): "
          f"{int(nt.min())}")
    if c.max() / max(c.min(), 1e-30) < 3.0:
        print(f"  carrier spans only {c.min():.2f}-{c.max():.2f}; too narrow to "
              f"separate the carrier term from the floor -- vary m or n")
        return 0
    slope, icept = np.polyfit(c, nt, 1)
    print(f"  least squares Nt_max = {slope:.1f} * carrier + {icept:.0f}"
          f"   (carrier = |m| wphi/wr + |n|)")
    print(f"  worst points per carrier cycle: "
          f"{float((nt / np.maximum(c, 1e-30)).max()):.1f}")
    return 0


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--a", type=float, default=0.9)
    ap.add_argument("--p", type=float, default=10.0)
    ap.add_argument("--e", type=float, default=0.3)
    ap.add_argument("--ntimes", type=int, default=257)

    ap.add_argument("--nr", type=int, default=5)
    ap.add_argument("--r-scheme", dest="r_scheme", default="linspace",
                    choices=("linspace", "anchored"))
    ap.add_argument("--routes", default="dtheta")
    ap.add_argument("--pad", type=float, default=0.5)
    ap.add_argument("--nz", type=int, default=4)
    ap.add_argument("--zmin", type=float, default=1e-10)
    ap.add_argument("--zmax", type=float, default=1e-2)

    ap.add_argument("--order", type=int, default=16)
    ap.add_argument("--max-levels", dest="max_levels", type=int, default=40)
    ap.add_argument("--ref-order", dest="ref_order", type=int, default=40)
    ap.add_argument("--ref-levels", dest="ref_levels", type=int, default=52)
    ap.add_argument("--epsabs", type=float, default=1e-10)
    ap.add_argument("--epsrel", type=float, default=1e-10)
    ap.add_argument("--no-ref-check", dest="ref_check", action="store_false")

    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--nt-sweep", dest="nt_sweep", default=None,
                    help="comma list of GRID_KINDS; replaces --methods with the "
                         "kind@Nt ladder from --nt-min up to --ntimes")
    ap.add_argument("--nt-min", dest="nt_min", type=int, default=129)
    ap.add_argument("--nt-target", dest="nt_target", type=float, default=1e-6)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--trend-outputs", dest="trend_outputs", default="Phi,src")
    ap.add_argument("--classes", default=",".join(R_CLASSES))
    ap.add_argument("--dynrange", type=float, default=1e-13)
    ap.add_argument("--assert-rel", dest="assert_rel", type=float, default=None)

    ap.add_argument("--aggregate", default=None,
                    help="glob of caches; report Nt required per configuration "
                         "against the carrier instead of scanning")
    ap.add_argument("--agg-kind", dest="agg_kind", default="fft")
    ap.add_argument("--agg-output", dest="agg_output", default="src")
    ap.add_argument("--agg-arm", dest="agg_arm", default="f64")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", dest="report_only", action="store_true")
    ap.add_argument("--quiet", dest="verbose", action="store_false")

    cfg = ap.parse_args(argv)
    if cfg.nt_sweep:
        kinds = dedupe(cfg.nt_sweep.split(","))
        for k in kinds:
            if k not in GRID_KINDS:
                ap.error(f"unknown grid kind {k}; choose from {GRID_KINDS}")
        ladder = nt_ladder(cfg.nt_min, cfg.ntimes)
        if not ladder:
            ap.error("--nt-min exceeds --ntimes; no ladder to sweep")
        cfg.methods = ",".join(f"{k}@{nt}" for k in kinds for nt in ladder)
    cfg.methods = dedupe(cfg.methods.split(","))
    cfg.routes = dedupe(cfg.routes.split(","))
    for rt in cfg.routes:
        if rt not in ROUTES:
            ap.error(f"unknown route {rt}; choose from {ROUTES}")
    bases = dedupe(cfg.arms.split(","))
    for base in bases:
        if base not in ARMS:
            ap.error(f"unknown arm {base}; choose from {ARMS}")
    cfg.arms = [arm_label(b, rt) for b in bases for rt in cfg.routes]
    if cfg.r_scheme == "anchored":
        cfg.nr = 6
    cfg.trend_outputs = dedupe(cfg.trend_outputs.split(","))
    cfg.classes = dedupe(cfg.classes.split(","))
    if not cfg.methods or not cfg.arms or not cfg.trend_outputs or not cfg.classes:
        ap.error("--methods, --arms, --trend-outputs and --classes "
                 "must be non-empty")
    for c in cfg.classes:
        if c not in R_CLASSES:
            ap.error(f"unknown class {c}; choose from {R_CLASSES}")
    if cfg.nr < 1 or cfg.nz < 1:
        ap.error("--nr and --nz must be at least 1")
    if cfg.pad <= 0.0:
        ap.error("--pad must be positive, or the anchored radii collide")
    if cfg.nt_min < 3:
        ap.error("--nt-min must be at least 3")
    if cfg.agg_kind not in GRID_KINDS:
        ap.error(f"--agg-kind must be one of {GRID_KINDS}")
    if cfg.agg_output not in OUTPUT_NAMES:
        ap.error(f"--agg-output must be one of {OUTPUT_NAMES}")
    for meth in cfg.methods:
        grid = parse_grid_method(meth)
        if grid is not None and grid[1] < 3:
            ap.error(f"{meth}: a grid needs at least 3 points to span a period")
    if cfg.ntimes < 3:
        ap.error("--ntimes must be at least 3 for a sample-based rule")
    if cfg.order < 2 or cfg.max_levels < 1 or cfg.ref_order < 2:
        ap.error("--order, --ref-order must be at least 2 and --max-levels 1")
    if not 0.0 < cfg.zmin <= cfg.zmax < 1.0:
        ap.error("need 0 < --zmin <= --zmax < 1 for z = cos(theta)")
    for meth in cfg.methods:
        if meth not in METHODS and parse_grid_method(meth) is None:
            ap.error(f"unknown method {meth}; choose from {METHODS} "
                     f"or kind@Nt with kind in {GRID_KINDS}")
    for name in cfg.trend_outputs:
        if name not in OUTPUT_NAMES:
            ap.error(f"unknown output {name}; choose from {OUTPUT_NAMES}")
    if cfg.a == 0.0 and cfg.e > 0.0:
        ap.error("korb returns NaN on the eccentric branch at a == 0 exactly; "
                 "use a small nonzero spin")
    if cfg.e >= 1.0:
        ap.error("bound orbits only: e < 1")
    return cfg


def main(argv=None):
    cfg = parse_args(sys.argv[1:] if argv is None else argv)
    if cfg.aggregate:
        return aggregate(cfg.aggregate, cfg.nt_target, cfg.agg_kind,
                         cfg.agg_output, cfg.agg_arm)
    here = os.path.dirname(os.path.abspath(__file__))
    path = cfg.out or os.path.join(
        here, "..", "data", "precision",
        f"nmode_scan_m{cfg.m}_n{cfg.n}_a{cfg.a}_p{cfg.p}_e{cfg.e}"
        f"_N{cfg.ntimes}.csv")
    t0 = time.perf_counter()
    rows, rs, zs, _dths, cls = scan(cfg, path)
    elapsed = time.perf_counter() - t0
    rc = report(rows, cfg, rs, zs, cls)
    print()
    print(f"wall time {elapsed:.1f}s   cache: {os.path.normpath(path)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
