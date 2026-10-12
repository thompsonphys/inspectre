import argparse
import csv
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import h5py
import numpy as np

import nphi_nt_map as nm
from mn_fft_sweep import parse_expr

TARGET = 1e-8
REF_TOL = 1e-9
FLOOR = 64
CAP = 2 ** 22
M_MAX = 20
FORMS = ("A", "B")
CS = (8, 10, 12, 14, 16, 20, 24, 28, 37)
NTS = (1024, 2048, 4096, 8192)
N_S, N_S_MAX = 2 ** 15, 2 ** 17
REF = ("A", 37)
C_RULE = ("B", 16)
SETS = ("map", "jittered", "innermost", "spectre")
VERDICTS = {"all": SETS, "to7": ("map", "jittered", "spectre")}
JITTER_SEED, SPECTRE_SEED = 20261012, 20261013
JITTER_DR, JITTER_U = 0.04, 0.15
SPECTRE_DRAW = {True: 60, False: 15}
OFFSET_BINS = (1e-3, 1e-2, 0.1)
NOISE_NPHI = 2 ** 14
INNER_X1 = (1.0 + np.polynomial.legendre.leggauss(4)[0][0]) / 2.0
INNER_DTH = math.asin(INNER_X1 * 0.5 / 2 ** 10)
R_INSIDE = tuple(rs for rs in nm.R_INTERIOR if 100 / 13 < parse_expr(rs) < 100 / 7)
BENCH_POINT = (11.0, math.pi / 2 + 0.5, 0.4)
VERSION = 2
CKPT_DIR = "nphi_rule_ckpt"
OUTPUTS = ("nphi_rule_points.csv", "nphi_rule.csv.gz", "nphi_rule_summary.csv", "nphi_rule_selection.csv",
           "nphi_rule_cost.csv", "nphi_rule.meta.json")
POINT_COLS = ("label", "set", "r", "dth", "kind")
ROW_COLS = ("label", "set", "kind", "N_ref", "form", "N_t", "c", "shift", "E", "E_t", "E_phi", "worst_m",
            "worst_n", "worst_q", "evals", "capped")
SUMMARY_COLS = ("label", "set", "kind", "r", "dth", "verified", "N_ref", "e_half", "capped_any", "bad_any", "noise",
                "noise_t", "form", "N_t", "c_min",
                "E_at_c_min", "E_t_max", "evals_at_c_min", "frac_floor", "seconds")
SELECTION_COLS = ("verdict", "form", "N_t", "c", "fail_map", "fail_jittered", "fail_innermost", "fail_spectre",
                  "fail_total", "E_max", "E_max_label", "selected_c", "total_evals", "mean_evals", "max_evals",
                  "selected_pair")
COST_COLS = ("form", "N_t", "c", "total_evals", "mean_evals", "max_evals", "fixed_65536_total", "map_rule_total",
             "map_fixed_total")


def _pow2_at_least(want):
    mant, ex = np.frexp(want)
    return np.where(mant == 0.5, np.ldexp(1.0, ex - 1), np.ldexp(1.0, ex))


def rule_arrays(form, c, c1, cap=CAP):
    c1 = np.asarray(c1, dtype=float)
    bad = ~(c1 > 0)
    want = np.maximum(FLOOR, c / np.sqrt(np.where(bad, 1.0, c1)) + (M_MAX if form == "B" else 0))
    capped = (want > cap) & ~bad
    n = np.where(bad, 0, np.where(capped, cap, _pow2_at_least(want))).astype(np.int64)
    return n, capped, bad


def rule_n(form, c, c1, cap=CAP):
    n, capped, bad = rule_arrays(form, c, np.array([c1], dtype=float), cap)
    return int(n[0]), bool(capped[0]), bool(bad[0])


def c1_now(insp, r, theta):
    ctx = insp.es._ef._ctx
    dr = r - ctx.xp.r
    dth = theta - ctx.xp.theta
    return (ctx.alpha20 * dr * dr + ctx.alpha02 * dth * dth) / ctx.beta


def projected(insp, r, theta, lam, n):
    A, nphi, _ = insp.mmode_projected_fast(list(nm.MS) + [n // 2], r, theta, lam, n)
    if nphi != n:
        raise RuntimeError(f"requested N_phi = {n}, the C route used {nphi}")
    return A[:len(nm.MS)]


def place(insp, r, theta, lam):
    A, nphi, _ = insp.mmode_projected_fast(list(nm.MS) + [FLOOR // 2], r, theta, lam, FLOOR)
    c1 = c1_now(insp, r, theta)
    if nphi == 0:
        return c1, None
    if nphi != FLOOR:
        raise RuntimeError(f"requested N_phi = {FLOOR}, the C route used {nphi}")
    return c1, A[:len(nm.MS)]


def radial_period(insp):
    return insp.t_from_lambda(insp.mino_period_r)


class Pass:
    def __init__(self, n_s, combos):
        self.n_s = n_s
        self.combos = list(combos)
        self.lam = np.full(n_s, np.nan)
        self.c1 = np.full(n_s, np.nan)
        self.bad = np.zeros(n_s, bool)
        self.stack = []
        self.index = {cb: np.full(n_s, -1, np.int64) for cb in self.combos}
        self.nphi = {cb: np.zeros(n_s, np.int64) for cb in self.combos}
        self.capped = {cb: np.zeros(n_s, bool) for cb in self.combos}
        self._array = None

    def array(self):
        if self._array is None or len(self._array) != len(self.stack) + 1:
            self._array = np.concatenate([np.full((1, len(nm.MS), 4), np.nan + 0j)] +
                                         [s[None] for s in self.stack])
        return self._array

    def values(self, cb):
        return np.moveaxis(self.array()[self.index[cb] + 1], 0, 2)


def _fill(p, insp, r, theta, js, cap, combos=None):
    m = np.array(nm.MS, dtype=float)
    T_r, omega = radial_period(insp), insp.omega_phi
    for j in js:
        t = j / p.n_s * T_r
        lam = insp.lambda_from_t(t)
        c1, a64 = place(insp, r, theta, lam)
        p.lam[j], p.c1[j] = lam, c1
        phase = np.exp(1j * m * omega * t)[:, None]
        got = {}
        for cb in combos or p.combos:
            n, capped, bad = rule_n(cb[0], cb[1], c1, cap)
            p.nphi[cb][j], p.capped[cb][j] = n, capped
            if bad or a64 is None:
                p.bad[j] = True
                continue
            if n not in got:
                got[n] = len(p.stack)
                p.stack.append((a64 if n == FLOOR else projected(insp, r, theta, lam, n)) * phase)
            p.index[cb][j] = got[n]


def run_pass(insp, r, theta, n_s, combos, cap=CAP, indices=None):
    p = Pass(n_s, combos)
    _fill(p, insp, r, theta, range(n_s) if indices is None else indices, cap)
    return p


def refine(p, insp, r, theta, cap=CAP, combos=None):
    q = Pass(2 * p.n_s, p.combos)
    q.stack = p.stack
    q.lam[0::2], q.c1[0::2], q.bad[0::2] = p.lam, p.c1, p.bad
    for cb in p.combos:
        q.index[cb][0::2], q.nphi[cb][0::2], q.capped[cb][0::2] = p.index[cb], p.nphi[cb], p.capped[cb]
    _fill(q, insp, r, theta, range(1, q.n_s, 2), cap, combos)
    return q


def grid_modes(v, n_s, n_t, s):
    d = n_s // n_t
    if n_t < 1 or d * n_t != n_s or not 0 <= s < d:
        raise ValueError(f"no grid of {n_t} samples with shift {s} in {n_s}")
    w = np.fft.ifft(v[:, :, s::d], axis=2)
    phase = np.exp(2j * np.pi * np.array(nm.NS) * s / n_s)
    return w[:, :, [n % n_t for n in nm.NS]] * phase


class Ref:
    def __init__(self, p, R, verified, e_half, capped_any, bad_any):
        self.p, self.R, self.verified, self.e_half = p, R, verified, e_half
        self.capped_any, self.bad_any = capped_any, bad_any


def reference(insp, r, theta, qs, n_s, n_s_max, combos, cap=CAP):
    p = run_pass(insp, r, theta, n_s, combos, cap)
    while True:
        v = p.values(REF)
        R = grid_modes(v, p.n_s, p.n_s, 0)
        e_half = nm.point_error(grid_modes(v, p.n_s, p.n_s // 2, 0), R, qs)[0]
        capped_any, bad_any = bool(p.capped[REF].any()), bool(p.bad.any())
        ok = not capped_any and not bad_any and e_half <= REF_TOL
        if ok or p.n_s >= n_s_max:
            return Ref(p, R, ok, e_half, capped_any, bad_any)
        p = refine(p, insp, r, theta, cap, [REF])


def noise_level(insp, r, theta, ref, qs, n_ts, n_s_base, threshold=NOISE_NPHI):
    p = ref.p
    js = np.nonzero(p.nphi[REF] >= threshold)[0]
    if len(js) == 0:
        return 0.0, {n_t: 0.0 for n_t in n_ts}
    v0 = p.values(REF)
    v = v0.copy()
    m = np.array(nm.MS, dtype=float)
    T_r, omega = radial_period(insp), insp.omega_phi
    for j in js:
        t = j / p.n_s * T_r
        lam = np.nextafter(p.lam[j], np.inf)
        v[:, :, j] = projected(insp, r, theta, lam, int(p.nphi[REF][j])) * np.exp(1j * m * omega * t)[:, None]
    stride = p.n_s // n_s_base
    per = {}
    for n_t in n_ts:
        per[n_t] = float(max(nm.point_error(grid_modes(v, p.n_s, n_t, sb * stride),
                                            grid_modes(v0, p.n_s, n_t, sb * stride), qs)[0]
                             for sb in range(n_s_base // n_t)))
    return nm.point_error(grid_modes(v, p.n_s, p.n_s, 0), ref.R, qs)[0], per


def c_min(cs, emax):
    return nm.required_level(list(cs), list(emax), TARGET)


def selected_c(points, sets):
    vals = [p["c_min"] for p in points if p["set"] in sets and p["verified"]]
    if not vals or any(v is None for v in vals):
        return None
    return max(vals)


def selected_pair(sel, cost):
    cands = [(cost[(n_t, c)], n_t, c) for n_t, c in sel.items() if c is not None]
    if not cands:
        return None
    return min(cands)[1:]


def analyse(pt, ref, qs, cfg):
    p, R = ref.p, ref.R
    stride = p.n_s // cfg["n_s"]
    base = (pt["label"], pt["set"], pt["kind"], p.n_s)
    grids = [(n_t, sb) for n_t in cfg["n_ts"] for sb in range(cfg["n_s"] // n_t)]
    vref = p.values(REF)
    bref, e_t = {}, {}
    for n_t, sb in grids:
        bref[(n_t, sb)] = grid_modes(vref, p.n_s, n_t, sb * stride)
        e_t[(n_t, sb)] = nm.point_error(bref[(n_t, sb)], R, qs)[0]
    rows, emax, evals0 = [], {}, {}
    for form in FORMS:
        for c in cfg["cs"]:
            cb = (form, c)
            v = p.values(cb)
            for n_t, sb in grids:
                s, d = sb * stride, p.n_s // n_t
                B = grid_modes(v, p.n_s, n_t, s)
                e, wm, wn, wq = nm.point_error(B, R, qs)
                ep = nm.point_error(B, bref[(n_t, sb)], qs)[0]
                ev = int(p.nphi[cb][s::d].sum())
                rows.append(base + (form, n_t, c, sb, e, e_t[(n_t, sb)], ep, wm, wn, nm.QNAMES[wq], ev,
                                    int(p.capped[cb][s::d].sum())))
                key = (form, n_t, c)
                emax.setdefault(key, []).append(e)
                if sb == 0:
                    evals0[key] = ev
    emax = {k: float(np.max(v)) for k, v in emax.items()}
    summary = []
    for form in FORMS:
        for n_t in cfg["n_ts"]:
            cm = c_min(cfg["cs"], [emax[(form, n_t, c)] for c in cfg["cs"]])
            frac = float(np.mean(p.nphi[(form, cm)][0::p.n_s // n_t] == FLOOR)) if cm is not None else None
            summary.append(dict(label=pt["label"], set=pt["set"], kind=pt["kind"], r=pt["r"], dth=pt["dth"],
                                verified=ref.verified, N_ref=p.n_s, form=form, N_t=n_t, c_min=cm,
                                E_at_c_min=emax[(form, n_t, cm)] if cm is not None else None,
                                E_t_max=max(e_t[(n_t, sb)] for sb in range(cfg["n_s"] // n_t)),
                                evals_at_c_min=evals0[(form, n_t, cm)] if cm is not None else None,
                                frac_floor=frac))
    return rows, summary


_insp = None


def _init():
    global _insp
    from inspectre import Inspectre
    _insp = Inspectre(spin=nm.SPIN, semilatus_rectum=nm.SEMILATUS, eccentricity=nm.ECC)


def ckpt_path(out_dir, label):
    return os.path.join(out_dir, CKPT_DIR, hashlib.md5(label.encode()).hexdigest() + ".pkl")


def settings_key(cfg):
    return [VERSION, cfg["n_s"], cfg["n_s_max"], cfg["cap"], list(cfg["cs"]), list(cfg["n_ts"]), TARGET, REF_TOL,
            NOISE_NPHI, M_MAX, FLOOR]


def load_ckpt(path, cfg, pt):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError):
        return None
    if data.get("settings") != settings_key(cfg) or data.get("point") != pt:
        return None
    return data


def run_point(job):
    pt, cfg = job
    t0 = time.time()
    qs = nm.GROUP[pt["kind"]]
    theta = math.pi / 2 + pt["dth"]
    combos = [(f, c) for f in FORMS for c in cfg["cs"]]
    ref = reference(_insp, pt["r"], theta, qs, cfg["n_s"], cfg["n_s_max"], combos, cfg["cap"])
    noise, noise_t = noise_level(_insp, pt["r"], theta, ref, qs, cfg["n_ts"], cfg["n_s"])
    rows, summary = analyse(pt, ref, qs, cfg)
    seconds = time.time() - t0
    for s in summary:
        s.update(e_half=ref.e_half, capped_any=ref.capped_any, bad_any=ref.bad_any, noise=noise,
                 noise_t=noise_t[s["N_t"]], seconds=seconds)
    path = ckpt_path(cfg["out_dir"], pt["label"])
    with open(path + ".part", "wb") as f:
        pickle.dump(dict(settings=settings_key(cfg), point=pt, rows=rows, summary=summary), f)
    os.replace(path + ".part", path)
    return pt["label"], seconds


def map_points():
    return [dict(p, set="map") for p in nm.points()]


def jittered_points():
    rng = np.random.default_rng(JITTER_SEED)
    pts = []
    for rs in R_INSIDE:
        for ds in nm.DTH:
            if parse_expr(ds) > 1e-2:
                continue
            for _ in range(2):
                r = parse_expr(rs) + rng.uniform(-JITTER_DR, JITTER_DR)
                dth = parse_expr(ds) * 10 ** rng.uniform(-JITTER_U, JITTER_U)
                pts.append(dict(label=f"jit:r={r:.17g},dth={dth:.17g}", set="jittered", r=r, dth=dth,
                                kind="interior"))
    return pts


def innermost_points():
    return [dict(label=f"inner:r={rs},dth={INNER_DTH:.17g}", set="innermost", r=parse_expr(rs), dth=INNER_DTH,
                 kind="interior") for rs in R_INSIDE]


def worldtube_points(volume):
    with h5py.File(volume, "r") as f:
        g = f["VolumeData.vol"]
        key = max((g[k].attrs["observation_value"], k) for k in g if isinstance(g[k], h5py.Group))[1]
        x = g[key]["InertialCoordinates_x"][()]
        y = g[key]["InertialCoordinates_y"][()]
        r = g[key]["BoyerLindquistRadius"][()]
    wt = (x > nm.RSTAR_P - 10 - 1e-9) & (x < nm.RSTAR_P + 10 + 1e-9) & (y < 0) & (y >= -0.5 - 1e-12)
    return np.unique(np.c_[r[wt], np.arcsin(-y[wt])], axis=0)


def inside(r):
    return bool(100 / 13 < r < 100 / 7)


def offset_bin(dth):
    return int(np.searchsorted(OFFSET_BINS, dth, side="right"))


def spectre_points(pairs):
    rng = np.random.default_rng(SPECTRE_SEED)
    flags = np.array([inside(r) for r in pairs[:, 0]])
    bins = np.array([offset_bin(d) for d in pairs[:, 1]])
    pts = []
    for flag in (True, False):
        for b in range(len(OFFSET_BINS) + 1):
            idx = np.nonzero((flags == flag) & (bins == b))[0]
            for i in sorted(rng.choice(idx, SPECTRE_DRAW[flag], replace=False)):
                r, dth = float(pairs[i, 0]), float(pairs[i, 1])
                pts.append(dict(label=f"spec:r={r:.17g},dth={dth:.17g}", set="spectre", r=r, dth=dth,
                                kind="interior"))
    return pts


def build_points(pairs):
    return map_points() + jittered_points() + innermost_points() + spectre_points(pairs)


def read_points(path):
    with open(path) as f:
        return [dict(label=row["label"], set=row["set"], r=float(row["r"]), dth=float(row["dth"]),
                     kind=row["kind"]) for row in csv.DictReader(f)]


def benchmark(insp):
    r, theta, lam = BENCH_POINT
    for _ in range(200):
        projected(insp, r, theta, lam, FLOOR)
    times = {}
    for k in range(6, 23):
        n = 2 ** k
        best = math.inf
        projected(insp, r, theta, lam, n)
        for _ in range(3):
            t0 = time.perf_counter()
            projected(insp, r, theta, lam, n)
            best = min(best, time.perf_counter() - t0)
        times[n] = best
    top = max(times)
    return dict(t_eval=(times[top] - times[FLOOR]) / (top - FLOOR), t_call=times[FLOOR], t_max=times[top])


def worldtube_cost(insp, pairs, n_ts, cs, cap):
    n_max = max(n_ts)
    T_r = radial_period(insp)
    r, dth = pairs[:, 0], pairs[:, 1]
    evals = {(f, n_t, c): np.zeros(len(pairs)) for f in FORMS for n_t in n_ts for c in cs}
    ctx = insp.es._ef._ctx
    for j in range(n_max):
        lam = insp.lambda_from_t(j / n_max * T_r)
        place(insp, BENCH_POINT[0], BENCH_POINT[1], lam)
        dr = r - ctx.xp.r
        c1 = (ctx.alpha20 * dr * dr + ctx.alpha02 * dth * dth) / ctx.beta
        users = [n_t for n_t in n_ts if j % (n_max // n_t) == 0]
        for f in FORMS:
            for c in cs:
                n = rule_arrays(f, c, c1, cap)[0]
                for n_t in users:
                    evals[(f, n_t, c)] += n
    return {k: (float(v.sum()), float(v.mean()), float(v.max())) for k, v in evals.items()}


def selection_tables(points, rows, summary, cfg, cost, nphi_all, n_pairs):
    emax = {}
    for row in rows:
        emax.setdefault((row[0], row[4], row[5], row[6]), []).append(row[8])
    emax = {k: float(np.max(v)) for k, v in emax.items()}
    sets_of = {p["label"]: p["set"] for p in points}
    verified = {s["label"]: s["verified"] for s in summary}
    sel_rows, pairs_out = [], {}
    for verdict, sets in VERDICTS.items():
        pairs_out[verdict] = {}
        for form in FORMS:
            sel = {}
            for n_t in cfg["n_ts"]:
                pts = [dict(set=s["set"], verified=s["verified"], c_min=s["c_min"]) for s in summary
                       if s["form"] == form and s["N_t"] == n_t]
                sel[n_t] = selected_c(pts, sets)
            pair = selected_pair(sel, {(n_t, c): cost[(form, n_t, c)][0] for n_t in cfg["n_ts"] for c in cfg["cs"]})
            pairs_out[verdict][form] = list(pair) if pair else None
            for n_t in cfg["n_ts"]:
                for c in cfg["cs"]:
                    fails = {s: 0 for s in SETS}
                    worst, worst_label = -1.0, ""
                    for label, st in sets_of.items():
                        if st not in sets or not verified.get(label, False):
                            continue
                        e = emax.get((label, form, n_t, c), math.nan)
                        if not e <= TARGET:
                            fails[st] += 1
                        if not math.isnan(worst) and not e <= worst:
                            worst, worst_label = e, label
                    tot, mean, mx = cost[(form, n_t, c)]
                    sel_rows.append(dict(verdict=verdict, form=form, N_t=n_t, c=c, fail_map=fails["map"],
                                         fail_jittered=fails["jittered"], fail_innermost=fails["innermost"],
                                         fail_spectre=fails["spectre"], fail_total=sum(fails.values()),
                                         E_max=worst, E_max_label=worst_label, selected_c=sel[n_t],
                                         total_evals=tot, mean_evals=mean, max_evals=mx,
                                         selected_pair=pair == (n_t, c)))
    map_interior = {p["label"] for p in points if p["set"] == "map" and p["kind"] == "interior" and p["label"] in nphi_all}
    rule_map = {}
    for row in rows:
        if row[0] in map_interior and row[7] == 0:
            key = (row[4], row[5], row[6])
            rule_map[key] = rule_map.get(key, 0) + row[14]
    cost_rows = []
    for form in FORMS:
        for n_t in cfg["n_ts"]:
            fixed_map = sum(n_t * nphi_all[label] for label in map_interior)
            for c in cfg["cs"]:
                tot, mean, mx = cost[(form, n_t, c)]
                cost_rows.append(dict(form=form, N_t=n_t, c=c, total_evals=tot, mean_evals=mean, max_evals=mx,
                                      fixed_65536_total=float(n_t) * 65536 * n_pairs,
                                      map_rule_total=rule_map.get((form, n_t, c), 0),
                                      map_fixed_total=fixed_map))
    return sel_rows, cost_rows, pairs_out


def _is_pow2(n, low):
    return n >= low and n & (n - 1) == 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectre-volume", required=True)
    ap.add_argument("--map-summary", required=True)
    ap.add_argument("--points-file")
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--points", nargs="+")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-samples", type=int, default=N_S)
    ap.add_argument("--n-samples-max", type=int, default=N_S_MAX)
    ap.add_argument("--n-t", type=int, nargs="+", default=list(NTS))
    ap.add_argument("--cs", type=int, nargs="+", default=list(CS))
    ap.add_argument("--cap", type=int, default=CAP)
    args = ap.parse_args(argv)

    n_ts, cs = sorted(set(args.n_t)), sorted(set(args.cs))
    if not (_is_pow2(args.n_samples, 64) and _is_pow2(args.n_samples_max, 64)
            and args.n_samples_max >= args.n_samples):
        raise SystemExit("--n-samples and --n-samples-max must be powers of two >= 64, max >= samples")
    if not all(_is_pow2(n, 64) and n <= args.n_samples for n in n_ts):
        raise SystemExit("--n-t values must be powers of two in [64, --n-samples]")
    if REF[1] not in cs or min(cs) <= 0:
        raise SystemExit(f"--cs must be positive and contain {REF[1]}")
    if not _is_pow2(args.cap, FLOOR):
        raise SystemExit("--cap must be a power of two >= 64")

    t_start = time.time()
    pairs = worldtube_points(args.spectre_volume)
    pts = read_points(args.points_file) if args.points_file else build_points(pairs)
    pts = [p for p in pts if p["set"] in args.sets]
    if args.points:
        pts = [p for p in pts if p["label"] in args.points]
        missing = set(args.points) - {p["label"] for p in pts}
        if missing:
            raise SystemExit(f"unknown points: {sorted(missing)}")
    with open(args.map_summary) as f:
        nphi_all = {row["label"]: float(row["Nphi_all"]) for row in csv.DictReader(f) if row["Nphi_all"]}

    cfg = dict(n_s=args.n_samples, n_s_max=args.n_samples_max, cap=args.cap, cs=tuple(cs), n_ts=tuple(n_ts),
               out_dir=args.out_dir)
    os.makedirs(os.path.join(args.out_dir, CKPT_DIR), exist_ok=True)
    _init()
    bench = benchmark(_insp)
    cost = worldtube_cost(_insp, pairs, n_ts, cs, args.cap)

    todo = []
    for p in pts:
        if load_ckpt(ckpt_path(args.out_dir, p["label"]), cfg, p) is None:
            todo.append(p)
        else:
            print(f"cached {p['label']}", flush=True)
    t_pool = time.time()
    if todo:
        with multiprocessing.Pool(args.workers, initializer=_init) as pool:
            jobs = [(p, cfg) for p in todo]
            for k, (label, sec) in enumerate(pool.imap_unordered(run_point, jobs), 1):
                print(f"computed {label} {sec:.0f} s ({k}/{len(jobs)})", flush=True)
    t_pool = time.time() - t_pool

    rows, summary = [], []
    for p in pts:
        data = load_ckpt(ckpt_path(args.out_dir, p["label"]), cfg, p)
        if data is None:
            raise SystemExit(f"missing checkpoint for {p['label']}")
        rows += data["rows"]
        summary += data["summary"]
    sel_rows, cost_rows, selected = selection_tables(pts, rows, summary, cfg, cost, nphi_all, len(pairs))

    finals = [os.path.join(args.out_dir, n) for n in OUTPUTS]
    parts = [f + ".part" for f in finals]
    try:
        with open(parts[0], "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=POINT_COLS)
            w.writeheader()
            w.writerows({k: p[k] for k in POINT_COLS} for p in pts)
        with gzip.open(parts[1], "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(ROW_COLS)
            w.writerows(rows)
        for path, cols, data in ((parts[2], SUMMARY_COLS, summary), (parts[3], SELECTION_COLS, sel_rows),
                                 (parts[4], COST_COLS, cost_rows)):
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows({k: ("" if d[k] is None else d[k]) for k in cols} for d in data)
        meta = dict(orbit=[nm.SPIN, nm.SEMILATUS, nm.ECC], target=TARGET, ref_tol=REF_TOL, floor=FLOOR, cap=args.cap,
                    m_max=M_MAX, forms=list(FORMS), cs=cs, n_ts=n_ts, n_samples=args.n_samples,
                    n_samples_max=args.n_samples_max, noise_nphi=NOISE_NPHI, inner_dth=INNER_DTH,
                    seeds=dict(jitter=JITTER_SEED, spectre=SPECTRE_SEED),
                    inputs=dict(spectre_volume=os.path.abspath(args.spectre_volume),
                                map_summary=os.path.abspath(args.map_summary),
                                points_file=os.path.abspath(args.points_file) if args.points_file else None),
                    worldtube_points=int(len(pairs)), points=len(pts), unit_costs=bench, selected=selected,
                    wall_seconds=time.time() - t_start, pool_seconds=t_pool)
        with open(parts[5], "w") as f:
            json.dump(meta, f, indent=1)
        for part, final in zip(parts, finals):
            os.replace(part, final)
    finally:
        for part in parts:
            if os.path.exists(part):
                os.remove(part)
    print("wrote " + ", ".join(finals))


if __name__ == "__main__":
    main()
