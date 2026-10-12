import argparse
import csv
import gzip
import hashlib
import math
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from inspectre import Inspectre
from mn_fft_sweep import parse_expr, read_reference
import nphi_nt_map as nm
import nphi_rule as nr

ap = argparse.ArgumentParser()
ap.add_argument("--crosscheck", required=True)
ap.add_argument("--spectre-volume", required=True)
ap.add_argument("--map-summary", required=True)
ap.add_argument("names", nargs="*")
ARGS = ap.parse_args()
SELECTED = set(ARGS.names)
I = Inspectre(spin=nm.SPIN, semilatus_rectum=nm.SEMILATUS, eccentricity=nm.ECC)
results = []
_cache = {}


def check(name, fn):
    if SELECTED and name not in SELECTED:
        return
    try:
        ok, val = fn()
    except Exception as exc:
        ok, val = False, f"{type(exc).__name__}: {exc}"
    results.append(bool(ok))
    print("check", name, "PASS" if ok else "FAIL", val, flush=True)


def ref_pass(r, dth, n_s):
    key = (r, dth, n_s)
    if key not in _cache:
        _cache[key] = nr.run_pass(I, r, math.pi / 2 + dth, n_s, [nr.REF])
    return _cache[key]


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def c_emulation():
    th = math.pi / 2 + 1e-2
    p = nr.run_pass(I, 11.0, th, 1024, [nr.C_RULE], cap=2 ** 20)
    B = nr.grid_modes(p.values(nr.C_RULE), 1024, 1024, 0)
    A, _ = I.fft_nmodes_fast(nm.MS, 11.0, th, 1024, "numerical", 2 ** 20)
    e = nm.point_error(B, nm.bins(A, 1024, 1024), (0, 1, 2, 3))[0]
    return e <= 1e-12, f"{e:.1e}"


def c_c1():
    th = math.pi / 2 + 1e-4
    bad = []
    for lam in 0.7176224454015108 + np.linspace(-1e-3, 1e-3, 20):
        _, nphi, _ = I.mmode_projected_fast(list(nm.MS), 11.0, th, lam, 2 ** 22)
        c1 = nr.place(I, 11.0, th, lam)[0]
        n = nr.rule_n(*nr.C_RULE, c1)[0]
        if n != nphi:
            bad.append((float(lam), n, nphi))
    return not bad, bad[:3] or "20/20"


def c_rule():
    cases = [(("A", 16, 1.0), (64, False, False)), (("A", 16, 0.01), (256, False, False)),
             (("A", 12, 0.04), (64, False, False)), (("B", 12, 0.04), (128, False, False)),
             (("B", 37, 1.0), (64, False, False)), (("A", 37, 1e-14), (nr.CAP, True, False)),
             (("A", 16, 0.0), (0, False, True)), (("B", 16, float("nan")), (0, False, True)),
             (("A", 16, 1 / 16), (64, False, False)), (("A", 32, 1 / 16), (128, False, False))]
    got = [nr.rule_n(*a) for a, _ in cases]
    ok = all(g == w for g, (_, w) in zip(got, cases))
    th = math.pi / 2 + 0.5
    for n in (64, 1024, 2 ** 16):
        nr.projected(I, 11.0, th, 0.4, n)
    return ok, got


def c_shifts():
    th = math.pi / 2 + 1e-2
    p = nr.run_pass(I, 11.0, th, 1024, [nr.REF])
    q = nr.run_pass(I, 11.0, th, 1024, [nr.REF], indices=range(3, 1024, 4))
    v, w = p.values(nr.REF), q.values(nr.REF)
    same = bool(np.array_equal(v[:, :, 3::4], w[:, :, 3::4]))
    k = np.arange(256)
    kernel = np.exp(2j * np.pi * np.outer(nm.NS, 3 + 4 * k) / 1024)
    direct = np.einsum("mqk,nk->mqn", v[:, :, 3::4], kernel) / 256
    e = nm.point_error(nr.grid_modes(v, 1024, 256, 3), direct, (0, 1, 2, 3))[0]
    u = nr.run_pass(I, 11.0, th, 256, [nr.REF])
    zero = bool(np.array_equal(nr.grid_modes(v, 1024, 256, 0), nr.grid_modes(u.values(nr.REF), 256, 256, 0)))
    return same and e <= 1e-13 and zero, (same, f"{e:.1e}", zero)


def c_selection():
    cs = (8, 16, 24, 37)
    a = nr.c_min(cs, [2e-8, 5e-9, 2e-8, 1e-9])
    b = nr.c_min(cs, [1e-9, 2e-9, 3e-9, 1e-9])
    c = nr.c_min(cs, [1e-9, 2e-9, 3e-9, 2e-8])
    d = nr.c_min(cs, [2e-8, 5e-9, 4e-9, float("nan")])
    e = nr.c_min(cs, [2e-8, 5e-9, 4e-9, 1e-9])
    pts = [dict(set="map", verified=True, c_min=16), dict(set="spectre", verified=True, c_min=24),
           dict(set="innermost", verified=True, c_min=None), dict(set="map", verified=False, c_min=None)]
    s_all = nr.selected_c(pts, nr.VERDICTS["all"])
    s_to7 = nr.selected_c(pts, nr.VERDICTS["to7"])
    cost = {(1024, 24): 5.0, (2048, 24): 3.0, (2048, 16): 1.0, (4096, 37): 0.5}
    pair = nr.selected_pair({1024: 24, 2048: 24, 4096: None}, cost)
    none = nr.selected_pair({1024: None, 2048: None}, cost)
    got = (a, b, c, d, e, s_all, s_to7, pair, none)
    return got == (37, 8, None, None, 16, None, 24, (2048, 24), None), got


def c_escalated():
    th = math.pi / 2 + 1e-4
    cfg = dict(n_s=256, n_s_max=512, cap=nr.CAP, cs=(16, 37), n_ts=(64,))
    combos = [(f, c) for f in nr.FORMS for c in cfg["cs"]]
    ref = nr.reference(I, 11.0, th, (3,), 256, 512, combos)
    base = nr.run_pass(I, 11.0, th, 256, combos)
    pt = dict(label="x", set="map", kind="interior", r=11.0, dth=1e-4)
    rows, _ = nr.analyse(pt, ref, (3,), cfg)
    worst = 0.0
    for row in rows:
        form, n_t, c, sb = row[4], row[5], row[6], row[7]
        B = nr.grid_modes(base.values((form, c)), 256, n_t, sb)
        Bref = nr.grid_modes(base.values(nr.REF), 256, n_t, sb)
        worst = max(worst, abs(row[10] - nm.point_error(B, Bref, (3,))[0]), abs(row[9] - nm.point_error(Bref, ref.R, (3,))[0]))
    return ref.p.n_s == 512 and len(rows) == 4 * 4 and worst == 0.0, (ref.p.n_s, len(rows), worst)


def c_selection_tables():
    pts = [dict(label=l, set=s, kind="interior") for l, s in (("a", "map"), ("b", "innermost"), ("c", "spectre"), ("d", "map"))]
    cfg = dict(cs=(16, 37), n_ts=(64,))
    rows = []
    for label, e16, e37 in (("a", 2e-8, 1e-9), ("b", 1e-9, float("nan")), ("c", 1e-9, 1e-9), ("d", 5e-8, 5e-8)):
        for form in nr.FORMS:
            for c, e in ((16, e16), (37, e37)):
                for sb in (0, 1):
                    rows.append((label, None, "interior", 256, form, 64, c, sb, e * (1 + sb), 0.0, 0.0, 0, 0, "src", 100 * c, 0))
    summary = [dict(label=l, set=s, form=f, N_t=64, verified=v, c_min=cm) for l, s, v, cm in
               (("a", "map", True, 37), ("b", "innermost", True, None), ("c", "spectre", True, 16), ("d", "map", False, None))
               for f in nr.FORMS]
    cost = {(f, 64, c): (float(c), 1.0, 2.0) for f in nr.FORMS for c in (16, 37)}
    sel, cost_rows, chosen = nr.selection_tables(pts, rows, summary, cfg, cost, {"a": 128.0}, 10)
    get = {(r["verdict"], r["form"], r["c"]): r for r in sel}
    all16, all37, to7_37 = get[("all", "A", 16)], get[("all", "A", 37)], get[("to7", "A", 37)]
    got = (all16["fail_map"], all16["fail_innermost"], all16["fail_total"], all37["fail_innermost"], all37["E_max_label"],
           to7_37["fail_total"], chosen["all"]["A"], chosen["to7"]["A"], cost_rows[0]["map_rule_total"], cost_rows[0]["map_fixed_total"])
    want = (1, 0, 1, 1, "b", 0, None, [64, 37], 1600, 64 * 128.0)
    return got == want, got


def c_noise():
    th = math.pi / 2 + 1e-2
    ref = nr.reference(I, 11.0, th, (3,), 512, 512, [nr.REF])
    full, per = nr.noise_level(I, 11.0, th, ref, (3,), (128, 512), 512, threshold=64)
    again = nr.noise_level(I, 11.0, th, ref, (3,), (128, 512), 512, threshold=64)
    none = nr.noise_level(I, 11.0, th, ref, (3,), (128, 512), 512)
    ok = (sorted(per) == [128, 512] and per[512] == full and 0 < full < 1e-9 and per[128] > 0
          and (full, per) == again and none == (0.0, {128: 0.0, 512: 0.0}))
    return ok, (f"{full:.1e}", {k: f"{v:.1e}" for k, v in per.items()}, none)


def c_reference():
    xref = read_reference(ARGS.crosscheck)
    p = ref_pass(11.0, 1e-4, 2 ** 14)
    R = nr.grid_modes(p.values(nr.REF), 2 ** 14, 2 ** 14, 0)
    e = nm.crosscheck_error(R, xref, "r=11,dth=1/10000", (3,))[0]
    ref = nr.reference(I, 11.0, math.pi / 2 + nr.INNER_DTH, (3,), 2 ** 15, 2 ** 16, [nr.REF])
    ok = e <= 1e-9 and ref.verified and not ref.capped_any
    return ok, (f"{e:.2e}", ref.verified, f"{ref.e_half:.2e}", ref.p.n_s, ref.capped_any)


def c_map():
    p = ref_pass(11.0, 1e-4, 2 ** 14)
    v = p.values(nr.REF)
    R = nr.grid_modes(v, 2 ** 14, 2 ** 14, 0)
    rows_path = os.path.join(os.path.dirname(os.path.abspath(ARGS.map_summary)), "nphi_nt_map.csv.gz")
    stored = {}
    with gzip.open(rows_path, "rt") as f:
        for row in csv.DictReader(f):
            if row["label"] == "r=11,dth=1/10000" and row["nphi_max"] == "0" and row["N_t"] in ("1024", "2048"):
                stored[int(row["N_t"])] = float(row["E"])
    mine = {nt: nm.point_error(nr.grid_modes(v, 2 ** 14, nt, 0), R, (3,))[0] for nt in (1024, 2048)}
    rel = max(abs(mine[nt] / stored[nt] - 1) for nt in (1024, 2048))
    return rel <= 0.05, ({k: f"{x:.3e}" for k, x in mine.items()}, {k: f"{x:.3e}" for k, x in stored.items()}, f"{rel:.3f}")


def c_spectre_points():
    pairs = nr.worldtube_points(ARGS.spectre_volume)
    r_in, r_out = parse_expr(nm.R_IN_S), parse_expr(nm.R_OUT_S)
    inb = bool(all(r_in <= r <= r_out and 0 < d <= math.pi / 6 for r, d in pairs))
    s1, s2 = nr.spectre_points(pairs), nr.spectre_points(pairs)
    counts = {}
    for p in s1:
        key = (nr.inside(p["r"]), nr.offset_bin(p["dth"]))
        counts[key] = counts.get(key, 0) + 1
    want = {(True, b): 60 for b in range(4)} | {(False, b): 15 for b in range(4)}
    ok = len(pairs) == 9218 and inb and s1 == s2 and counts == want
    return ok, (len(pairs), inb, s1 == s2, sorted(counts.items()))


def c_benchmark():
    b = nr.benchmark(I)
    ok = 0.2e-6 <= b["t_eval"] <= 2e-6 and 0.1e-3 <= b["t_call"] <= 5e-3
    return ok, {k: f"{x:.3g}" for k, x in b.items()}


def c_end_to_end():
    script = [sys.executable, os.path.join(ROOT, "test", "nphi_rule.py"), "--spectre-volume", ARGS.spectre_volume,
              "--map-summary", ARGS.map_summary]
    common = ["--points", "r=11,dth=1/100", "r=11,dth=1/10000", "--n-samples", "1024", "--n-samples-max", "1024",
              "--n-t", "128", "256", "--cs", "16", "37", "--workers", "2"]
    with tempfile.TemporaryDirectory() as tmp:
        good = subprocess.run(script + ["--out-dir", tmp] + common, capture_output=True, text=True)
        names = sorted(n for n in os.listdir(tmp) if n != nr.CKPT_DIR)
        heads, summary = {}, []
        if good.returncode == 0:
            with open(os.path.join(tmp, "nphi_rule_summary.csv")) as f:
                summary = list(csv.DictReader(f))
            with gzip.open(os.path.join(tmp, "nphi_rule.csv.gz"), "rt") as f:
                heads["rows"] = tuple(next(csv.reader(f)))
            for name, key in (("nphi_rule_summary.csv", "summary"), ("nphi_rule_selection.csv", "selection"),
                              ("nphi_rule_cost.csv", "cost"), ("nphi_rule_points.csv", "points")):
                with open(os.path.join(tmp, name)) as f:
                    heads[key] = tuple(next(csv.reader(f)))
        before = {n: md5(os.path.join(tmp, n)) for n in names}
        bad_points = os.path.join(tmp, "bad_points.csv")
        with open(bad_points, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(nr.POINT_COLS)
            w.writerow(["r=11,dth=1/100", "map", 11.0, 0.01, "interior"])
            w.writerow(["bogus", "map", 11.0, 0.02, "bogus"])
        bad = subprocess.run(script + ["--out-dir", tmp, "--points-file", bad_points, "--n-samples", "1024",
                                       "--n-samples-max", "1024", "--n-t", "128", "256", "--cs", "16", "37",
                                       "--workers", "1"], capture_output=True, text=True)
        after = {n: md5(os.path.join(tmp, n)) for n in names}
        ck = os.path.join(tmp, nr.CKPT_DIR)
        files = sorted(os.listdir(ck)) if os.path.isdir(ck) else []
        if files:
            os.remove(os.path.join(ck, files[0]))
        resumed = subprocess.run(script + ["--out-dir", tmp] + common, capture_output=True, text=True)
        computed = [ln for ln in resumed.stdout.splitlines() if ln.startswith("computed ")]
        cached = [ln for ln in resumed.stdout.splitlines() if ln.startswith("cached ")]
    s2 = [s for s in summary if s["label"] == "r=11,dth=1/10000"]
    ok = (good.returncode == 0 and names == sorted(nr.OUTPUTS)
          and heads.get("rows") == nr.ROW_COLS and heads.get("summary") == nr.SUMMARY_COLS
          and heads.get("selection") == nr.SELECTION_COLS and heads.get("cost") == nr.COST_COLS
          and heads.get("points") == nr.POINT_COLS
          and len(s2) == 4 and all(s["verified"] == "False" for s in s2)
          and bad.returncode != 0 and after == before
          and resumed.returncode == 0 and len(computed) == 1 and len(cached) == 1)
    return ok, (good.returncode, names, [s["verified"] for s in s2], bad.returncode, after == before,
                resumed.returncode, len(computed), len(cached), good.stderr[-300:] if good.returncode else "")


check("emulation", c_emulation)
check("c1", c_c1)
check("rule", c_rule)
check("shifts", c_shifts)
check("selection", c_selection)
check("escalated", c_escalated)
check("selection-tables", c_selection_tables)
check("noise", c_noise)
check("reference", c_reference)
check("map", c_map)
check("spectre-points", c_spectre_points)
check("benchmark", c_benchmark)
check("end-to-end", c_end_to_end)
sys.exit(0 if results and all(results) else 1)
