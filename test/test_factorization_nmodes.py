"""Regression test for the kernel-factored n-mode workflow
(inspectre/factorization.py).

The seven-channel calc_m_split (A + L*ln(alpha) + sum_q Pq/alpha^q, all
channels analytic across the particle) lets the n-modes of any calc_m output
be assembled by convolution: A direct, plus L and P1..P5 convolved against the
six universal scalar kernels {ln alpha, 1/alpha, ..., 1/alpha^5}.  This gates
that assembly against PANEL_GL (Inspectre.panel_nmodes_fast), the same
reference test/test_panel_gl.py trusts.

Criterion mirrors test_panel_gl.py:69 -- relative where the amplitude is alive,
absolute where the reference spectrum has decayed into its own noise floor.
PANEL_GL's src amplitude passes through spectral nulls (|X| ~ 1e-6) where only
absolute agreement is meaningful; the convolution itself is exact to the
reference's ~1e-11 absolute floor (verified KG- and NB-independent).

Run inside the spectre conda env from the inspectre project root:
    PYTHONPATH=.:../effectivesource:../kerrgeodesic python test/test_factorization_nmodes.py
"""
import math
import sys

import numpy as np

from inspectre.inspectre import Inspectre
import inspectre.factorization as fz

A, P, E, M = 0.5, 10.0, 0.3, 2

failures = 0


def check(name, cond, detail=""):
    global failures
    print(f"[{'ok ' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures += 1


insp = Inspectre(spin=A, semilatus_rectum=P, eccentricity=E)
# canonical near-orbit interior field point (same geometry as
# test_panel_gl.py "int_near"): closest-approach radius, small dtheta.
r_f = P / (1 + 0.5 * E)
th_f = math.pi / 2 + 0.01
n_list = list(range(0, 129, 4))

# per-output pass tolerances: (rel bound for n<=64, abs floor).  A mode passes
# if EITHER holds -- rel where alive, abs at spectral nulls.
TOL = {"PhiS": (1e-8, 1e-10), "dPhiS_r": (1e-8, 1e-10), "src": (1e-6, 1e-10)}

for output, (rel_tol, abs_tol) in TOL.items():
    rep = fz.compare_to_panel(insp, M, r_f, th_f, n_list, output=output,
                              NB=4096, NK=1 << 20, KG=400)
    nl = rep["n_list"]
    dabs = np.abs(rep["X_conv"] - rep["X_panel"])
    rel = rep["rel"]
    ok = (rel < rel_tol) | (dabs < abs_tol)
    le64 = nl <= 64
    worst = int(np.argmax(np.where(le64, rel, 0.0)))
    check(f"{output} n<=64 convolution vs PANEL_GL",
          bool(ok[le64].all()),
          f"max rel(n<=64)={rel[le64].max():.1e} max abs={dabs[le64].max():.1e} "
          f"(worst rel at n={nl[worst]}, |panel|={abs(rep['X_panel'][worst]):.1e})")

# ---------------------------------------------------------------------------
# Pure-C port (inspectre_fact_nodes_* via Inspectre.fact_nmodes_fast): one
# build, all outputs from the same handle. Same criterion vs PANEL_GL.
# ---------------------------------------------------------------------------
C_IDX = {"PhiS": (0, 0), "dPhiS_r": (1, 2), "src": (2, 0)}

res_c = insp.fact_nmodes_fast(M, n_list, r_f, th_f, NB=4096, NK=1 << 20, KG=400)
for output, (rel_tol, abs_tol) in TOL.items():
    blk, comp = C_IDX[output]
    X_c = np.array([complex(r[blk][comp], r[blk][comp + 1]) for r in res_c])
    X_pan = fz.panel_truth(insp, M, r_f, th_f, n_list, output=output)
    dabs = np.abs(X_c - X_pan)
    rel = dabs / np.maximum(np.abs(X_pan), 1e-300)
    nl = np.asarray(n_list)
    le64 = nl <= 64
    ok = (rel < rel_tol) | (dabs < abs_tol)
    check(f"{output} n<=64 C fact_nmodes_fast vs PANEL_GL",
          bool(ok[le64].all()),
          f"max rel(n<=64)={rel[le64].max():.1e} max abs={dabs[le64].max():.1e}")

print(f"\n{failures} failure(s)")
sys.exit(1 if failures else 0)
