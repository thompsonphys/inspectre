"""Chebyshev cache for fixed-(m, n) n-modes, with panel fallback near the orbit.

SpECTRE consumes n-modes at AMR-chosen field points: fixed (m, n), arbitrary
(r, theta) inside/on the worldtube, not reusable between solves.  Per-point
panel integration (Inspectre.panel_nmodes_fast) costs a full node build per
point.  But the amplitude A_n(r, theta) is analytic in the field point off the
particle's sweep segment {theta = pi/2, r in [r_min, r_max]}, with variation
scale ~ the distance d to the segment.  So: build tensor-Chebyshev interpolants
of A_n on patches that keep clear of the segment (one panel build per node,
each serving the whole n list), then answer arbitrary queries by evaluation.
Very near the segment interpolation degrades (d -> 0 pulls the function's
complex singularities onto the patch), so `nmodes_at` switches to direct
per-point panel evaluation inside a clearance band d < d_switch.

Interpolation acts on Re/Im of the complex amplitudes (smooth), never on
magnitudes (spectral nulls).  Calibration of node counts and d_switch:
notebooks/chebcache_error_study.ipynb.  Regression: test/test_chebcache.py.

Run env: spectre conda env, PYTHONPATH=.:../effectivesource:../kerrgeodesic.
"""
import time
import warnings

import numpy as np
from numpy.polynomial import chebyshev as nch

# complex component c <-> (block, re-index) of a panel_nmodes_fast tuple
# (PhiS[2], dPhiS[8], src[2]); c = 0 PhiS, 1..4 dPhiS pairs, 5 src.
_NCOMP = 6


def segment_distance(r, th, r_min, r_max):
    """Euclidean meridional distance to the sweep segment.

    Chord metric of the C mesh grading (insp_distance):
        d(r_p)^2 = r_p^2 + r^2 - 2 r_p r sin(th),
    minimized over r_p in [r_min, r_max]; the quadratic's vertex is
    r_p = r sin(th), so the minimizer is its clamp onto the interval.
    Vectorized over r, th (broadcast).
    """
    r = np.asarray(r, dtype=float)
    th = np.asarray(th, dtype=float)
    rp = np.clip(r * np.sin(th), r_min, r_max)
    d2 = rp * rp + r * r - 2.0 * rp * r * np.sin(th)
    return np.sqrt(np.maximum(d2, 0.0))


def _tuples_to_complex(res):
    """panel_nmodes_fast result for one n -> complex[_NCOMP]."""
    PhiS, dPhiS, src = res
    out = np.empty(_NCOMP, dtype=complex)
    out[0] = complex(PhiS[0], PhiS[1])
    for k in range(4):
        out[1 + k] = complex(dPhiS[2 * k], dPhiS[2 * k + 1])
    out[5] = complex(src[0], src[1])
    return out


def _complex_to_tuples(vals):
    """complex[_NCOMP] -> (PhiS[2], dPhiS[8], src[2]) like panel_nmodes_fast."""
    PhiS = [vals[0].real, vals[0].imag]
    dPhiS = []
    for k in range(4):
        dPhiS += [vals[1 + k].real, vals[1 + k].imag]
    src = [vals[5].real, vals[5].imag]
    return PhiS, dPhiS, src


class ChebCache:
    """Tensor-Chebyshev interpolant of A_n(r, theta) on one rectangular patch.

    One orbit, one m, a list of n; all six complex components (PhiS, the four
    dPhiS pairs, src) are cached from the same panel builds.
    """

    def __init__(self, insp, m, n_list, patch, Nr=16, Nth=16,
                 order=16, max_levels=40):
        r0, r1, th0, th1 = patch
        self.patch = tuple(patch)
        self.m = m
        self.n_list = [int(n) for n in n_list]
        self.Nr, self.Nth = Nr, Nth
        self.r_min = insp.semilatus_rectum / (1 + insp.eccentricity)
        self.r_max = insp.semilatus_rectum / (1 - insp.eccentricity)

        # first-kind Chebyshev points mapped onto the patch
        self.rn = 0.5 * (r0 + r1) + 0.5 * (r1 - r0) * nch.chebpts1(Nr)
        self.thn = 0.5 * (th0 + th1) + 0.5 * (th1 - th0) * nch.chebpts1(Nth)

        dmin = segment_distance(self.rn[:, None], self.thn[None, :],
                                self.r_min, self.r_max).min()
        if dmin <= 0.0:
            raise ValueError("patch touches the sweep segment; "
                             "interpolation cannot converge there")
        diam = float(np.hypot(r1 - r0, th1 - th0))
        if dmin < diam / 10.0:
            warnings.warn(f"patch clearance {dmin:.3g} < diameter/10 "
                          f"({diam / 10:.3g}); expect slow Chebyshev "
                          "convergence -- shrink the patch or raise N")
        self.clearance = float(dmin)

        # sample: one panel build per node serves every n in n_list
        vals = np.empty((len(self.n_list), _NCOMP, Nr, Nth), dtype=complex)
        t0 = time.perf_counter()
        for i, r in enumerate(self.rn):
            for j, th in enumerate(self.thn):
                res = insp.panel_nmodes_fast(m, self.n_list, float(r),
                                             float(th), order=order,
                                             max_levels=max_levels)
                for q in range(len(self.n_list)):
                    vals[q, :, i, j] = _tuples_to_complex(res[q])
        self.build_seconds = time.perf_counter() - t0
        self.vals = vals

        # 2D Chebyshev coefficients: invert the Vandermonde along each axis
        # (exact at first-kind points; N <= ~32 so lstsq is plenty stable)
        Vr = nch.chebvander(nch.chebpts1(Nr), Nr - 1)
        Vth = nch.chebvander(nch.chebpts1(Nth), Nth - 1)
        VrI = np.linalg.pinv(Vr)
        VthI = np.linalg.pinv(Vth)
        # coef[q, c] = VrI @ vals[q, c] @ VthI.T
        self.coef = np.einsum('ai,qcij,bj->qcab', VrI, vals, VthI)

    # -- queries --------------------------------------------------------------

    def contains(self, r, th):
        r0, r1, th0, th1 = self.patch
        return (r0 <= r <= r1) and (th0 <= th <= th1)

    def _xy(self, r, th):
        r0, r1, th0, th1 = self.patch
        return ((2.0 * r - (r0 + r1)) / (r1 - r0),
                (2.0 * th - (th0 + th1)) / (th1 - th0))

    def eval_complex(self, r, th):
        """complex[len(n_list), _NCOMP] at one point."""
        x, y = self._xy(r, th)
        out = np.empty((len(self.n_list), _NCOMP), dtype=complex)
        for q in range(len(self.n_list)):
            for c in range(_NCOMP):
                out[q, c] = nch.chebval2d(x, y, self.coef[q, c])
        return out

    def eval(self, r, th):
        """List over n_list of (PhiS[2], dPhiS[8], src[2]) tuples."""
        vals = self.eval_complex(r, th)
        return [_complex_to_tuples(vals[q]) for q in range(len(self.n_list))]

    def sup_norm(self):
        """max |value| over the sampled grid, shape (len(n_list), _NCOMP).

        The interpolation error is relative to THIS scale (per n and
        component), not to the pointwise value: a component whose amplitude
        ranges over decades across the patch (high n decays like
        exp(-n c d(r,theta))) cannot be recovered below err ~ tol * sup_norm
        at points where it is locally tiny.  Accuracy criteria should read
        abs_err < tol * sup_norm  OR  rel_err < tol.
        """
        return np.abs(self.vals).max(axis=(2, 3))

    def coeff_decay(self):
        """max |c_ab| over anti-diagonals a+b=k, per n: the convergence
        diagnostic (geometric decay <=> patch resolved)."""
        Nr, Nth = self.Nr, self.Nth
        K = Nr + Nth - 1
        out = np.zeros((len(self.n_list), K))
        mag = np.abs(self.coef).max(axis=1)          # worst component
        a = np.arange(Nr)[:, None] + np.arange(Nth)[None, :]
        for k in range(K):
            out[:, k] = mag[:, a == k].max(axis=1)
        return out


class ChebCacheSet:
    """Several patches + point -> patch lookup (first containing patch wins)."""

    def __init__(self, caches):
        self.caches = list(caches)
        if not self.caches:
            raise ValueError("empty cache set")
        m = {c.m for c in self.caches}
        nl = {tuple(c.n_list) for c in self.caches}
        if len(m) > 1 or len(nl) > 1:
            raise ValueError("all patches must share m and n_list")
        self.m = self.caches[0].m
        self.n_list = self.caches[0].n_list
        self.r_min = self.caches[0].r_min
        self.r_max = self.caches[0].r_max

    def which_patch(self, r, th):
        for k, cch in enumerate(self.caches):
            if cch.contains(r, th):
                return k
        return -1


def nmodes_at(insp, m, n_list, points, cacheset=None, d_switch=0.05,
              order=16, max_levels=40):
    """n-modes at arbitrary field points with cache/panel switching.

    points: iterable of (r, theta).  For each point:
      - inside a cache patch AND segment_distance >= d_switch -> Chebyshev
        evaluation ('cheb'),
      - otherwise -> direct per-point panel integration ('panel', exact).
    Returns (results, methods): results[i] is the panel_nmodes_fast-style list
    over n_list; methods[i] in {'cheb', 'panel'}.
    """
    n_list = [int(n) for n in n_list]
    if cacheset is not None and cacheset.n_list != n_list:
        raise ValueError("cacheset was built for a different n_list")
    if cacheset is not None and cacheset.m != m:
        raise ValueError("cacheset was built for a different m")
    r_min = insp.semilatus_rectum / (1 + insp.eccentricity)
    r_max = insp.semilatus_rectum / (1 - insp.eccentricity)

    results, methods = [], []
    for (r, th) in points:
        d = float(segment_distance(r, th, r_min, r_max))
        k = cacheset.which_patch(r, th) if cacheset is not None else -1
        if k >= 0 and d >= d_switch:
            results.append(cacheset.caches[k].eval(r, th))
            methods.append('cheb')
        else:
            results.append(insp.panel_nmodes_fast(
                m, n_list, float(r), float(th),
                order=order, max_levels=max_levels))
            methods.append('panel')
    return results, methods
