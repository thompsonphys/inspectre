# Batched m-mode projection of the effective source

Status: proposed, 2026-09-22. Supersedes the extended-precision plan for `src`.

## Problem

The m-mode effective source `src_m(r,theta,t)` is currently built analytically, as a
finite sum over a Laurent representation of the puncture:

    src_m = ( sum_n T_n S_|m-n| ) exp(-i m (phi_p + c dr))

with `S_mu` the q=5 kernel and `T_n` the Laurent coefficients of the numerator over
`s2^(11/2)`, `s2 = alpha + beta sin^2(dphib/2)`.

That sum is ill-conditioned. `S_mu ~ alpha^(-11/2)`, which is enormous near the
particle, while `src_m` stays finite, so the terms are orders of magnitude larger
than their sum. Writing

    kappa = sum_n |T_n S_|m-n|| / |sum_n T_n S_|m-n||

the double-precision error is `kappa * eps`, verified to within 18x while the error
itself ranges over fourteen decades. No arithmetic site owns it: the block
polynomials are accurate to 1e-16, the Laurent assembly does not cancel, and the
wave-operator weighted sum costs one digit. `kappa` amplifies all of them equally.

Measured consequences over the worldtube (a=0.5, p=10, e=0.3):

- `ei_table` (`calc_m_offset`): kappa saturates at 1.85e8; worst src error 3.1e-05.
- `legendre` (Laurent + downward recurrence): kappa from 2e2 at C1=0.35 to 8e16 at
  C1=3.6e-4.
- The two fail in complementary regimes and both are poor in the crossover band,
  which is 13% (m=20) to 45% (m=5) of the worldtube.

Extended precision does not fix this. It buys 2.5-3 decades against a defect that
reaches 1e-3, and `PhiS` never needed it at all (kappa = 1 exactly).

## Approach

Project numerically in the field point's azimuthal coordinate instead. `phi` here is
the **field point** angle, not the particle's `phi_p(t)`: at fixed time the particle
is stationary, so `dr` and `dtheta` are constant across the sweep and only
`dphi = phi_f - phi_p` varies.

    src_m = (1/N) sum_k src(dr, dtheta, dphi_k) exp(-i m dphi_k) * 2 pi exp(-i m phi_p)

This is better conditioned by construction, because it builds `src_m` from samples of
`src` rather than from terms of size `alpha^(-11/2)`. Measured `kappa_quad` is 1e2 to
4e4 against 1e9 to 1e13 for the analytic sum.

`src(phi)` is analytic and 2pi-periodic, so the trapezoid rule converges
geometrically at rate `exp(-2 N sqrt(C1))`, set by the complex singularity at
`dphib = +- 2i asinh sqrt(alpha/beta)`.

## The node count

    N* = max( 37 / sqrt(C1), 5m/2, 64 ),   rounded up to a power of two

The first term is machine precision from the convergence rate; the second is Nyquist.
The power-of-two rounding is for FFTW, which costs 3.9 us at N=1172 against 1.1 us at
N=2048.

Validated: errors are flat across N*, 2N* and 4N*, so N* is correctly sized and the
residual is conditioning rather than resolution.

## Algorithm

Per (field point, time slice):

1. `dr = r_f - r_p`, `dtheta = theta_f - theta_p`, `C1 = alpha/beta`, `N = N*(C1, m_max)`.
2. Compute the block coefficients `P[0..d]`, `Q[0..d]` once, `d ~ 14`. Every component
   has the form `P(dQ^2) + dR Q(dQ^2)`, and the wave-operator weights depend only on
   `r, theta`, so the ten components collapse to one pair.
3. For each k: `dQ_k`, `dR_k` by angle addition from tabulated `sin(pi k/N)`,
   `cos(pi k/N)`; two Horner evaluations; divide by `s2^(11/2)`.
4. Real-to-complex FFT. Bin m is `src_m` up to the `2 pi exp(-i m phi_p)` factor.

Steps 2-4 give every m up to Nyquist in one pass, so cost is independent of mode count.

The `sin(pi k/N)` tables depend only on N, so they are shared across all field points
and all time slices.

The n-mode transform is unchanged: `src_m(t)` per slice, then the existing quadrature
over the radial period.

## C surface

    void effsource_equatorial_ctx_src_m_sweep(
        struct effsource_equatorial_ctx *ctx,
        double dr, double dtheta, int nphi, double *src_out);

`src_out[nphi]` receives `src` at `dphi_k = 2 pi k / nphi`. The caller owns the FFT;
FFTW is already linked by both inspectre and SpECTRE.

Nothing else is added. `calc_offset` stays as the unbatched path and the reference the
sweep is graded against.

## Cost

Per sample, measured (gcc -O2 -march=native):

| variant | ns |
|---|---|
| `calc_offset` today, 16 outputs, expanded | 223 |
| hoisted blocks, src only, tabulated trig, degree 14 | 4.6 |

Three savings compound: `calc_offset` computes sixteen quantities where one is used;
the `(dr,dtheta)` blocks are constant across the sweep; the trig is tabulable.

Against twenty analytic modes at 1108 ns each:

| N* | sample | FFT | total | vs analytic |
|---|---|---|---|---|
| 64 | 0.29 us | 0.04 us | 0.34 us | 66x cheaper |
| 148 | 0.68 us | 0.37 us | 1.05 us | 21x cheaper |
| 1172 | 5.39 us | 3.91 us | 9.30 us | 2.4x cheaper |

`set_particle*` costs 955 us but is shared across every field point in a slice, so it
amortizes away above ~64 field points and does not enter this comparison.

## Accuracy

Unbatched, over 576 (spin, orbit, mix, C1, m) combinations spanning C1 from 0.6 to
1e-5 at m <= 20: worst src error **1.24e-08**, typical 1e-11. The analytic route's
worst over the same domain is 3.1e-05.

End to end through the n-mode transform: per-slice 1e-13 to 1.6e-11, beating the
analytic path at every slice and every m. `kappa_n = mean|src_m(t)| / |src_mn|` is
O(1-10) for n <= 16, so the absolute error in `src_mn` is `~1e-11 * mean|src_m(t)|`,
uniform in n.

## Limits

- **`PhiS` keeps the analytic path.** Through the projection it degrades to 1.27e-06
  at m=20, C1=0.6: `s2^(-7/2)` is smoother than `s2^(-11/2)`, so its modes sit further
  below the peak and `kappa_quad` is larger. The analytic path has kappa = 1.
- **A ceiling at `m sqrt(C1) > 8.5`**, where `kappa_quad ~ r^(-m)` with
  `r = z - sqrt(z^2-1)`, `z = 1 + 2 C1`. It does not bite at m <= 20 (worst 4.2e-09 at
  the C1=0.6 corner) but would above m ~ 25. The `legendre` branch is accurate exactly
  there, so it is the fallback if the mode cap rises.
- Deep near zone costs nodes: N* = 81566 at the worldtube minimum C1 = 2.06e-7, one
  point in 61500. Accuracy is fine there; only the node count grows.

## Validation

1. Sweep output against `calc_offset` sample by sample, bitwise-close.
2. `src_m` from the sweep against the mpmath reference across spin, p, e, dr/dtheta
   mix, C1 and m, reusing the existing 576-combination grid.
3. `src_mn` end to end against `panel_nmodes_fast`.
4. The N* rule re-checked: error flat across N*, 2N*, 4N*.

The mpmath reference is independent of everything being changed:
`test_reference_selfstanding.py` establishes it from the pointwise evaluator alone.

## Scope

Changes: one new C entry point and its block-coefficient helper; inspectre grows a
sweep-based `src_m` path.

Unchanged: `PhiS` in all forms, the n-mode quadrature, `calc_m_split` and
`calc_m_extended` (the factorization node build depends on them), the coefficient
translation units and their -O0 build.

SpECTRE is out of scope until this is proven in inspectre.

## Open questions

- Degree `d` of the combined block polynomials is estimated at ~14 from the Laurent
  index range; the port fixes it exactly. Cost scales mildly: 2.2 ns at d=4, 4.6 at
  d=14, 6.6 at d=20.
- Whether the sweep should also return `dPhiS`, which the elliptic solver needs at the
  worldtube boundary. Not measured; the projection's conditioning for derivatives is
  unknown and may differ from both `src` and `PhiS`.
