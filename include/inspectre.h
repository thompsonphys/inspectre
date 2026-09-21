#ifndef INSPECTRE_H
#define INSPECTRE_H

#include "../../kerrgeodesic/korb.h"
#include "../../effectivesource/effsource.h"
#include "../../effectivesource/effsource_equatorial.h"

/* ---------------------------------------------------------------------------
 * Orbit / n-mode helpers (implemented in src/inspectre_lib.c)
 * ------------------------------------------------------------------------- */

/* X-function used in the equatorial four-velocity. */
double xFuncInsp(double a, double p, double e);

/* Radial four-velocity u^r from the radial phase psi. */
double fourVel(double psi, double a, double p, double e, double E);

/* Circular limit: korb leaves Vr / Yr unset and wr = 0 for e == 0, so supply
   the epicyclic frequency instead. _circular_fix patches wr / Yr / Vr in place
   after korb_getparams; no-op when eccentric. */
double inspectre_epicyclic_frequency(const korb_params *orbpar);
double inspectre_radial_frequency(const korb_params *orbpar);
double inspectre_radial_mino_period(const korb_params *orbpar);
int    inspectre_orbit_circular_fix(korb_params *orbpar);

/* Frequency shift exp(i Omega t) applied to a complex (re, im) pair, where
   Omega = mMode*omegaPhi + nMode*omegaR. */
double frequencyShiftReal(double t, double omegaPhi, double omegaR,
                          double fieldRE, double fieldIM, int mMode, int nMode);
double frequencyShiftImag(double t, double omegaPhi, double omegaR,
                          double fieldRE, double fieldIM, int mMode, int nMode);

/* Apply the frequency shift to the field (2), its derivatives (8) and the
   effective source (2), producing the n-mode integrands. */
void generateNModeIntegrands(double t, double omegaPhi, double omegaR,
                             int mMode, int nMode,
                             double *Field, double *FieldDeriv, double *EffSrc,
                             double *nModeField, double *nModeFieldDeriv,
                             double *nModeEffSrc);

/* ---------------------------------------------------------------------------
 * (1) Effective source + puncture at a field point and time.
 *
 * The particle position is computed internally from the orbit, so the caller
 * only supplies a field point and a time. Uses the equatorial context API.
 * ------------------------------------------------------------------------- */

/* Field point carrying the polar offset directly.

   dtheta = theta - pi/2 is the only way theta ever enters, and routing it
   through an absolute theta caps its relative accuracy at 1.1e-16/dtheta
   (6e-9 at dtheta = 1e-8). Since the kernel is
   alpha = alpha20 dr^2 + alpha02 dtheta^2, the dtheta term floors alpha, so dr
   need never be resolved below ~dtheta and the long-double dr formed internally
   suffices -- but nothing floors dtheta itself, and an error in it is
   systematic across the whole orbit. Hence dtheta is supplied, not derived.

   phi is currently read by no routine; kept for symmetry. */
typedef struct {
    double r;
    double dtheta;
    double phi;
} inspectre_field_point;

/* Primary: parameterized by Mino time lambda. */
void inspectre_eval_at_lambda_fp(struct effsource_equatorial_ctx *ctx, int mMode,
                              const inspectre_field_point *fp, double lambda,
                              korb_params *orbpar, double a, double p, double e,
                              double *PhiS, double *dPhiS, double *ddPhiS,
                              double *src);

/* As _fp, taking an absolute-coordinate field point (dtheta = theta - pi/2). */
void inspectre_eval_at_lambda(struct effsource_equatorial_ctx *ctx, int mMode,
                              struct coordinate *xField, double lambda,
                              korb_params *orbpar, double a, double p, double e,
                              double *PhiS, double *dPhiS, double *ddPhiS,
                              double *src);

/* Invert t = korb_tfromla(lambda) for lambda (monotonic) via a Brent solver. */
double inspectre_lambda_from_t(double t, korb_params *orbpar);

/* Wrapper: parameterized by coordinate time t (root-solves lambda(t)). */
void inspectre_eval_at_time_fp(struct effsource_equatorial_ctx *ctx, int mMode,
                            const inspectre_field_point *fp, double t,
                            korb_params *orbpar, double a, double p, double e,
                            double *PhiS, double *dPhiS, double *ddPhiS,
                            double *src);

void inspectre_eval_at_time(struct effsource_equatorial_ctx *ctx, int mMode,
                            struct coordinate *xField, double t,
                            korb_params *orbpar, double a, double p, double e,
                            double *PhiS, double *dPhiS, double *ddPhiS,
                            double *src);

/* Reference evaluator: the seven analytic kernel channels of calc_m_split
   reassembled in long double, with the P1/alpha + P2/alpha^2 pair fused as
   (P1 alpha + P2)/alpha^2 so their ~3.6e6 mutual cancellation is paid once at
   extended precision. Same outputs as _eval_at_lambda_fp minus ddPhiS, which
   calc_m leaves partly NAN. Slower; for grading the double path, not for
   production. */
void inspectre_eval_extended_at_lambda_fp(struct effsource_equatorial_ctx *ctx,
                              int mMode,
                              const inspectre_field_point *fp, double lambda,
                              korb_params *orbpar, double a, double p, double e,
                              double *PhiS, double *dPhiS, double *src);

/* Source-evaluation counter. Incremented once per puncture/source evaluation by
   every path (adaptive, node builds, extended). Reset before a batch, read after,
   separate evaluation count from wall time -- QAG additionally pays a Brent
   inversion per evaluation, which timing alone conflates with cost. */
void inspectre_eval_count_reset(void);
long inspectre_eval_count(void);

/* Evaluator selected by inspectre_eval_at_lambda_fp, hence by every quadrature
   that samples through it: 0 = calc_m_offset (double), 1 = calc_m_extended
   (long-double channel reassembly). Set around a batch to swap the evaluator
   without changing the quadrature, so a double-vs-extended comparison holds the
   node set fixed. The extended evaluator yields no second derivatives, so
   ddPhiS is zeroed.
   INSPECTRE_INTEG_FACT_CONV reads calc_m_split directly and is unaffected. */
void inspectre_eval_precision_set(int extended);
int  inspectre_eval_precision_get(void);

/* ---------------------------------------------------------------------------
 * (2) n-mode Fourier amplitude integration over one radial period.
 * ------------------------------------------------------------------------- */

/* Everything the gsl_function integrand needs, plus which scalar to return.
   component indexes the 12 scalars [field re/im (0,1), deriv (2..9),
   src re/im (10,11)]. */
typedef struct {
    struct effsource_equatorial_ctx *ctx;
    int    mMode, nMode, component;
    const inspectre_field_point *fp;
    korb_params *orbpar;
    double a, p, e, omegaPhi, omegaR;
} inspectre_nmode_params;

/* gsl_function: evaluate (1) at coordinate time t, frequency-shift, return the
   single scalar selected by params->component. */
double inspectre_nmode_integrand(double t, void *params);

/* gsl_function: as inspectre_nmode_integrand but parameterized by Mino time
   lambda. Evaluates (1) at lambda directly (no Brent inversion), frequency-shifts
   using t(lambda), and multiplies by the time Jacobian dt/dlambda so that the
   lambda-integral over [0, Vr] equals the coordinate-time integral over [0, Tr]. */
double inspectre_nmode_integrand_lambda(double lambda, void *params);

enum { INSPECTRE_INTEG_QAG        = 0,
       INSPECTRE_INTEG_TIMESERIES = 1,   /* trapezoid over the samples */
       INSPECTRE_INTEG_SIMPSON    = 2,   /* composite Simpson over the samples */
       INSPECTRE_INTEG_SPLINE     = 3,   /* gsl cubic-spline quadrature */
       INSPECTRE_INTEG_QAG_MINO   = 4,   /* adaptive QAG over Mino time lambda (no Brent) */
       INSPECTRE_INTEG_MINO_SPLINE= 5,   /* graded-lambda mesh at closest approach + spline */
       INSPECTRE_INTEG_PANEL_GL   = 6,   /* panels split at closest approach + Gauss-Legendre */
       INSPECTRE_INTEG_FACT_CONV  = 7 }; /* seven-channel kernel factorization + convolution */

/* Driver. mode == INSPECTRE_INTEG_QAG: adaptive GSL QAG over coordinate time,
   evaluating (1) on demand (each eval Brent-inverts lambda(t)).
   INSPECTRE_INTEG_TIMESERIES / SIMPSON / SPLINE integrate the precomputed samples
   (tSamples + interleaved fieldSamples[2], derivSamples[8], srcSamples[2] per
   point): trapezoid, composite Simpson, gsl cubic-spline quadrature respectively.

   INSPECTRE_INTEG_QAG_MINO and INSPECTRE_INTEG_MINO_SPLINE are SELF-CONTAINED:
   they sample the source internally over Mino time lambda (no Brent inversion)
   and ignore tSamples/field/deriv/srcSamples (pass NULL). QAG_MINO is adaptive;
   MINO_SPLINE places `nSamples` nodes graded toward closest approach and applies
   the gsl spline quadrature. Outputs the complex n-mode amplitudes (re/im
   interleaved like dPhiS).

   Both QAG modes raise `epsabs` per component to the roundoff level the
   integrand carries; a component whose exact integral vanishes cannot be
   certified below that, and asking for less makes QAG bisect to its limit. */
void inspectre_integrate_nmode_fp(int mode,
        struct effsource_equatorial_ctx *ctx, int mMode, int nMode,
        const inspectre_field_point *fp, korb_params *orbpar,
        double a, double p, double e, double omegaPhi, double omegaR,
        double epsabs, double epsrel,
        double *tSamples, double *fieldSamples, double *derivSamples,
        double *srcSamples, int nSamples,
        double *nModePhiS, double *nModeDPhiS, double *nModesrc);

void inspectre_integrate_nmode(int mode,
        struct effsource_equatorial_ctx *ctx, int mMode, int nMode,
        struct coordinate *xField, korb_params *orbpar,
        double a, double p, double e, double omegaPhi, double omegaR,
        double epsabs, double epsrel,
        double *tSamples, double *fieldSamples, double *derivSamples,
        double *srcSamples, int nSamples,
        double *nModePhiS, double *nModeDPhiS, double *nModesrc);

/* QAG iteration-limit tracking. inspectre_integrate_nmode increments an internal
   counter each time a QAG / QAG_MINO component integration returns GSL_EMAXITER.
   Reset before a batch, read after, to flag unreliable adaptive integrations
   without changing the integrate signature or parsing stderr. */
void inspectre_qag_limit_reset(void);
long inspectre_qag_limit_count(void);

/* ---------------------------------------------------------------------------
 * Reusable Mino-time graded-mesh samples.
 *
 * The MINO_SPLINE node geometry and the RAW (unshifted, un-Jacobian-weighted)
 * source samples at those nodes are independent of the mode number n -- only
 * the per-n frequency shift differs. Build the samples once per field point
 * with inspectre_mino_samples_build, then call inspectre_mino_samples_integrate
 * once per n (cheap: frequency-shift + spline quadrature, no source evals).
 * This is exactly what the self-contained INSPECTRE_INTEG_MINO_SPLINE branch of
 * inspectre_integrate_nmode does internally for a single n.
 * ------------------------------------------------------------------------- */
typedef struct {
    int     n;          /* node count (= nSamples) */
    double  Tr, Vr;
    double *lam;        /* [n]    graded Mino-time nodes */
    double *t;          /* [n]    korb_tfromla(lam[i]) */
    double *J;          /* [n]    dt/dlambda at lam[i] */
    double *raw;        /* [12*n] raw components, contiguous-per-component:
                           raw[c*n + i]; order PhiS re/im (0,1), dPhiS (2..9),
                           src re/im (10,11) */
} inspectre_mino_samples;

/* Build the graded mesh + raw source samples (the only source-evaluating step).
   Caller frees with inspectre_mino_samples_free. */
void inspectre_mino_samples_build_fp(struct effsource_equatorial_ctx *ctx, int mMode,
        const inspectre_field_point *fp, korb_params *orbpar,
        double a, double p, double e, int nSamples,
        inspectre_mino_samples *out);

void inspectre_mino_samples_build(struct effsource_equatorial_ctx *ctx, int mMode,
        struct coordinate *xField, korb_params *orbpar,
        double a, double p, double e, int nSamples,
        inspectre_mino_samples *out);

void inspectre_mino_samples_free(inspectre_mino_samples *s);

/* Frequency-shift the prebuilt samples for mode n and spline-integrate. No
   source evals; reuses the n-independent samples from _build. Outputs the
   complex n-mode amplitudes (re/im interleaved like dPhiS). */
void inspectre_mino_samples_integrate(const inspectre_mino_samples *s,
        int mMode, int nMode, double omegaPhi, double omegaR,
        double *nModePhiS, double *nModeDPhiS, double *nModesrc);

/* As _integrate, but integrate g(t) directly over the non-uniform node times
   t[] with a plain sample rule instead of the lambda-spline quadrature (no
   Jacobian weight). `rule` is one of INSPECTRE_INTEG_TIMESERIES (trapezoid),
   INSPECTRE_INTEG_SIMPSON, or INSPECTRE_INTEG_SPLINE (spline over t). Tests how
   the graded mesh fares under a non-spline non-uniform quadrature. */
void inspectre_mino_samples_integrate_rule(const inspectre_mino_samples *s,
        int mMode, int nMode, double omegaPhi, double omegaR, int rule,
        double *nModePhiS, double *nModeDPhiS, double *nModesrc);

/* ---------------------------------------------------------------------------
 * Panel Gauss-Legendre nodes split at the particle's closest approach.
 *
 * The n-mode integrand over one radial period is smooth except near the Mino
 * times lambda_c where the particle passes closest to the field point: for
 * field points the orbit actually crosses (theta = pi/2, r in [r_min, r_max])
 * the source is only C^0 there and the puncture has a log(distance) profile;
 * for off-orbit points the integrand is analytic but peaked on the scale of
 * the minimum distance. Both lambda_c are known analytically from the orbit
 * (psi_c = arccos((p/r_f - 1)/e)), so instead of grading a global mesh we
 * split [0, Vr] into panels at the lambda_c and refine geometrically toward
 * them (panel widths halving down to the peak's own lambda-scale, capped at
 * maxLevels halvings), with a fixed-order Gauss-Legendre rule per panel.
 * Off-orbit this resolves the peak exactly like a sinh map; at a crossing the
 * geometric stack converges exponentially in the level count for the C^0/log
 * endpoint behavior where a single global rule is stuck at low order.
 *
 * Like inspectre_mino_samples, the node set and the raw samples are
 * n-independent: build once per field point, then integrate per n at the cost
 * of a weighted phase sum (no further source evaluations).
 * ------------------------------------------------------------------------- */
typedef struct {
    int     n;          /* total node count */
    double  Tr, Vr;
    int     nBreak;     /* number of distinct closest-approach breakpoints (1 or 2) */
    int     nNonFinite; /* non-finite raw samples zeroed during the build (deep
                           near-zone source evals on innermost sliver nodes) */
    double  lamBreak[2];/* breakpoint Mino times in [0, Vr) */
    double  dlam[2];    /* lambda-scale of each peak (0 => exact crossing) */
    double *lam;        /* [n] node Mino times (in [lamBreak[0], lamBreak[0]+Vr]) */
    double *t;          /* [n] coordinate time at node, folded to the principal
                           period (the full n-mode integrand is Tr-periodic) */
    double *J;          /* [n] dt/dlambda at node */
    double *w;          /* [n] quadrature weight (panel-scaled GL weight) */
    double *raw;        /* [12*n] raw components, contiguous-per-component, same
                           layout as inspectre_mino_samples.raw */
} inspectre_panel_nodes;

/* Build breakpoints, panels, GL nodes and raw source samples. `order` is the
   Gauss-Legendre points per panel (e.g. 16); `maxLevels` caps the geometric
   refinement toward each breakpoint (e.g. 40 => innermost panel ~ 1e-12 Vr).
   `nMax` is the largest |n| the node set must resolve: panels are subdivided
   until (|m| wphi + nMax wr) * dt_panel <= order, the Gauss-Legendre
   resolution bound for the oscillatory carrier -- integrating beyond nMax
   with these nodes silently degrades. Caller frees with
   inspectre_panel_nodes_free. */
void inspectre_panel_nodes_build_fp(struct effsource_equatorial_ctx *ctx, int mMode,
        const inspectre_field_point *fp, korb_params *orbpar,
        double a, double p, double e, int order, int maxLevels, int nMax,
        double omegaPhi, double omegaR,
        inspectre_panel_nodes *out);

void inspectre_panel_nodes_build(struct effsource_equatorial_ctx *ctx, int mMode,
        struct coordinate *xField, korb_params *orbpar,
        double a, double p, double e, int order, int maxLevels, int nMax,
        double omegaPhi, double omegaR,
        inspectre_panel_nodes *out);

void inspectre_panel_nodes_free(inspectre_panel_nodes *s);

/* Frequency-shift the prebuilt panel samples for mode n and sum the quadrature.
   No source evals. Outputs the complex n-mode amplitudes (re/im interleaved
   like dPhiS). */
void inspectre_panel_nodes_integrate(const inspectre_panel_nodes *s,
        int mMode, int nMode, double omegaPhi, double omegaR,
        double *nModePhiS, double *nModeDPhiS, double *nModesrc);

/* ---------------------------------------------------------------------------
 * Kernel-factorization n-modes (C port of inspectre/factorization.py).
 *
 * Every calc_m output splits exactly (effectivesource calc_m_split) into seven
 * channels {A, L, P1..P5},
 *     X_m(t) = A(t) + L(t) ln a(t) + sum_{q=1..5} Pq(t)/a(t)^q,
 *     a(t)   = alpha20(t) dr(t)^2 + alpha02(t) dtheta^2,
 * with every channel analytic (narrow-spectrum) along the worldline at a legal
 * field point. The n-modes then follow by convolution against the FFTs of the
 * six universal scalar kernels {ln a, 1/a, ..., 1/a^5} (orbit quantities only):
 *     X_n = A^(n) + sum_ch sum_{k=-KG..KG} ch^(k) K_ch^(n-k).
 * Like the panel nodes, the build is the only source-evaluating step (NB
 * channel-split evaluations); each n then costs one 6x(2KG+1) complex dot per
 * component. Kernel FFT coefficients are stored trimmed to |j| <= nMax + KG,
 * so integration is exact for |n| <= nMax and silently truncated beyond.
 * ------------------------------------------------------------------------- */
typedef struct {
    int     NB;         /* base-grid points (uniform in t; forced even) */
    int     NK;         /* dense kernel FFT length used during the build */
    int     KG;         /* convolution half-width (clamped to NB/2) */
    int     nMax;       /* largest |n| the stored kernel coefficients resolve */
    int     nKmax;      /* kernel coefficient bound = nMax + KG */
    int     nNonFinite; /* non-finite channel/kernel samples zeroed in the build */
    double  Tr, Vr;
    double *chat;       /* [2*7*6*NB] channel Fourier coefficients, re/im
                           interleaved: chat[2*((ch*6 + c)*NB + j)] with
                           ch in {A,L,P1..P5}, c in {PhiS, dPhiS pairs 0..3,
                           src}, j the FFT bin (numpy ifft normalization) */
    double *khat;       /* [2*6*(2*nKmax+1)] kernel Fourier coefficients,
                           re/im interleaved: khat[2*(kq*(2*nKmax+1) + j +
                           nKmax)] with kq in {ln a, 1/a..1/a^5}, j in
                           [-nKmax, nKmax] */
} inspectre_fact_nodes;

/* Sample the seven split channels of every output on the NB-point uniform-t
   grid (the only source-evaluating step; one calc_m_split per node), FFT them,
   and FFT the six kernels on an NK-point spectrally-resampled dense grid.
   NK is rounded up so that nMax + KG kernel coefficients exist; defaults that
   reproduce the validated Python workflow are NB = 4*nMax, NK = 1<<20,
   KG = 400. Caller frees with inspectre_fact_nodes_free. */
void inspectre_fact_nodes_build_fp(struct effsource_equatorial_ctx *ctx, int mMode,
        const inspectre_field_point *fp, korb_params *orbpar,
        double a, double p, double e, int NB, int NK, int KG, int nMax,
        double omegaPhi, double omegaR,
        inspectre_fact_nodes *out);

void inspectre_fact_nodes_build(struct effsource_equatorial_ctx *ctx, int mMode,
        struct coordinate *xField, korb_params *orbpar,
        double a, double p, double e, int NB, int NK, int KG, int nMax,
        double omegaPhi, double omegaR,
        inspectre_fact_nodes *out);

void inspectre_fact_nodes_free(inspectre_fact_nodes *s);

/* n-mode amplitudes for mode n by the seven-channel convolution. No source
   evals. Exact for |n| <= nMax (beyond, missing kernel coefficients are
   treated as zero). Outputs the complex amplitudes (re/im interleaved like
   dPhiS); the mixed/second derivatives calc_m leaves NAN are not produced. */
void inspectre_fact_nodes_integrate(const inspectre_fact_nodes *s, int nMode,
        double *nModePhiS, double *nModeDPhiS, double *nModesrc);

/* FFT source n-mode amplitudes: one FFTW transform of S_m sampled on a uniform
   periodic t-grid of N points yields the complex amplitude A_n for every n at
   once. outRe[k]/outIm[k] (length N) hold mode n = (k <= N/2) ? k : k - N; for a
   requested n read bin ((n % N) + N) % N. Requires |n| <= N/2 to be resolved. */
void inspectre_fft_source_nmodes_fp(struct effsource_equatorial_ctx *ctx, int mMode,
        const inspectre_field_point *fp, korb_params *orbpar,
        double a, double p, double e, double omegaPhi, double omegaR,
        int N, double *outRe, double *outIm);

void inspectre_fft_source_nmodes(struct effsource_equatorial_ctx *ctx, int mMode,
        struct coordinate *xField, korb_params *orbpar,
        double a, double p, double e, double omegaPhi, double omegaR,
        int N, double *outRe, double *outIm);

#endif /* INSPECTRE_H */
