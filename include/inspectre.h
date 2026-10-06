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

/* Primary: parameterized by Mino time lambda. */
void inspectre_eval_at_lambda(struct effsource_equatorial_ctx *ctx, int mMode,
                              struct coordinate *xField, double lambda,
                              korb_params *orbpar, double a, double p, double e,
                              double *PhiS, double *dPhiS, double *ddPhiS,
                              double *src);

/* Invert t = korb_tfromla(lambda) for lambda (monotonic) via a Brent solver. */
double inspectre_lambda_from_t(double t, korb_params *orbpar);

/* Wrapper: parameterized by coordinate time t (root-solves lambda(t)). */
void inspectre_eval_at_time(struct effsource_equatorial_ctx *ctx, int mMode,
                            struct coordinate *xField, double t,
                            korb_params *orbpar, double a, double p, double e,
                            double *PhiS, double *dPhiS, double *ddPhiS,
                            double *src);

/* ---------------------------------------------------------------------------
 * (2) n-mode Fourier amplitude integration over one radial period.
 * ------------------------------------------------------------------------- */

/* Everything the gsl_function integrand needs, plus which scalar to return.
   component indexes the 12 scalars [field re/im (0,1), deriv (2..9),
   src re/im (10,11)]. */
typedef struct {
    struct effsource_equatorial_ctx *ctx;
    int    mMode, nMode, component;
    struct coordinate *xField;
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
       INSPECTRE_INTEG_MINO_SPLINE= 5 }; /* graded-lambda mesh at closest approach + spline */

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
   interleaved like dPhiS). */
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

/* FFT source n-mode amplitudes: one FFTW transform of S_m sampled on a uniform
   periodic t-grid of N points yields the complex amplitude A_n for every n at
   once. outRe[k]/outIm[k] (length N) hold mode n = (k <= N/2) ? k : k - N; for a
   requested n read bin ((n % N) + N) % N. Requires |n| <= N/2 to be resolved. */
void inspectre_fft_source_nmodes(struct effsource_equatorial_ctx *ctx, int mMode,
        struct coordinate *xField, korb_params *orbpar,
        double a, double p, double e, double omegaPhi, double omegaR,
        int N, double *outRe, double *outIm);

#endif /* INSPECTRE_H */
