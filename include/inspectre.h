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

enum { INSPECTRE_INTEG_QAG        = 0,
       INSPECTRE_INTEG_TIMESERIES = 1,   /* trapezoid over the samples */
       INSPECTRE_INTEG_SIMPSON    = 2,   /* composite Simpson over the samples */
       INSPECTRE_INTEG_SPLINE     = 3 }; /* gsl cubic-spline quadrature */

/* Driver. mode == INSPECTRE_INTEG_QAG: adaptive GSL QAG, evaluating (1) on
   demand. The remaining modes integrate the precomputed samples (tSamples +
   interleaved fieldSamples[2], derivSamples[8], srcSamples[2] per point):
   INSPECTRE_INTEG_TIMESERIES trapezoid, INSPECTRE_INTEG_SIMPSON composite
   Simpson, INSPECTRE_INTEG_SPLINE gsl cubic-spline quadrature. Outputs the
   complex n-mode amplitudes (re/im interleaved like dPhiS). */
void inspectre_integrate_nmode(int mode,
        struct effsource_equatorial_ctx *ctx, int mMode, int nMode,
        struct coordinate *xField, korb_params *orbpar,
        double a, double p, double e, double omegaPhi, double omegaR,
        double epsabs, double epsrel,
        double *tSamples, double *fieldSamples, double *derivSamples,
        double *srcSamples, int nSamples,
        double *nModePhiS, double *nModeDPhiS, double *nModesrc);

#endif /* INSPECTRE_H */
