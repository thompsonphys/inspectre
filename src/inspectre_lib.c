/*******************************************************************************
 * inspectre_lib.c
 *
 * Library routines for inspectre:
 *   - orbit / n-mode helpers (moved here so mains and the QAG integrand share
 *     one copy)
 *   - (1) effective source + puncture at a given field point and time
 *   - (2) GSL QAG / timeseries n-mode Fourier amplitude integration
 ******************************************************************************/

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <fftw3.h>

#include "../include/inspectre.h"

/* Mino-time graded mesh (INSPECTRE_INTEG_MINO_SPLINE): node density ~ 1/dist^BETA
   so samples cluster where the source peaks (closest approach). FLOOR caps the
   density when the particle passes very close to the field point. */
#define INSPECTRE_MINO_GRADE_BETA  1.5
#define INSPECTRE_MINO_DIST_FLOOR  1e-3   /* clip distance, as a fraction of rField */

/* ===========================================================================
 * Orbit / n-mode helpers
 * ========================================================================= */

double xFuncInsp(double a, double p, double e)
{
    double x, F, N, C, signA;
    double p2, p3, e2p3;

    signA = a > 0.0 ? 1.0 : -1.0;

    p2 = p * p;
    p3 = p2 * p;
    e2p3 = 3.0 + e * e;

    C = (a * a - p);
    C *= C;
    N = 2.0 / p * (-p2 + (e2p3 - a * a) * p - a * a * (1 + 3.0 * e * e));
    F = 1 / p3 * (p3 - 2.0 * e2p3 * p2 + e2p3 * e2p3 * p - 4.0 * a * a * (1.0 - e * e) * (1.0 - e * e));

    x = sqrt((-N - signA * sqrt(N * N - 4.0 * F * C)) / (2.0 * F));

    return x;
}

double fourVel(double psi, double a, double p, double e, double E)
{
    double X = xFuncInsp(a, p, e);
    double X2 = X * X;
    double R2 = X2 + a * a + 2.0 * X * a * E - 2.0 * X2 / p * (3 + e * cos(psi));

    return e * sin(psi) / p * sqrt(R2 > 0.0 ? R2 : 0.0);
}

double inspectre_epicyclic_frequency(const korb_params *orbpar)
{
    const double a = orbpar->a, p = orbpar->p;
    double R = 1.0 - 6.0 / p + 8.0 * a / (p * sqrt(p)) - 3.0 * a * a / (p * p);

    return orbpar->wphi * sqrt(R > 0.0 ? R : 0.0);
}

double inspectre_radial_frequency(const korb_params *orbpar)
{
    return orbpar->eccentric ? orbpar->wr : inspectre_epicyclic_frequency(orbpar);
}

double inspectre_radial_mino_period(const korb_params *orbpar)
{
    if (orbpar->eccentric)
        return orbpar->Vr;

    return 2.0 * M_PI / (orbpar->Ga * inspectre_epicyclic_frequency(orbpar));
}

int inspectre_orbit_circular_fix(korb_params *orbpar)
{
    if (orbpar->eccentric)
        return 0;

    orbpar->wr = inspectre_epicyclic_frequency(orbpar);
    orbpar->Yr = orbpar->Ga * orbpar->wr;
    orbpar->Vr = 2.0 * M_PI / orbpar->Yr;
    return 1;
}

/* Assuming exp(i Omega t) rotation */
double frequencyShiftReal(double t, double omegaPhi, double omegaR, double fieldRE, double fieldIM, int mMode, int nMode)
{
    double frequency = (double)mMode * omegaPhi + (double)nMode * omegaR;
    double angle = frequency * t;
    return fieldRE * cos(angle) - fieldIM * sin(angle);
}

double frequencyShiftImag(double t, double omegaPhi, double omegaR, double fieldRE, double fieldIM, int mMode, int nMode)
{
    double frequency = (double)mMode * omegaPhi + (double)nMode * omegaR;
    double angle = frequency * t;
    return fieldIM * cos(angle) + fieldRE * sin(angle);
}

/* just brute force this */
void generateNModeIntegrands(double t, double omegaPhi, double omegaR, int mMode, int nMode, double *Field, double *FieldDeriv, double *EffSrc, double *nModeField, double *nModeFieldDeriv, double *nModeEffSrc)
{
    nModeField[0] = frequencyShiftReal(t, omegaPhi, omegaR, Field[0], Field[1], mMode, nMode);
    nModeField[1] = frequencyShiftImag(t, omegaPhi, omegaR, Field[0], Field[1], mMode, nMode);

    nModeFieldDeriv[0] = frequencyShiftReal(t, omegaPhi, omegaR, FieldDeriv[0], FieldDeriv[1], mMode, nMode);
    nModeFieldDeriv[1] = frequencyShiftImag(t, omegaPhi, omegaR, FieldDeriv[0], FieldDeriv[1], mMode, nMode);
    nModeFieldDeriv[2] = frequencyShiftReal(t, omegaPhi, omegaR, FieldDeriv[2], FieldDeriv[3], mMode, nMode);
    nModeFieldDeriv[3] = frequencyShiftImag(t, omegaPhi, omegaR, FieldDeriv[2], FieldDeriv[3], mMode, nMode);
    nModeFieldDeriv[4] = frequencyShiftReal(t, omegaPhi, omegaR, FieldDeriv[4], FieldDeriv[5], mMode, nMode);
    nModeFieldDeriv[5] = frequencyShiftImag(t, omegaPhi, omegaR, FieldDeriv[4], FieldDeriv[5], mMode, nMode);
    nModeFieldDeriv[6] = frequencyShiftReal(t, omegaPhi, omegaR, FieldDeriv[6], FieldDeriv[7], mMode, nMode);
    nModeFieldDeriv[7] = frequencyShiftImag(t, omegaPhi, omegaR, FieldDeriv[6], FieldDeriv[7], mMode, nMode);

    nModeEffSrc[0] = frequencyShiftReal(t, omegaPhi, omegaR, EffSrc[0], EffSrc[1], mMode, nMode);
    nModeEffSrc[1] = frequencyShiftImag(t, omegaPhi, omegaR, EffSrc[0], EffSrc[1], mMode, nMode);
}

/* ===========================================================================
 * (1) Effective source + puncture at a field point and time
 * ========================================================================= */

static double insp_set_particle(struct effsource_equatorial_ctx *ctx, double lambda,
                                korb_params *orbpar, double a, double p, double e,
                                int zeroPhi)
{
    double psi   = korb_psifromla(lambda, *orbpar);
    double r_p   = korb_rfrompsi(psi, *orbpar);
    double phi_p = korb_phifromla(lambda, *orbpar);
    double ur    = fourVel(psi, a, p, e, orbpar->E);

    struct coordinate xParticle;
    xParticle.t     = 0.0;
    xParticle.r     = r_p;
    xParticle.theta = M_PI_2;
    xParticle.phi   = zeroPhi ? 0.0 : phi_p;

    effsource_equatorial_ctx_set_particle(ctx, &xParticle, orbpar->E, orbpar->Lz, ur);
    return phi_p;
}

void inspectre_eval_at_lambda(struct effsource_equatorial_ctx *ctx, int mMode,
                              struct coordinate *xField, double lambda,
                              korb_params *orbpar, double a, double p, double e,
                              double *PhiS, double *dPhiS, double *ddPhiS,
                              double *src)
{
    insp_set_particle(ctx, lambda, orbpar, a, p, e, 0);
    effsource_equatorial_ctx_calc_m(ctx, mMode, xField, PhiS, dPhiS, ddPhiS, src);
}

/* residual t(lambda) - t_target for the root solve */
struct insp_tsolve_params { double t; korb_params *orbpar; };

static double insp_t_residual(double lambda, void *params)
{
    struct insp_tsolve_params *tp = (struct insp_tsolve_params *)params;
    return korb_tfromla(lambda, *tp->orbpar) - tp->t;
}

double inspectre_lambda_from_t(double t, korb_params *orbpar)
{
    /* t(lambda) is monotonically increasing with t(0) = 0. The mean rate
       dt/dlambda = Gamma gives a good first guess; expand around it to bracket. */
    struct insp_tsolve_params tp = { t, orbpar };

    if (t <= 0.0)
        return 0.0;

    double guess = t / orbpar->Ga;
    double width = fabs(guess) * 0.1 + 1.0;

    double lo = guess - width;
    double hi = guess + width;
    if (lo < 0.0) lo = 0.0;

    /* widen until the root is bracketed */
    int it = 0;
    while (insp_t_residual(lo, &tp) > 0.0 && it < 200) { lo -= width; if (lo < 0.0) { lo = 0.0; break; } it++; }
    it = 0;
    while (insp_t_residual(hi, &tp) < 0.0 && it < 200) { hi += width; it++; }

    const gsl_root_fsolver_type *T = gsl_root_fsolver_brent;
    gsl_root_fsolver *s = gsl_root_fsolver_alloc(T);

    gsl_function F;
    F.function = &insp_t_residual;
    F.params   = &tp;
    gsl_root_fsolver_set(s, &F, lo, hi);

    double root = guess;
    int status, iter = 0;
    do {
        iter++;
        gsl_root_fsolver_iterate(s);
        root   = gsl_root_fsolver_root(s);
        lo     = gsl_root_fsolver_x_lower(s);
        hi     = gsl_root_fsolver_x_upper(s);
        status = gsl_root_test_interval(lo, hi, 1.0e-13, 1.0e-13);
    } while (status == GSL_CONTINUE && iter < 200);

    gsl_root_fsolver_free(s);
    return root;
}

void inspectre_eval_at_time(struct effsource_equatorial_ctx *ctx, int mMode,
                            struct coordinate *xField, double t,
                            korb_params *orbpar, double a, double p, double e,
                            double *PhiS, double *dPhiS, double *ddPhiS,
                            double *src)
{
    double lambda = inspectre_lambda_from_t(t, orbpar);
    inspectre_eval_at_lambda(ctx, mMode, xField, lambda, orbpar, a, p, e,
                             PhiS, dPhiS, ddPhiS, src);
}

/* ===========================================================================
 * (2) n-mode Fourier amplitude integration
 * ========================================================================= */

/* Count of QAG component integrations that hit GSL_EMAXITER since the last
   reset. Lets callers (whose integrate signature must stay stable) detect
   unreliable adaptive integrations without parsing stderr. */
static long insp_qag_limit_hits = 0;
void inspectre_qag_limit_reset(void) { insp_qag_limit_hits = 0; }
long inspectre_qag_limit_count(void) { return insp_qag_limit_hits; }

/* Pick scalar `c` out of the 12 n-mode components:
     0,1   -> field re/im
     2..9  -> derivatives
     10,11 -> source re/im                                                    */
static double insp_select_component(int c, double *nField, double *nDeriv, double *nSrc)
{
    if (c < 2)  return nField[c];
    if (c < 10) return nDeriv[c - 2];
    return nSrc[c - 10];
}

double inspectre_nmode_integrand(double t, void *params)
{
    inspectre_nmode_params *ip = (inspectre_nmode_params *)params;

    double PhiS[2], dPhiS[8], ddPhiS[20], src[2];
    inspectre_eval_at_time(ip->ctx, ip->mMode, ip->xField, t, ip->orbpar,
                           ip->a, ip->p, ip->e, PhiS, dPhiS, ddPhiS, src);

    double nField[2], nDeriv[8], nSrc[2];
    generateNModeIntegrands(t, ip->omegaPhi, ip->omegaR, ip->mMode, ip->nMode,
                            PhiS, dPhiS, src, nField, nDeriv, nSrc);

    return insp_select_component(ip->component, nField, nDeriv, nSrc);
}

/* Instantaneous time Jacobian dt/dlambda at Mino time lambda. The coordinate
   time advances as dt = (T_r + T_theta) dlambda; for equatorial orbits the
   T_theta piece is constant. Integrates to korb_tfromla over [0, Vr]. */
static double insp_dtdlambda(double lambda, korb_params *o)
{
    return korb_Tr(korb_psifromla(lambda, *o), *o)
         + korb_Tth(korb_chifromla(lambda, *o), *o);
}

/* Euclidean field-to-particle distance (particle equatorial, same phi) used to
   grade the Mino-time mesh. NB: the graded mesh equidistributes nodes by the
   cumulative density 1/dist^beta, which is invariant under a global rescale of
   dist, so no rField normalization is needed here (unlike the absolute stepping
   in inspectre_adaptive.c). The only scale that matters is the floor, applied
   relative to rField by the caller. */
static double insp_distance(double rField, double theta, double rParticle)
{
    double s = rParticle * rParticle + rField * rField
             - 2.0 * rParticle * rField * sin(theta);
    return sqrt(s > 0.0 ? s : 0.0);
}

/* gsl_function: as inspectre_nmode_integrand but parameterized by Mino time
   lambda. Evaluates at lambda directly (no Brent inversion), frequency-shifts
   with t(lambda), and weights by the Jacobian dt/dlambda so that integrating
   over [0, Vr] reproduces the coordinate-time integral over [0, Tr]. */
double inspectre_nmode_integrand_lambda(double lambda, void *params)
{
    inspectre_nmode_params *ip = (inspectre_nmode_params *)params;

    double PhiS[2], dPhiS[8], ddPhiS[20], src[2];
    inspectre_eval_at_lambda(ip->ctx, ip->mMode, ip->xField, lambda, ip->orbpar,
                             ip->a, ip->p, ip->e, PhiS, dPhiS, ddPhiS, src);

    double t = korb_tfromla(lambda, *ip->orbpar);
    double J = insp_dtdlambda(lambda, ip->orbpar);

    double nField[2], nDeriv[8], nSrc[2];
    generateNModeIntegrands(t, ip->omegaPhi, ip->omegaR, ip->mMode, ip->nMode,
                            PhiS, dPhiS, src, nField, nDeriv, nSrc);

    return insp_select_component(ip->component, nField, nDeriv, nSrc) * J;
}

/* store the 12 scalar results into the interleaved output arrays */
static void insp_store_results(const double *res,
                               double *nModePhiS, double *nModeDPhiS, double *nModesrc)
{
    nModePhiS[0]  = res[0];
    nModePhiS[1]  = res[1];
    for (int k = 0; k < 8; k++)
        nModeDPhiS[k] = res[2 + k];
    nModesrc[0]   = res[10];
    nModesrc[1]   = res[11];
}

/* gsl cubic-spline quadrature of each of the 12 components against abscissa
   x[0..n-1] (uniform or graded), normalized by Tr. comp[c*n + i] is component c
   at node i. Shared by the SPLINE (coordinate time) and MINO_SPLINE (Mino time)
   paths. */
static void insp_spline_quad(const double *x, const double *comp, int n,
                             double Tr, double *res)
{
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *sp = gsl_spline_alloc(gsl_interp_cspline, n);
    double x0 = x[0], xN = x[n - 1];
    for (int c = 0; c < 12; c++)
    {
        gsl_spline_init(sp, x, &comp[c * n], n);
        res[c] = gsl_spline_eval_integ(sp, x0, xN, acc) / Tr;
        gsl_interp_accel_reset(acc);
    }
    gsl_spline_free(sp);
    gsl_interp_accel_free(acc);
}

/* Composite trapezoid quadrature of each of the 12 components against abscissa
   x[0..n-1], normalized by Tr. Handles non-uniform spacing (uses each interval's
   own dx), so it works on the uniform-t driver grid and the graded Mino mesh
   alike. comp[c*n + i] is component c at node i. */
static void insp_quad_trap(const double *x, const double *comp, int n,
                           double Tr, double *res)
{
    for (int c = 0; c < 12; c++) res[c] = 0.0;
    for (int i = 1; i < n; i++)
    {
        double dt = x[i] - x[i - 1];
        for (int c = 0; c < 12; c++)
            res[c] += 0.5 * (comp[c * n + i - 1] + comp[c * n + i]) * dt;
    }
    for (int c = 0; c < 12; c++) res[c] /= Tr;
}

/* Composite Simpson quadrature in the general unequal-spacing form (reduces to
   the 1/3 rule on a uniform grid), normalized by Tr. If the interval count
   (n-1) is odd, the trailing single interval is closed with the trapezoid rule.
   Handles non-uniform spacing, so it also works on the graded Mino mesh. */
static void insp_quad_simpson(const double *x, const double *comp, int n,
                              double Tr, double *res)
{
    for (int c = 0; c < 12; c++) res[c] = 0.0;
    int i;
    for (i = 0; i + 2 < n; i += 2)
    {
        double h0 = x[i + 1] - x[i];
        double h1 = x[i + 2] - x[i + 1];
        double w  = (h0 + h1) / 6.0;
        for (int c = 0; c < 12; c++)
        {
            const double *f = &comp[c * n + i];
            res[c] += w * ((2.0 - h1 / h0) * f[0]
                         + (h0 + h1) * (h0 + h1) / (h0 * h1) * f[1]
                         + (2.0 - h0 / h1) * f[2]);
        }
    }
    if (i + 1 < n)   /* odd interval count: trapezoid on the last */
    {
        double dt = x[i + 1] - x[i];
        for (int c = 0; c < 12; c++)
            res[c] += 0.5 * (comp[c * n + i] + comp[c * n + i + 1]) * dt;
    }
    for (int c = 0; c < 12; c++) res[c] /= Tr;
}

/* ---------------------------------------------------------------------------
 * Reusable Mino-time graded-mesh samples (build once, integrate per n).
 * ------------------------------------------------------------------------- */

void inspectre_mino_samples_build(struct effsource_equatorial_ctx *ctx, int mMode,
        struct coordinate *xField, korb_params *orbpar,
        double a, double p, double e, int nSamples,
        inspectre_mino_samples *out)
{
    const int N = nSamples;
    const double Vr = inspectre_radial_mino_period(orbpar);

    out->n  = N;
    out->Vr = Vr;
    out->Tr = korb_tfromla(Vr, *orbpar);

    /* fine orbit-only pre-grid: build the cumulative node density (no source
       evals here, just the trajectory) so we can equidistribute N nodes. */
    int M = 8 * N; if (M < 4096) M = 4096;
    double *preLam = malloc((size_t)M * sizeof(double));
    double *cum    = malloc((size_t)M * sizeof(double));
    double prevRho = 0.0;
    for (int i = 0; i < M; i++)
    {
        double lam  = (double)i / (double)(M - 1) * Vr;
        double r_p  = korb_rfrompsi(korb_psifromla(lam, *orbpar), *orbpar);
        double dist = insp_distance(xField->r, xField->theta, r_p);
        double floor = INSPECTRE_MINO_DIST_FLOOR * xField->r;  /* scale-relative clip */
        if (dist < floor) dist = floor;
        double rho  = 1.0 / pow(dist, INSPECTRE_MINO_GRADE_BETA);
        preLam[i] = lam;
        cum[i] = (i == 0) ? 0.0
               : cum[i-1] + 0.5 * (prevRho + rho) * (lam - preLam[i-1]);
        prevRho = rho;
    }

    /* invert lambda(cum) to place N nodes at equal cumulative-density steps */
    gsl_interp_accel *iacc = gsl_interp_accel_alloc();
    gsl_interp *inv = gsl_interp_alloc(gsl_interp_linear, M);
    gsl_interp_init(inv, cum, preLam, M);
    double cumTotal = cum[M - 1];

    out->lam = malloc((size_t)N * sizeof(double));
    out->t   = malloc((size_t)N * sizeof(double));
    out->J   = malloc((size_t)N * sizeof(double));
    out->raw = malloc((size_t)12 * N * sizeof(double));
    for (int k = 0; k < N; k++)
    {
        double lam = (k == 0)     ? 0.0
                   : (k == N - 1) ? Vr
                   : gsl_interp_eval(inv, cum, preLam,
                                     (double)k / (double)(N - 1) * cumTotal, iacc);
        out->lam[k] = lam;

        double PhiS[2], dPhiS[8], ddPhiS[20], src[2];
        inspectre_eval_at_lambda(ctx, mMode, xField, lam, orbpar, a, p, e,
                                 PhiS, dPhiS, ddPhiS, src);
        out->t[k] = korb_tfromla(lam, *orbpar);
        out->J[k] = insp_dtdlambda(lam, orbpar);
        /* raw, unshifted components, contiguous per component */
        out->raw[0 * N + k] = PhiS[0];
        out->raw[1 * N + k] = PhiS[1];
        for (int c = 0; c < 8; c++)
            out->raw[(2 + c) * N + k] = dPhiS[c];
        out->raw[10 * N + k] = src[0];
        out->raw[11 * N + k] = src[1];
    }

    gsl_interp_free(inv);
    gsl_interp_accel_free(iacc);
    free(preLam); free(cum);
}

void inspectre_mino_samples_free(inspectre_mino_samples *s)
{
    free(s->lam); free(s->t); free(s->J); free(s->raw);
    s->lam = s->t = s->J = s->raw = NULL;
    s->n = 0;
}

void inspectre_mino_samples_integrate(const inspectre_mino_samples *s,
        int mMode, int nMode, double omegaPhi, double omegaR,
        double *nModePhiS, double *nModeDPhiS, double *nModesrc)
{
    const int N = s->n;
    double res[12];
    double *comp = malloc((size_t)12 * N * sizeof(double));
    for (int k = 0; k < N; k++)
    {
        double Field[2] = { s->raw[0 * N + k], s->raw[1 * N + k] };
        double Deriv[8];
        for (int c = 0; c < 8; c++) Deriv[c] = s->raw[(2 + c) * N + k];
        double Src[2] = { s->raw[10 * N + k], s->raw[11 * N + k] };

        double nField[2], nDeriv[8], nSrc[2];
        generateNModeIntegrands(s->t[k], omegaPhi, omegaR, mMode, nMode,
                                Field, Deriv, Src, nField, nDeriv, nSrc);
        for (int c = 0; c < 12; c++)
            comp[c * N + k] = insp_select_component(c, nField, nDeriv, nSrc) * s->J[k];
    }
    insp_spline_quad(s->lam, comp, N, s->Tr, res);
    free(comp);
    insp_store_results(res, nModePhiS, nModeDPhiS, nModesrc);
}

void inspectre_mino_samples_integrate_rule(const inspectre_mino_samples *s,
        int mMode, int nMode, double omegaPhi, double omegaR, int rule,
        double *nModePhiS, double *nModeDPhiS, double *nModesrc)
{
    const int N = s->n;
    double res[12];
    double *comp = malloc((size_t)12 * N * sizeof(double));
    for (int k = 0; k < N; k++)
    {
        double Field[2] = { s->raw[0 * N + k], s->raw[1 * N + k] };
        double Deriv[8];
        for (int c = 0; c < 8; c++) Deriv[c] = s->raw[(2 + c) * N + k];
        double Src[2] = { s->raw[10 * N + k], s->raw[11 * N + k] };

        double nField[2], nDeriv[8], nSrc[2];
        generateNModeIntegrands(s->t[k], omegaPhi, omegaR, mMode, nMode,
                                Field, Deriv, Src, nField, nDeriv, nSrc);
        /* integrate g(t) directly over the non-uniform node times t[]: NO
           Jacobian here (the dt/dlambda weight is only for the lambda-
           parametrized spline path). This tests the graded mesh under a plain
           non-uniform-sample rule instead of the cubic-spline quadrature. */
        for (int c = 0; c < 12; c++)
            comp[c * N + k] = insp_select_component(c, nField, nDeriv, nSrc);
    }
    if      (rule == INSPECTRE_INTEG_SIMPSON) insp_quad_simpson(s->t, comp, N, s->Tr, res);
    else if (rule == INSPECTRE_INTEG_SPLINE)  insp_spline_quad(s->t, comp, N, s->Tr, res);
    else                                      insp_quad_trap(s->t, comp, N, s->Tr, res);
    free(comp);
    insp_store_results(res, nModePhiS, nModeDPhiS, nModesrc);
}

void inspectre_integrate_nmode(int mode,
        struct effsource_equatorial_ctx *ctx, int mMode, int nMode,
        struct coordinate *xField, korb_params *orbpar,
        double a, double p, double e, double omegaPhi, double omegaR,
        double epsabs, double epsrel,
        double *tSamples, double *fieldSamples, double *derivSamples,
        double *srcSamples, int nSamples,
        double *nModePhiS, double *nModeDPhiS, double *nModesrc)
{
    /* coordinate-time radial period; n-mode amplitude is the period-average */
    double Tr = korb_tfromla(inspectre_radial_mino_period(orbpar), *orbpar);

    double res[12];

    if (mode == INSPECTRE_INTEG_QAG)
    {
        /* GSL requires the workspace size to be >= the qag `limit` argument;
           if limit exceeds it, qag returns GSL_EINVAL and sets result=0. Tie
           both to one constant so they can never diverge. */
        const size_t QAG_LIMIT = 20000;
        gsl_integration_workspace *w = gsl_integration_workspace_alloc(QAG_LIMIT);

        inspectre_nmode_params ip;
        ip.ctx = ctx; ip.mMode = mMode; ip.nMode = nMode; ip.component = 0;
        ip.xField = xField; ip.orbpar = orbpar;
        ip.a = a; ip.p = p; ip.e = e;
        ip.omegaPhi = omegaPhi; ip.omegaR = omegaR;

        gsl_function F;
        F.function = &inspectre_nmode_integrand;
        F.params   = &ip;

        for (int c = 0; c < 12; c++)
        {
            double result, abserr;
            ip.component = c;
            int st = gsl_integration_qag(&F, 0.0, Tr, epsabs, epsrel, QAG_LIMIT,
                                GSL_INTEG_GAUSS61, w, &result, &abserr);
            /* error handler is off, so a nonzero status would otherwise be
               swallowed (result left at 0). GSL_EROUND just means the tolerance
               is tighter than the roundoff floor; keep the best estimate. */
            if (st == GSL_EMAXITER) {
                insp_qag_limit_hits++;
                fprintf(stderr, "[qag] n=%d c=%d hit iteration limit (%zu) -- "
                        "result may be inaccurate\n", nMode, c, QAG_LIMIT);
            }
            else if (st && st != GSL_EROUND)
                fprintf(stderr, "[qag] n=%d c=%d status=%d (%s)\n",
                        nMode, c, st, gsl_strerror(st));
            res[c] = result / Tr;
        }

        gsl_integration_workspace_free(w);
    }
    else if (mode == INSPECTRE_INTEG_QAG_MINO)
    {
        /* Same adaptive QAG, but in Mino time: integrate g(lambda)*dt/dlambda
           over [0, Vr]. Each integrand eval hits inspectre_eval_at_lambda
           directly (no Brent inversion of t(lambda)). */
        const size_t QAG_LIMIT = 20000;
        gsl_integration_workspace *w = gsl_integration_workspace_alloc(QAG_LIMIT);

        inspectre_nmode_params ip;
        ip.ctx = ctx; ip.mMode = mMode; ip.nMode = nMode; ip.component = 0;
        ip.xField = xField; ip.orbpar = orbpar;
        ip.a = a; ip.p = p; ip.e = e;
        ip.omegaPhi = omegaPhi; ip.omegaR = omegaR;

        gsl_function F;
        F.function = &inspectre_nmode_integrand_lambda;
        F.params   = &ip;

        for (int c = 0; c < 12; c++)
        {
            double result, abserr;
            ip.component = c;
            int st = gsl_integration_qag(&F, 0.0, inspectre_radial_mino_period(orbpar), epsabs, epsrel, QAG_LIMIT,
                                GSL_INTEG_GAUSS61, w, &result, &abserr);
            if (st == GSL_EMAXITER) {
                insp_qag_limit_hits++;
                fprintf(stderr, "[qag_mino] n=%d c=%d hit iteration limit (%zu) -- "
                        "result may be inaccurate\n", nMode, c, QAG_LIMIT);
            }
            else if (st && st != GSL_EROUND)
                fprintf(stderr, "[qag_mino] n=%d c=%d status=%d (%s)\n",
                        nMode, c, st, gsl_strerror(st));
            res[c] = result / Tr;
        }

        gsl_integration_workspace_free(w);
    }
    else if (mode == INSPECTRE_INTEG_MINO_SPLINE)
    {
        /* Self-contained: place `nSamples` nodes in Mino time, graded toward
           closest approach (density ~ 1/dist^BETA), then spline-integrate the
           Jacobian-weighted integrand over lambda. Ignores tSamples/field/etc.
           Build the n-independent samples then integrate this single n; callers
           sweeping many n should build once and call _integrate per n. */
        inspectre_mino_samples s;
        inspectre_mino_samples_build(ctx, mMode, xField, orbpar, a, p, e,
                                     nSamples, &s);
        inspectre_mino_samples_integrate(&s, mMode, nMode, omegaPhi, omegaR,
                                         nModePhiS, nModeDPhiS, nModesrc);
        inspectre_mino_samples_free(&s);
        return;
    }
    else /* sample-based methods integrate the precomputed samples */
    {
        /* These modes read the caller-supplied per-sample arrays directly; with
           NULL arrays (or too few samples to form an interval) the loop below
           would dereference NULL and crash the process. Fail loudly with NaN
           outputs instead so a misuse is visible rather than fatal. */
        if (!tSamples || !fieldSamples || !derivSamples || !srcSamples
            || nSamples < 2)
        {
            fprintf(stderr, "[integrate_nmode] mode=%d n=%d requires tSamples + "
                    "fieldSamples/derivSamples/srcSamples and nSamples>=2; got "
                    "%s samples (nSamples=%d) -- returning NaN\n",
                    mode, nMode,
                    (tSamples && fieldSamples && derivSamples && srcSamples)
                        ? "too few" : "NULL",
                    nSamples);
            for (int c = 0; c < 12; c++) res[c] = NAN;
            insp_store_results(res, nModePhiS, nModeDPhiS, nModesrc);
            return;
        }

        /* per-sample n-mode component values, contiguous per component so the
           spline path can hand each component straight to gsl_spline_init:
           comp[c * nSamples + i] is component c at sample i. */
        double *comp = malloc((size_t)12 * nSamples * sizeof(double));
        for (int i = 0; i < nSamples; i++)
        {
            double nField[2], nDeriv[8], nSrc[2];
            generateNModeIntegrands(tSamples[i], omegaPhi, omegaR, mMode, nMode,
                                    &fieldSamples[2 * i], &derivSamples[8 * i],
                                    &srcSamples[2 * i], nField, nDeriv, nSrc);
            for (int c = 0; c < 12; c++)
                comp[c * nSamples + i] = insp_select_component(c, nField, nDeriv, nSrc);
        }

        if (mode == INSPECTRE_INTEG_SPLINE)
            insp_spline_quad(tSamples, comp, nSamples, Tr, res);
        else if (mode == INSPECTRE_INTEG_SIMPSON)
            insp_quad_simpson(tSamples, comp, nSamples, Tr, res);
        else /* INSPECTRE_INTEG_TIMESERIES: trapezoid over the samples */
            insp_quad_trap(tSamples, comp, nSamples, Tr, res);

        free(comp);
    }

    insp_store_results(res, nModePhiS, nModeDPhiS, nModesrc);
}

/* ===========================================================================
 * (3) FFT n-mode source amplitudes — all n in one transform
 *
 * The source n-mode amplitude is the Fourier coefficient
 *     A_n = (1/Tr) integral_0^Tr exp(i(mMode*wphi + n*wr) t) S_m(t) dt.
 * Because the m-carrier cancels the secular phase of S_m, the product
 *     g(t) = exp(i mMode*wphi t) S_m(t)
 * is exactly Tr-periodic, and wr = 2*pi/Tr. Sampling g on the uniform periodic
 * grid t_j = j*Tr/N (j=0..N-1) makes
 *     A_n = (1/N) sum_j g(t_j) exp(+2*pi*i*n*j/N),
 * i.e. one FFTW backward transform yields every n at once. Bin k holds mode
 * n = (k <= N/2) ? k : k - N; for a requested n use bin ((n % N) + N) % N.
 * ========================================================================= */
void inspectre_fft_source_nmodes(struct effsource_equatorial_ctx *ctx, int mMode,
        struct coordinate *xField, korb_params *orbpar,
        double a, double p, double e, double omegaPhi, double omegaR,
        int N, double *outRe, double *outIm)
{
    double Vr = inspectre_radial_mino_period(orbpar);
    double Tr = korb_tfromla(Vr, *orbpar);

    /* lambda(t) spline so the source can be sampled on a uniform-t grid */
    double *latT = malloc((size_t)N * sizeof(double));
    double *latL = malloc((size_t)N * sizeof(double));
    for (int i = 0; i < N; i++) {
        double lam = (double)i / (double)(N - 1) * Vr;
        latL[i] = lam;
        latT[i] = korb_tfromla(lam, *orbpar);
    }
    gsl_interp_accel *acc = gsl_interp_accel_alloc();
    gsl_spline *sp = gsl_spline_alloc(gsl_interp_cspline, N);
    gsl_spline_init(sp, latT, latL, N);

    fftw_complex *in  = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (size_t)N);
    fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (size_t)N);

    for (int j = 0; j < N; j++) {
        double t   = (double)j / (double)N * Tr;          /* [0, Tr), periodic */
        double lam = gsl_spline_eval(sp, t, acc);
        double PhiS[2], dPhiS[8], ddPhiS[20], src[2];
        inspectre_eval_at_lambda(ctx, mMode, xField, lam, orbpar, a, p, e,
                                 PhiS, dPhiS, ddPhiS, src);
        /* apply only the m-carrier; the FFT supplies exp(i n wr t) for every n */
        in[j][0] = frequencyShiftReal(t, omegaPhi, omegaR, src[0], src[1], mMode, 0);
        in[j][1] = frequencyShiftImag(t, omegaPhi, omegaR, src[0], src[1], mMode, 0);
    }

    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    for (int k = 0; k < N; k++) {
        outRe[k] = out[k][0] / (double)N;
        outIm[k] = out[k][1] / (double)N;
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    gsl_spline_free(sp);
    gsl_interp_accel_free(acc);
    free(latT);
    free(latL);
}

static int insp_is_pow2_at_least_64(int n)
{
    return n >= 64 && (n & (n - 1)) == 0;
}

static int insp_check_mlist(int route, int nm, const int *mlist, int nphiMax)
{
    if (nm < 1)
        return 1;
    for (int i = 0; i < nm; i++) {
        if (mlist[i] < 0)
            return 1;
        if (route == INSPECTRE_MROUTE_ANALYTIC && mlist[i] > 20)
            return 1;
        if (route == INSPECTRE_MROUTE_PROJECTED && mlist[i] > nphiMax / 2)
            return 1;
    }
    return 0;
}

static double insp_c1(const struct effsource_equatorial_ctx *ctx, const struct coordinate *xField)
{
    double dr  = xField->r - ctx->xp.r;
    double dth = xField->theta - ctx->xp.theta;
    return (ctx->alpha20 * dr * dr + ctx->alpha02 * dth * dth) / ctx->beta;
}

static int insp_nphi(double C1, int mmax, int nphiMax, int *capped)
{
    double want = (double)mmax + 16.0 / sqrt(C1);
    if (want < 64.0)
        want = 64.0;
    if (want < 2.5 * mmax)
        want = 2.5 * mmax;
    if (!(want <= (double)nphiMax)) {
        *capped = 1;
        return nphiMax;
    }
    *capped = 0;
    int n = 64;
    while (n < want)
        n *= 2;
    return n;
}

typedef struct {
    double       *in;
    fftw_complex *out;
    fftw_plan     plan;
} insp_rfft;

static insp_rfft insp_rfft_cache[31];

static insp_rfft *insp_rfft_get(int n)
{
    int k = 0;
    while ((1 << k) < n)
        k++;
    insp_rfft *c = &insp_rfft_cache[k];
    if (!c->plan) {
        int h = n / 2 + 1;
        c->in   = (double *)fftw_malloc(sizeof(double) * 4 * (size_t)n);
        c->out  = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * 4 * (size_t)h);
        c->plan = fftw_plan_many_dft_r2c(1, &n, 4, c->in, NULL, 1, n,
                                         c->out, NULL, 1, h, FFTW_ESTIMATE);
    }
    return c;
}

int inspectre_mmode_projected(struct effsource_equatorial_ctx *ctx,
        int nm, const int *mlist, struct coordinate *xField,
        double lambda, korb_params *orbpar, double a, double p, double e,
        int nphiMax, double *out, int *nphi, int *capped)
{
    if (!insp_is_pow2_at_least_64(nphiMax)
        || insp_check_mlist(INSPECTRE_MROUTE_PROJECTED, nm, mlist, nphiMax))
        return 1;

    double phi_p = insp_set_particle(ctx, lambda, orbpar, a, p, e, 1);
    double C1 = insp_c1(ctx, xField);
    *nphi = 0;
    *capped = 0;
    if (!(C1 > 0.0)) {
        for (int i = 0; i < 8 * nm; i++)
            out[i] = NAN;
        return 0;
    }

    int mmax = 0;
    for (int i = 0; i < nm; i++)
        if (mlist[i] > mmax)
            mmax = mlist[i];
    int n = insp_nphi(C1, mmax, nphiMax, capped);
    *nphi = n;

    insp_rfft *c = insp_rfft_get(n);
    struct coordinate x = *xField;
    for (int k = 0; k < n; k++) {
        double PhiS, dPhiS[4], d2PhiS[10], src;
        x.phi = 2.0 * M_PI * (double)k / (double)n;
        effsource_equatorial_ctx_calc(ctx, &x, &PhiS, dPhiS, d2PhiS, &src);
        c->in[k]         = PhiS;
        c->in[n + k]     = dPhiS[1];
        c->in[2 * n + k] = dPhiS[2];
        c->in[3 * n + k] = src;
    }
    fftw_execute(c->plan);

    int h = n / 2 + 1;
    double w = 2.0 * M_PI / (double)n;
    for (int i = 0; i < nm; i++) {
        int m = mlist[i];
        double cm = cos(m * phi_p), sm = sin(m * phi_p);
        for (int q = 0; q < 4; q++) {
            double fr = c->out[q * h + m][0], fi = c->out[q * h + m][1];
            out[(i * 4 + q) * 2]     = w * (fr * cm + fi * sm);
            out[(i * 4 + q) * 2 + 1] = w * (fi * cm - fr * sm);
        }
    }
    return 0;
}

static double insp_lambda_newton(double t, double lambda, korb_params *orbpar, int *stalled)
{
    for (int it = 0; it < 20; it++) {
        double dl = (korb_tfromla(lambda, *orbpar) - t) / insp_dtdlambda(lambda, orbpar);
        lambda -= dl;
        if (fabs(dl) <= 4.0 * DBL_EPSILON * fabs(lambda))
            return lambda;
    }
    *stalled = 1;
    return lambda;
}

int inspectre_fft_nmodes(struct effsource_equatorial_ctx *ctx, int route,
        int nm, const int *mlist, struct coordinate *xField,
        korb_params *orbpar, double a, double p, double e, double omegaPhi,
        int N, int nphiMax, double *out, double *info)
{
    if ((route != INSPECTRE_MROUTE_ANALYTIC && route != INSPECTRE_MROUTE_PROJECTED)
        || N < 41 || !insp_is_pow2_at_least_64(nphiMax)
        || insp_check_mlist(route, nm, mlist, nphiMax))
        return 1;

    double Vr = inspectre_radial_mino_period(orbpar);
    double Tr = korb_tfromla(Vr, *orbpar);
    int nq = 4 * nm;

    fftw_complex *buf = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (size_t)nq * (size_t)N);
    fftw_plan plan = fftw_plan_many_dft(1, &N, nq, buf, NULL, 1, N,
                                        buf, NULL, 1, N, FFTW_BACKWARD, FFTW_ESTIMATE);
    double *slice = (double *)malloc(sizeof(double) * 8 * (size_t)nm);

    for (int i = 0; i < 5; i++)
        info[i] = 0.0;

    double lambda = 0.0;
    for (int j = 0; j < N; j++) {
        double t = (double)j / (double)N * Tr;
        if (j > 0) {
            int stalled = 0;
            lambda = insp_lambda_newton(t, lambda + (Tr / N) / insp_dtdlambda(lambda, orbpar),
                                        orbpar, &stalled);
            info[4] += stalled;
        }
        double res = fabs(korb_tfromla(lambda, *orbpar) - t);
        if (res > info[3])
            info[3] = res;

        if (route == INSPECTRE_MROUTE_ANALYTIC) {
            insp_set_particle(ctx, lambda, orbpar, a, p, e, 0);
            if (!(insp_c1(ctx, xField) > 0.0))
                info[2] += 1.0;
            for (int i = 0; i < nm; i++) {
                double PhiS[2], dPhiS[8], ddPhiS[20], src[2];
                effsource_equatorial_ctx_calc_m(ctx, mlist[i], xField, PhiS, dPhiS, ddPhiS, src);
                double *s = slice + 8 * i;
                s[0] = PhiS[0];   s[1] = PhiS[1];
                s[2] = dPhiS[2];  s[3] = dPhiS[3];
                s[4] = dPhiS[4];  s[5] = dPhiS[5];
                s[6] = src[0];    s[7] = src[1];
            }
        } else {
            int nphi, capped;
            inspectre_mmode_projected(ctx, nm, mlist, xField, lambda, orbpar, a, p, e,
                                      nphiMax, slice, &nphi, &capped);
            if (nphi == 0)
                info[2] += 1.0;
            if (nphi > info[0])
                info[0] = nphi;
            info[1] += capped;
        }

        for (int i = 0; i < nm; i++)
            for (int q = 0; q < 4; q++) {
                double re = slice[(i * 4 + q) * 2], im = slice[(i * 4 + q) * 2 + 1];
                size_t idx = (size_t)(i * 4 + q) * (size_t)N + (size_t)j;
                buf[idx][0] = frequencyShiftReal(t, omegaPhi, 0.0, re, im, mlist[i], 0);
                buf[idx][1] = frequencyShiftImag(t, omegaPhi, 0.0, re, im, mlist[i], 0);
            }
    }

    fftw_execute(plan);
    for (size_t idx = 0; idx < (size_t)nq * (size_t)N; idx++) {
        out[2 * idx]     = buf[idx][0] / (double)N;
        out[2 * idx + 1] = buf[idx][1] / (double)N;
    }

    fftw_destroy_plan(plan);
    fftw_free(buf);
    free(slice);
    return 0;
}
