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

/* Count of puncture/source evaluations since the last reset, across every path.
   Separates evaluation count from wall time; QAG also pays a Brent inversion
   per evaluation, which timing alone cannot distinguish from source cost. */
static long insp_eval_hits = 0;
void inspectre_eval_count_reset(void) { insp_eval_hits = 0; }
long inspectre_eval_count(void) { return insp_eval_hits; }

/* Evaluator swap read by inspectre_eval_at_lambda_fp, and so by every quadrature
   that samples through it. Held as state rather than passed as an argument so the
   integrator signatures stay put and the node geometry is provably identical
   across the two arms of a precision comparison. */
static int insp_eval_gold = 0;
void inspectre_eval_precision_set(int gold) { insp_eval_gold = gold ? 1 : 0; }
int  inspectre_eval_precision_get(void) { return insp_eval_gold; }

/* Seat the effsource context on the particle at this Mino time and return the
   field-point offsets. r_p(psi) = p/(1 + e cos psi) is re-evaluated in extended
   precision so that dr = r_field - r_p keeps its relative accuracy when the
   particle passes close to the field point; forming it from the double-rounded
   absolute radii would leave dr with an absolute error ~ulp(r_p), which the
   near-zone puncture and source amplify catastrophically. Accuracy is then
   limited only by psi itself. */
static void insp_seat_particle(struct effsource_equatorial_ctx *ctx,
                               const inspectre_field_point *fp, double lambda,
                               korb_params *orbpar, double a, double p, double e,
                               double *drOut, double *dthetaOut)
{
    double psi   = korb_psifromla(lambda, *orbpar);
    double r_p   = korb_rfrompsi(psi, *orbpar);
    double phi_p = korb_phifromla(lambda, *orbpar);
    double ur    = fourVel(psi, a, p, e, orbpar->E);

    struct coordinate xParticle;
    xParticle.t     = 0.0;
    xParticle.r     = r_p;
    xParticle.theta = M_PI_2; /* enforce equatorial orbits for now */
    xParticle.phi   = phi_p;

    effsource_equatorial_ctx_set_particle(ctx, &xParticle, orbpar->E, orbpar->Lz, ur);

    const long double r_p_l = (long double)p
        / (1.0L + (long double)e * cosl((long double)psi));
    *drOut     = (double)((long double)fp->r - r_p_l);
    *dthetaOut = fp->dtheta;
}

void inspectre_eval_at_lambda_fp(struct effsource_equatorial_ctx *ctx, int mMode,
                              const inspectre_field_point *fp, double lambda,
                              korb_params *orbpar, double a, double p, double e,
                              double *PhiS, double *dPhiS, double *ddPhiS,
                              double *src)
{
    double dr, dtheta;
    insp_seat_particle(ctx, fp, lambda, orbpar, a, p, e, &dr, &dtheta);
    insp_eval_hits++;

    if (insp_eval_gold)
    {
        effsource_equatorial_ctx_calc_m_gold(ctx, mMode, dr, dtheta,
                                             PhiS, dPhiS, src);
        if (ddPhiS)
            for (int i = 0; i < 20; i++)
                ddPhiS[i] = 0.0;
    }
    else
        effsource_equatorial_ctx_calc_m_offset(ctx, mMode, dr, dtheta,
                                               PhiS, dPhiS, ddPhiS, src);
}

void inspectre_eval_at_lambda(struct effsource_equatorial_ctx *ctx, int mMode,
                              struct coordinate *xField, double lambda,
                              korb_params *orbpar, double a, double p, double e,
                              double *PhiS, double *dPhiS, double *ddPhiS,
                              double *src)
{
    const inspectre_field_point fp = { xField->r, xField->theta - M_PI_2,
                                       xField->phi };
    inspectre_eval_at_lambda_fp(ctx, mMode, &fp, lambda, orbpar, a, p, e,
                                PhiS, dPhiS, ddPhiS, src);
}

void inspectre_eval_gold_at_lambda_fp(struct effsource_equatorial_ctx *ctx,
                              int mMode,
                              const inspectre_field_point *fp, double lambda,
                              korb_params *orbpar, double a, double p, double e,
                              double *PhiS, double *dPhiS, double *src)
{
    double dr, dtheta;
    insp_seat_particle(ctx, fp, lambda, orbpar, a, p, e, &dr, &dtheta);
    insp_eval_hits++;

    effsource_equatorial_ctx_calc_m_gold(ctx, mMode, dr, dtheta,
                                         PhiS, dPhiS, src);
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

void inspectre_eval_at_time_fp(struct effsource_equatorial_ctx *ctx, int mMode,
                            const inspectre_field_point *fp, double t,
                            korb_params *orbpar, double a, double p, double e,
                            double *PhiS, double *dPhiS, double *ddPhiS,
                            double *src)
{
    double lambda = inspectre_lambda_from_t(t, orbpar);
    inspectre_eval_at_lambda_fp(ctx, mMode, fp, lambda, orbpar, a, p, e,
                                PhiS, dPhiS, ddPhiS, src);
}

void inspectre_eval_at_time(struct effsource_equatorial_ctx *ctx, int mMode,
                            struct coordinate *xField, double t,
                            korb_params *orbpar, double a, double p, double e,
                            double *PhiS, double *dPhiS, double *ddPhiS,
                            double *src)
{
    const inspectre_field_point fp = { xField->r, xField->theta - M_PI_2,
                                       xField->phi };
    inspectre_eval_at_time_fp(ctx, mMode, &fp, t, orbpar, a, p, e,
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

/* All twelve frequency-shifted components at coordinate time t. One source
   evaluation serves all of them; the gsl_function below picks one, the epsabs
   floor needs the whole set. */
static void insp_nmode_components_time(double t, inspectre_nmode_params *ip,
                                       double *out)
{
    double PhiS[2], dPhiS[8], ddPhiS[20], src[2];
    inspectre_eval_at_time_fp(ip->ctx, ip->mMode, ip->fp, t, ip->orbpar,
                              ip->a, ip->p, ip->e, PhiS, dPhiS, ddPhiS, src);

    double nField[2], nDeriv[8], nSrc[2];
    generateNModeIntegrands(t, ip->omegaPhi, ip->omegaR, ip->mMode, ip->nMode,
                            PhiS, dPhiS, src, nField, nDeriv, nSrc);

    for (int c = 0; c < 12; c++)
        out[c] = insp_select_component(c, nField, nDeriv, nSrc);
}

double inspectre_nmode_integrand(double t, void *params)
{
    inspectre_nmode_params *ip = (inspectre_nmode_params *)params;

    double f[12];
    insp_nmode_components_time(t, ip, f);
    return f[ip->component];
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
static double insp_distance(double rField, double dtheta, double rParticle)
{
    double s = rParticle * rParticle + rField * rField
             - 2.0 * rParticle * rField * cos(dtheta);
    return sqrt(s > 0.0 ? s : 0.0);
}

/* As insp_nmode_components_time but at Mino time lambda, weighted by the
   Jacobian dt/dlambda so that integrating over [0, Vr] reproduces the
   coordinate-time integral over [0, Tr]. No Brent inversion. */
static void insp_nmode_components_lambda(double lambda, inspectre_nmode_params *ip,
                                         double *out)
{
    double PhiS[2], dPhiS[8], ddPhiS[20], src[2];
    inspectre_eval_at_lambda_fp(ip->ctx, ip->mMode, ip->fp, lambda, ip->orbpar,
                                ip->a, ip->p, ip->e, PhiS, dPhiS, ddPhiS, src);

    double t = korb_tfromla(lambda, *ip->orbpar);
    double J = insp_dtdlambda(lambda, ip->orbpar);

    double nField[2], nDeriv[8], nSrc[2];
    generateNModeIntegrands(t, ip->omegaPhi, ip->omegaR, ip->mMode, ip->nMode,
                            PhiS, dPhiS, src, nField, nDeriv, nSrc);

    for (int c = 0; c < 12; c++)
        out[c] = insp_select_component(c, nField, nDeriv, nSrc) * J;
}

/* gsl_function: as inspectre_nmode_integrand but parameterized by Mino time. */
double inspectre_nmode_integrand_lambda(double lambda, void *params)
{
    inspectre_nmode_params *ip = (inspectre_nmode_params *)params;

    double f[12];
    insp_nmode_components_lambda(lambda, ip, f);
    return f[ip->component];
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

void inspectre_mino_samples_build_fp(struct effsource_equatorial_ctx *ctx, int mMode,
        const inspectre_field_point *fp, korb_params *orbpar,
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
        double dist = insp_distance(fp->r, fp->dtheta, r_p);
        double floor = INSPECTRE_MINO_DIST_FLOOR * fp->r;  /* scale-relative clip */
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
        inspectre_eval_at_lambda_fp(ctx, mMode, fp, lam, orbpar, a, p, e,
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

/* Absolute-coordinate form of inspectre_mino_samples_build_fp: dtheta = theta - pi/2. */
void inspectre_mino_samples_build(struct effsource_equatorial_ctx *ctx, int mMode,
        struct coordinate *xField, korb_params *orbpar,
        double a, double p, double e, int nSamples,
        inspectre_mino_samples *out)
{
    const inspectre_field_point fp = { xField->r, xField->theta - M_PI_2,
                                       xField->phi };
    inspectre_mino_samples_build_fp(ctx, mMode, &fp, orbpar, a, p, e,
                                   nSamples, out);
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

/* ---------------------------------------------------------------------------
 * Panel Gauss-Legendre nodes split at closest approach (INSPECTRE_INTEG_PANEL_GL)
 * ------------------------------------------------------------------------- */

/* Invert psi(lambda) = psiC by bisection on [0, Vr/2], where psi increases
   monotonically from 0 to pi over the first half radial period. */
static double insp_lambda_from_psi(double psiC, korb_params *o)
{
    const double Vr = inspectre_radial_mino_period(o);
    double lo = 0.0, hi = 0.5 * Vr;
    if (psiC <= 0.0)  return lo;
    if (psiC >= M_PI) return hi;
    for (int i = 0; i < 120; i++)
    {
        double mid = 0.5 * (lo + hi);
        if (korb_psifromla(mid, *o) < psiC) lo = mid;
        else                                hi = mid;
        if (hi - lo <= 4.0 * DBL_EPSILON * Vr) break;
    }
    return 0.5 * (lo + hi);
}

static double insp_panel_dist(const inspectre_field_point *fp, korb_params *o,
                              double lam)
{
    double r_p = korb_rfrompsi(korb_psifromla(lam, *o), *o);
    return insp_distance(fp->r, fp->dtheta, r_p);
}

/* Lambda-scale of the peak at lamC: the offset at which the field-to-particle
   distance doubles (whichever side doubles first). Derivative-free, so it is
   equally valid at a transversal radial crossing (linear approach) and at a
   turning point (quadratic approach). Returns 0 for an exact crossing
   (d_min = 0): the caller then refines to the maxLevels floor. */
static double insp_peak_lambda_scale(const inspectre_field_point *fp,
                                     korb_params *o, double lamC)
{
    const double Vr = inspectre_radial_mino_period(o);
    double d0 = insp_panel_dist(fp, o, lamC);
    if (d0 <= 0.0) return 0.0;
    double dl = 1e-9 * Vr;
    while (dl < 0.25 * Vr)
    {
        double dp = insp_panel_dist(fp, o, lamC + dl);
        double dm = insp_panel_dist(fp, o, lamC - dl);
        if (dp >= 2.0 * d0 || dm >= 2.0 * d0) return dl;
        dl *= 2.0;
    }
    return 0.25 * Vr;
}

/* Closest-approach breakpoints. For r_f inside the libration range the radial
   phase of closest approach is psi_c = arccos((p/r_f - 1)/e), giving two
   breakpoints lam1 and Vr - lam1 (outbound/inbound legs); outside the range
   the clamp lands on the nearest turning point, a single breakpoint. */
static int insp_panel_find_peaks(const inspectre_field_point *fp, korb_params *o,
                                 double p, double e,
                                 double lamB[2], double dlam[2])
{
    const double Vr = inspectre_radial_mino_period(o);
    int nB;

    if (e <= 0.0)   /* circular: no radial peak; single arbitrary anchor */
    {
        lamB[0] = 0.0;
        dlam[0] = 0.25 * Vr;
        return 1;
    }

    double cosPsi = (p / fp->r - 1.0) / e;
    if (cosPsi >= 1.0)        { lamB[0] = 0.0;      nB = 1; }
    else if (cosPsi <= -1.0)  { lamB[0] = 0.5 * Vr; nB = 1; }
    else
    {
        double lam1 = insp_lambda_from_psi(acos(cosPsi), o);
        lamB[0] = lam1;
        lamB[1] = Vr - lam1;
        nB = (lamB[1] - lamB[0] > 1e-12 * Vr) ? 2 : 1;
    }

    for (int i = 0; i < nB; i++)
        dlam[i] = insp_peak_lambda_scale(fp, o, lamB[i]);
    return nB;
}

#define INSP_QAG_SCALE_NODES  16     /* coarse prepass mesh for the epsabs floor */
#define INSP_QAG_NOISE_FACTOR 64.0   /* multiples of DBL_EPSILON tolerated       */

/* Per-component absolute-error floor for the adaptive QAG modes.

   Roundoff in a complex n-mode quantity tracks the modulus of its (re, im)
   pair, not the individual part. A component whose exact integral vanishes --
   half of the twelve do on a circular orbit, where the m-carrier cancellation
   leaves the shifted integrand real -- therefore carries noise ~eps*|pair|,
   which no epsabs below that level can certify and epsrel cannot bound at all.
   QAG then bisects to `limit`: at r_f = p, e = 0 the Im(d_theta Phi) component
   ran 20000 intervals without converging.

   Sample each pair modulus on a coarse mesh plus the closest-approach
   breakpoints and raise epsabs to the level the integrand actually carries.
   Costs INSP_QAG_SCALE_NODES + 1 + nPeaks source evaluations for all twelve
   components, against the 12 * 61 minimum QAG itself needs. */
static void insp_qag_component_epsabs(int minoTime, inspectre_nmode_params *ip,
                                      double lo, double hi, double epsabs,
                                      double *epsabsC)
{
    double lamB[2], dlam[2];
    int nB = insp_panel_find_peaks(ip->fp, ip->orbpar, ip->p, ip->e, lamB, dlam);
    double scale[6] = { 0.0 };
    double f[12];

    for (int k = 0; k <= INSP_QAG_SCALE_NODES + nB; k++)
    {
        double x;
        if (k <= INSP_QAG_SCALE_NODES)
            x = lo + (hi - lo) * (double)k / (double)INSP_QAG_SCALE_NODES;
        else
        {
            double lam = lamB[k - INSP_QAG_SCALE_NODES - 1];
            x = minoTime ? lam : korb_tfromla(lam, *ip->orbpar);
        }

        if (minoTime) insp_nmode_components_lambda(x, ip, f);
        else          insp_nmode_components_time(x, ip, f);

        for (int j = 0; j < 6; j++)
        {
            double mod = hypot(f[2 * j], f[2 * j + 1]);
            if (isfinite(mod) && mod > scale[j]) scale[j] = mod;
        }
    }

    for (int c = 0; c < 12; c++)
        epsabsC[c] = fmax(epsabs, INSP_QAG_NOISE_FACTOR * DBL_EPSILON
                                  * scale[c / 2] * (hi - lo));
}

/* Geometric refinement level count toward a peak with lambda-scale dlamPeak,
   for a segment half-width h: halve until the innermost panel is no wider
   than the peak scale, capped at maxLevels. */
static int insp_panel_levels(double h, double dlamPeak, int maxLevels)
{
    if (dlamPeak >= h) return 0;
    if (dlamPeak <= 0.0) return maxLevels;
    int K = (int)ceil(log2(h / dlamPeak));
    if (K < 0) K = 0;
    if (K > maxLevels) K = maxLevels;
    return K;
}

void inspectre_panel_nodes_build_fp(struct effsource_equatorial_ctx *ctx, int mMode,
        const inspectre_field_point *fp, korb_params *orbpar,
        double a, double p, double e, int order, int maxLevels, int nMax,
        double omegaPhi, double omegaR,
        inspectre_panel_nodes *out)
{
    const double Vr = inspectre_radial_mino_period(orbpar);
    const double Tr = korb_tfromla(Vr, *orbpar);
    out->Vr = Vr;
    out->Tr = Tr;
    out->nNonFinite = 0;

    double lamB[2], dlam[2];
    int nB = insp_panel_find_peaks(fp, orbpar, p, e, lamB, dlam);
    out->nBreak = nB;
    for (int i = 0; i < nB; i++) { out->lamBreak[i] = lamB[i]; out->dlam[i] = dlam[i]; }

    /* segments between consecutive breakpoints, unwrapped from lamB[0] so the
       full domain [lamB[0], lamB[0]+Vr] is covered with peaks only at segment
       ends (never interior) */
    int nSeg = nB;
    double segA[2], segB[2];
    double dlamA[2], dlamB_[2];
    if (nB == 1)
    {
        segA[0] = lamB[0];          segB[0] = lamB[0] + Vr;
        dlamA[0] = dlam[0];         dlamB_[0] = dlam[0];
    }
    else
    {
        segA[0] = lamB[0];          segB[0] = lamB[1];
        dlamA[0] = dlam[0];         dlamB_[0] = dlam[1];
        segA[1] = lamB[1];          segB[1] = lamB[0] + Vr;
        dlamA[1] = dlam[1];         dlamB_[1] = dlam[0];
    }

    /* collect geometric panel edges [ga[i], gb[i]] */
    int maxGeo = nSeg * 2 * (maxLevels + 1);
    double *ga = malloc((size_t)maxGeo * sizeof(double));
    double *gb = malloc((size_t)maxGeo * sizeof(double));
    int nGeo = 0;
    for (int sgi = 0; sgi < nSeg; sgi++)
    {
        double A = segA[sgi], B = segB[sgi];
        double h = 0.5 * (B - A);
        int KA = insp_panel_levels(h, dlamA[sgi],  maxLevels);
        int KB = insp_panel_levels(h, dlamB_[sgi], maxLevels);

        /* toward A: sliver first, then geometrically growing panels to mid */
        ga[nGeo] = A;  gb[nGeo] = A + h * pow(2.0, -KA);  nGeo++;
        for (int k = KA; k >= 1; k--)
        {
            ga[nGeo] = A + h * pow(2.0, -k);
            gb[nGeo] = A + h * pow(2.0, -(k - 1));
            nGeo++;
        }
        /* toward B: mirror image */
        for (int k = 1; k <= KB; k++)
        {
            ga[nGeo] = B - h * pow(2.0, -(k - 1));
            gb[nGeo] = B - h * pow(2.0, -k);
            nGeo++;
        }
        ga[nGeo] = B - h * pow(2.0, -KB);  gb[nGeo] = B;  nGeo++;
    }

    /* Oscillation resolution: a q-point Gauss-Legendre rule only integrates
       exp(i w t) accurately while w*dt_panel <~ q, so subdivide each panel
       uniformly until the highest requested frequency w_max = |m| wphi +
       nMax*wr satisfies that bound. This is what caps the usable |n| of the
       node set; the geometric stack above only handles the peak. */
    double omegaMax = fabs((double)mMode) * fabs(omegaPhi)
                    + (double)(nMax > 0 ? nMax : 0) * fabs(omegaR);
    int nPan = 0;
    int *sub = malloc((size_t)nGeo * sizeof(int));
    for (int i = 0; i < nGeo; i++)
    {
        double Jmid = insp_dtdlambda(fmod(0.5 * (ga[i] + gb[i]), Vr), orbpar);
        double dt = (gb[i] - ga[i]) * Jmid;
        int ns = (omegaMax > 0.0) ? (int)ceil(omegaMax * dt / (double)order) : 1;
        if (ns < 1) ns = 1;
        sub[i] = ns;
        nPan += ns;
    }
    double *pa = malloc((size_t)nPan * sizeof(double));
    double *pb = malloc((size_t)nPan * sizeof(double));
    for (int i = 0, j = 0; i < nGeo; i++)
    {
        double w = (gb[i] - ga[i]) / (double)sub[i];
        for (int k = 0; k < sub[i]; k++, j++)
        {
            pa[j] = ga[i] + (double)k * w;
            pb[j] = (k == sub[i] - 1) ? gb[i] : ga[i] + (double)(k + 1) * w;
        }
    }
    free(ga); free(gb); free(sub);

    /* Gauss-Legendre nodes per panel + source samples */
    gsl_integration_glfixed_table *tbl =
        gsl_integration_glfixed_table_alloc((size_t)order);

    const int N = nPan * order;
    out->n   = N;
    out->lam = malloc((size_t)N * sizeof(double));
    out->t   = malloc((size_t)N * sizeof(double));
    out->J   = malloc((size_t)N * sizeof(double));
    out->w   = malloc((size_t)N * sizeof(double));
    out->raw = malloc((size_t)12 * N * sizeof(double));

    int idx = 0;
    for (int pi = 0; pi < nPan; pi++)
    {
        for (int j = 0; j < order; j++, idx++)
        {
            double lamj, wj;
            gsl_integration_glfixed_point(pa[pi], pb[pi], (size_t)j,
                                          &lamj, &wj, tbl);
            /* Nodes past the wrap are folded to the principal period for
               EVERYTHING -- source AND carrier time. The full n-mode
               integrand exp(i w_mn t) S_m(t) is exactly Tr-periodic (the
               m-carrier cancels the secular phase of S_m), but S_m alone is
               not: continuing t by Tr while sampling S_m at the principal
               lambda would tag wrapped nodes with a spurious
               exp(i m wphi Tr) factor. */
            double lamEval = lamj >= Vr ? lamj - Vr : lamj;

            out->lam[idx] = lamj;
            out->w[idx]   = wj;
            out->t[idx]   = korb_tfromla(lamEval, *orbpar);
            out->J[idx]   = insp_dtdlambda(lamEval, orbpar);

            double PhiS[2], dPhiS[8], ddPhiS[20], src[2];
            inspectre_eval_at_lambda_fp(ctx, mMode, fp, lamEval, orbpar,
                                     a, p, e, PhiS, dPhiS, ddPhiS, src);
            out->raw[0 * N + idx] = PhiS[0];
            out->raw[1 * N + idx] = PhiS[1];
            for (int c = 0; c < 8; c++)
                out->raw[(2 + c) * N + idx] = dPhiS[c];
            out->raw[10 * N + idx] = src[0];
            out->raw[11 * N + idx] = src[1];
            /* Innermost-sliver nodes at an exact crossing can land deep in the
               source's near zone (distance <~ 1e-8), where its intrinsic input
               cancellation can produce non-finite values. Their quadrature
               weight is ~2^-maxLevels of the period, so zeroing them changes
               the integral at only that level -- but a single NaN would poison
               the whole sum. Count them so callers can tell. */
            for (int c = 0; c < 12; c++)
                if (!isfinite(out->raw[c * N + idx]))
                {
                    out->raw[c * N + idx] = 0.0;
                    out->nNonFinite++;
                }
        }
    }

    gsl_integration_glfixed_table_free(tbl);
    free(pa); free(pb);
}

/* Absolute-coordinate form of inspectre_panel_nodes_build_fp: dtheta = theta - pi/2. */
void inspectre_panel_nodes_build(struct effsource_equatorial_ctx *ctx, int mMode,
        struct coordinate *xField, korb_params *orbpar,
        double a, double p, double e, int order, int maxLevels, int nMax,
        double omegaPhi, double omegaR,
        inspectre_panel_nodes *out)
{
    const inspectre_field_point fp = { xField->r, xField->theta - M_PI_2,
                                       xField->phi };
    inspectre_panel_nodes_build_fp(ctx, mMode, &fp, orbpar, a, p, e, order,
                                  maxLevels, nMax, omegaPhi, omegaR, out);
}

void inspectre_panel_nodes_free(inspectre_panel_nodes *s)
{
    free(s->lam); free(s->t); free(s->J); free(s->w); free(s->raw);
    s->lam = s->t = s->J = s->w = s->raw = NULL;
    s->n = 0;
}

void inspectre_panel_nodes_integrate(const inspectre_panel_nodes *s,
        int mMode, int nMode, double omegaPhi, double omegaR,
        double *nModePhiS, double *nModeDPhiS, double *nModesrc)
{
    const int N = s->n;
    double res[12] = { 0.0 };
    for (int k = 0; k < N; k++)
    {
        double Field[2] = { s->raw[0 * N + k], s->raw[1 * N + k] };
        double Deriv[8];
        for (int c = 0; c < 8; c++) Deriv[c] = s->raw[(2 + c) * N + k];
        double Src[2] = { s->raw[10 * N + k], s->raw[11 * N + k] };

        double nField[2], nDeriv[8], nSrc[2];
        generateNModeIntegrands(s->t[k], omegaPhi, omegaR, mMode, nMode,
                                Field, Deriv, Src, nField, nDeriv, nSrc);
        double W = s->w[k] * s->J[k];
        for (int c = 0; c < 12; c++)
            res[c] += W * insp_select_component(c, nField, nDeriv, nSrc);
    }
    for (int c = 0; c < 12; c++) res[c] /= s->Tr;
    insp_store_results(res, nModePhiS, nModeDPhiS, nModesrc);
}

void inspectre_integrate_nmode_fp(int mode,
        struct effsource_equatorial_ctx *ctx, int mMode, int nMode,
        const inspectre_field_point *fp, korb_params *orbpar,
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
        ip.fp = fp; ip.orbpar = orbpar;
        ip.a = a; ip.p = p; ip.e = e;
        ip.omegaPhi = omegaPhi; ip.omegaR = omegaR;

        gsl_function F;
        F.function = &inspectre_nmode_integrand;
        F.params   = &ip;

        double epsabsC[12];
        insp_qag_component_epsabs(0, &ip, 0.0, Tr, epsabs, epsabsC);

        for (int c = 0; c < 12; c++)
        {
            double result, abserr;
            ip.component = c;
            int st = gsl_integration_qag(&F, 0.0, Tr, epsabsC[c], epsrel, QAG_LIMIT,
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
        ip.fp = fp; ip.orbpar = orbpar;
        ip.a = a; ip.p = p; ip.e = e;
        ip.omegaPhi = omegaPhi; ip.omegaR = omegaR;

        gsl_function F;
        F.function = &inspectre_nmode_integrand_lambda;
        F.params   = &ip;

        const double Vr = inspectre_radial_mino_period(orbpar);
        double epsabsC[12];
        insp_qag_component_epsabs(1, &ip, 0.0, Vr, epsabs, epsabsC);

        for (int c = 0; c < 12; c++)
        {
            double result, abserr;
            ip.component = c;
            int st = gsl_integration_qag(&F, 0.0, Vr, epsabsC[c], epsrel, QAG_LIMIT,
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
        inspectre_mino_samples_build_fp(ctx, mMode, fp, orbpar, a, p, e,
                                     nSamples, &s);
        inspectre_mino_samples_integrate(&s, mMode, nMode, omegaPhi, omegaR,
                                         nModePhiS, nModeDPhiS, nModesrc);
        inspectre_mino_samples_free(&s);
        return;
    }
    else if (mode == INSPECTRE_INTEG_PANEL_GL)
    {
        /* Self-contained: split [0, Vr] at the analytic closest-approach
           breakpoints, refine panels geometrically toward them, Gauss-Legendre
           per panel. Default order/levels; callers sweeping many n should use
           inspectre_panel_nodes_build + _integrate directly. */
        inspectre_panel_nodes s;
        int nMax = abs(nMode) > 8 ? abs(nMode) : 8;
        inspectre_panel_nodes_build_fp(ctx, mMode, fp, orbpar, a, p, e,
                                    16, 40, nMax, omegaPhi, omegaR, &s);
        inspectre_panel_nodes_integrate(&s, mMode, nMode, omegaPhi, omegaR,
                                        nModePhiS, nModeDPhiS, nModesrc);
        inspectre_panel_nodes_free(&s);
        return;
    }
    else if (mode == INSPECTRE_INTEG_FACT_CONV)
    {
        /* Self-contained: seven-channel kernel factorization + convolution
           with the validated defaults (NB = 4 nMax, NK = 1<<20, KG = 400).
           The build is far heavier than one panel build; callers sweeping
           many n should use inspectre_fact_nodes_build + _integrate. */
        inspectre_fact_nodes s;
        int nMax = abs(nMode) > 64 ? abs(nMode) : 64;
        int NB = 4 * nMax;
        inspectre_fact_nodes_build_fp(ctx, mMode, fp, orbpar, a, p, e,
                                   NB, 1 << 20, 400, nMax, omegaPhi, omegaR, &s);
        inspectre_fact_nodes_integrate(&s, nMode,
                                       nModePhiS, nModeDPhiS, nModesrc);
        inspectre_fact_nodes_free(&s);
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

/* Absolute-coordinate form of inspectre_integrate_nmode_fp: dtheta = theta - pi/2. */
void inspectre_integrate_nmode(int mode,
        struct effsource_equatorial_ctx *ctx, int mMode, int nMode,
        struct coordinate *xField, korb_params *orbpar,
        double a, double p, double e, double omegaPhi, double omegaR,
        double epsabs, double epsrel,
        double *tSamples, double *fieldSamples, double *derivSamples,
        double *srcSamples, int nSamples,
        double *nModePhiS, double *nModeDPhiS, double *nModesrc)
{
    const inspectre_field_point fp = { xField->r, xField->theta - M_PI_2,
                                       xField->phi };
    inspectre_integrate_nmode_fp(mode, ctx, mMode, nMode, &fp, orbpar,
                                a, p, e, omegaPhi, omegaR, epsabs, epsrel,
                                tSamples, fieldSamples, derivSamples,
                                srcSamples, nSamples,
                                nModePhiS, nModeDPhiS, nModesrc);
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
void inspectre_fft_source_nmodes_fp(struct effsource_equatorial_ctx *ctx, int mMode,
        const inspectre_field_point *fp, korb_params *orbpar,
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
        inspectre_eval_at_lambda_fp(ctx, mMode, fp, lam, orbpar, a, p, e,
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

/* Absolute-coordinate form of inspectre_fft_source_nmodes_fp: dtheta = theta - pi/2. */
void inspectre_fft_source_nmodes(struct effsource_equatorial_ctx *ctx, int mMode,
        struct coordinate *xField, korb_params *orbpar,
        double a, double p, double e, double omegaPhi, double omegaR,
        int N, double *outRe, double *outIm)
{
    const inspectre_field_point fp = { xField->r, xField->theta - M_PI_2,
                                       xField->phi };
    inspectre_fft_source_nmodes_fp(ctx, mMode, &fp, orbpar, a, p, e,
                                  omegaPhi, omegaR, N, outRe, outIm);
}

/* ===========================================================================
 * (4) Kernel-factorization n-modes -- C port of inspectre/factorization.py
 *
 * The build samples the seven-channel calc_m_split on a uniform-t base grid
 * (fold applied so each channel series is Tr-periodic), FFTs the channels,
 * spectrally resamples the orbit series (r_p, alpha20, alpha02) to a dense
 * grid, and FFTs the six scalar kernels {ln a, 1/a, ..., 1/a^5}. Integration
 * per n is the convolution X_n = A^(n) + sum_ch sum_k ch^(k) K_ch^(n-k).
 * Conventions match numpy: X^(n) = (1/N) sum_j X_j exp(+2 pi i n j / N),
 * i.e. an FFTW_BACKWARD transform scaled by 1/N.
 * ========================================================================= */

/* channel-major complex component (ch, c) at base-grid node j:
   c = 0 -> PhiS, 1..4 -> dPhiS pairs, 5 -> src */
#define INSP_FACT_NCH   7
#define INSP_FACT_NCOMP 6
#define INSP_FACT_NKQ   6      /* kernels: ln a, 1/a .. 1/a^5 */

static int insp_fact_wrap(int j, int N) { return ((j % N) + N) % N; }

/* Spectral (zero-pad) resample of the real NB-periodic series x to NK points,
   matching factorization._resample_real: forward FFT, keep the lowest NB/2
   positive and NB/2 negative bins, backward FFT on the dense grid. planB_NK
   transforms inNK -> outNK (FFTW_BACKWARD, length NK); the result lands in
   xd (real part, numpy normalization). */
static void insp_fact_resample(const double *x, int NB, int NK,
        fftw_plan planF_NB, fftw_complex *inNB, fftw_complex *outNB,
        fftw_plan planB_NK, fftw_complex *inNK, fftw_complex *outNK,
        double *xd)
{
    const int h = NB / 2;
    for (int j = 0; j < NB; j++) { inNB[j][0] = x[j]; inNB[j][1] = 0.0; }
    fftw_execute(planF_NB);
    for (int k = 0; k < NK; k++) { inNK[k][0] = 0.0; inNK[k][1] = 0.0; }
    for (int k = 0; k < h; k++)
    {
        inNK[k][0] = outNB[k][0];               inNK[k][1] = outNB[k][1];
        inNK[NK - h + k][0] = outNB[h + k][0];  inNK[NK - h + k][1] = outNB[h + k][1];
    }
    fftw_execute(planB_NK);
    for (int k = 0; k < NK; k++) xd[k] = outNK[k][0] / (double)NB;
}

void inspectre_fact_nodes_build_fp(struct effsource_equatorial_ctx *ctx, int mMode,
        const inspectre_field_point *fp, korb_params *orbpar,
        double a, double p, double e, int NB, int NK, int KG, int nMax,
        double omegaPhi, double omegaR,
        inspectre_fact_nodes *out)
{
    (void)omegaR;   /* carried for signature symmetry with the panel build */

    if (NB < 4) NB = 4;
    if (NB % 2) NB++;                    /* resample needs NB even */
    if (KG < 0) KG = 0;
    if (KG > NB / 2) KG = NB / 2;        /* beyond NB/2 the base-grid FFT bins
                                            wrap and k would double-count */
    if (nMax < 0) nMax = 0;
    const int nKmax = nMax + KG;
    if (NK < 2) NK = 2;
    while (NK < 2 * nKmax + 2 || NK < 2 * NB) NK *= 2;

    const double Vr = inspectre_radial_mino_period(orbpar);
    const double Tr = korb_tfromla(Vr, *orbpar);
    out->NB = NB; out->NK = NK; out->KG = KG;
    out->nMax = nMax; out->nKmax = nKmax;
    out->nNonFinite = 0;
    out->Tr = Tr; out->Vr = Vr;
    out->chat = malloc((size_t)2 * INSP_FACT_NCH * INSP_FACT_NCOMP * NB
                       * sizeof(double));
    out->khat = malloc((size_t)2 * INSP_FACT_NKQ * (2 * (size_t)nKmax + 1)
                       * sizeof(double));

    /* ---- sample the split channels + orbit series on the base grid ---- */
    double *g   = malloc((size_t)2 * INSP_FACT_NCH * INSP_FACT_NCOMP * NB
                         * sizeof(double));
    double *drP = malloc((size_t)NB * sizeof(double));
    double *a20 = malloc((size_t)NB * sizeof(double));
    double *a02 = malloc((size_t)NB * sizeof(double));
    const double dth = fp->dtheta;

    for (int j = 0; j < NB; j++)
    {
        double t   = (double)j * Tr / (double)NB;
        double lam = inspectre_lambda_from_t(t, orbpar);
        double psi = korb_psifromla(lam, *orbpar);
        double r_p = korb_rfrompsi(psi, *orbpar);
        double phi_p = korb_phifromla(lam, *orbpar);
        double ur  = fourVel(psi, a, p, e, orbpar->E);

        struct coordinate xParticle;
        xParticle.t = 0.0; xParticle.r = r_p;
        xParticle.theta = M_PI_2; xParticle.phi = phi_p;
        effsource_equatorial_ctx_set_particle(ctx, &xParticle,
                                              orbpar->E, orbpar->Lz, ur);

        const long double r_p_l = (long double)p
            / (1.0L + (long double)e * cosl((long double)psi));
        const double dr = (double)((long double)fp->r - r_p_l);

        insp_eval_hits++;
        double PhiS_s[14], dPhiS_s[56], d2PhiS_s[140], src_s[14], al[4];
        effsource_equatorial_ctx_calc_m_split(ctx, mMode, dr, dth,
                                              PhiS_s, dPhiS_s, d2PhiS_s, src_s);
        effsource_equatorial_ctx_get_alpha(ctx, al);
        drP[j] = dr; a20[j] = al[0]; a02[j] = al[1];

        /* fold exp(i m wphi t): cancels the secular phase so every channel
           series is Tr-periodic (same role as the m-carrier in the panel /
           FFT paths) */
        double fr = cos((double)mMode * omegaPhi * t);
        double fi = sin((double)mMode * omegaPhi * t);
        for (int ch = 0; ch < INSP_FACT_NCH; ch++)
            for (int c = 0; c < INSP_FACT_NCOMP; c++)
            {
                double re, im;
                if (c == 0)      { re = PhiS_s[ch * 2];  im = PhiS_s[ch * 2 + 1]; }
                else if (c < 5)  { re = dPhiS_s[ch * 8 + 2 * (c - 1)];
                                   im = dPhiS_s[ch * 8 + 2 * (c - 1) + 1]; }
                else             { re = src_s[ch * 2];   im = src_s[ch * 2 + 1]; }
                if (!isfinite(re) || !isfinite(im))
                {
                    re = 0.0; im = 0.0;
                    out->nNonFinite++;
                }
                size_t o = 2 * ((size_t)(ch * INSP_FACT_NCOMP + c) * NB + j);
                g[o]     = fr * re - fi * im;
                g[o + 1] = fr * im + fi * re;
            }
    }

    /* ---- channel FFTs (backward / NB = numpy ifft) ---- */
    fftw_complex *inNB  = fftw_malloc(sizeof(fftw_complex) * (size_t)NB);
    fftw_complex *outNB = fftw_malloc(sizeof(fftw_complex) * (size_t)NB);
    fftw_plan planB_NB = fftw_plan_dft_1d(NB, inNB, outNB,
                                          FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_plan planF_NB = fftw_plan_dft_1d(NB, inNB, outNB,
                                          FFTW_FORWARD, FFTW_ESTIMATE);
    for (int s = 0; s < INSP_FACT_NCH * INSP_FACT_NCOMP; s++)
    {
        for (int j = 0; j < NB; j++)
        {
            inNB[j][0] = g[2 * ((size_t)s * NB + j)];
            inNB[j][1] = g[2 * ((size_t)s * NB + j) + 1];
        }
        fftw_execute(planB_NB);
        for (int j = 0; j < NB; j++)
        {
            out->chat[2 * ((size_t)s * NB + j)]     = outNB[j][0] / (double)NB;
            out->chat[2 * ((size_t)s * NB + j) + 1] = outNB[j][1] / (double)NB;
        }
    }
    free(g);

    /* ---- dense kernel grid: spectral resample of the orbit series ---- */
    fftw_complex *inNK  = fftw_malloc(sizeof(fftw_complex) * (size_t)NK);
    fftw_complex *outNK = fftw_malloc(sizeof(fftw_complex) * (size_t)NK);
    fftw_plan planB_NK = fftw_plan_dft_1d(NK, inNK, outNK,
                                          FFTW_BACKWARD, FFTW_ESTIMATE);

    double *alphaD = malloc((size_t)NK * sizeof(double));
    double *tmpD   = malloc((size_t)NK * sizeof(double));
    /* alpha_d = a20_d dr_d^2 + a02_d dth^2, accumulated so only two dense
       scratch arrays are alive at once. dr is resampled directly rather than
       resampling r_p and subtracting r_f: the resample is linear, so the two
       agree exactly in arithmetic, but interpolating the small quantity keeps
       dr's relative accuracy instead of rounding it through r_p's magnitude. */
    insp_fact_resample(drP, NB, NK, planF_NB, inNB, outNB,
                       planB_NK, inNK, outNK, tmpD);
    for (int k = 0; k < NK; k++) alphaD[k] = tmpD[k] * tmpD[k];
    insp_fact_resample(a20, NB, NK, planF_NB, inNB, outNB,
                       planB_NK, inNK, outNK, tmpD);
    for (int k = 0; k < NK; k++) alphaD[k] *= tmpD[k];
    insp_fact_resample(a02, NB, NK, planF_NB, inNB, outNB,
                       planB_NK, inNK, outNK, tmpD);
    for (int k = 0; k < NK; k++) alphaD[k] += tmpD[k] * dth * dth;
    free(drP); free(a20); free(a02);

    /* ---- kernel FFTs, trimmed to |j| <= nKmax ---- */
    double *invD = tmpD;                          /* reuse as 1/alpha scratch */
    for (int k = 0; k < NK; k++) invD[k] = 1.0 / alphaD[k];
    const size_t kW = 2 * (size_t)nKmax + 1;
    for (int kq = 0; kq < INSP_FACT_NKQ; kq++)
    {
        for (int k = 0; k < NK; k++)
        {
            double v = (kq == 0) ? log(alphaD[k]) : alphaD[k];
            if (!isfinite(v)) { v = 0.0; out->nNonFinite++; }
            inNK[k][0] = v; inNK[k][1] = 0.0;
        }
        fftw_execute(planB_NK);
        for (int j = -nKmax; j <= nKmax; j++)
        {
            int bin = insp_fact_wrap(j, NK);
            out->khat[2 * ((size_t)kq * kW + (size_t)(j + nKmax))]
                = outNK[bin][0] / (double)NK;
            out->khat[2 * ((size_t)kq * kW + (size_t)(j + nKmax)) + 1]
                = outNK[bin][1] / (double)NK;
        }
        /* next power: alpha^-1 after the log pass, then alpha^-(q+1) */
        if (kq < INSP_FACT_NKQ - 1)
            for (int k = 0; k < NK; k++)
                alphaD[k] = (kq == 0) ? invD[k] : alphaD[k] * invD[k];
    }

    free(alphaD); free(tmpD);
    fftw_destroy_plan(planB_NB); fftw_destroy_plan(planF_NB);
    fftw_destroy_plan(planB_NK);
    fftw_free(inNB); fftw_free(outNB);
    fftw_free(inNK); fftw_free(outNK);
}

/* Absolute-coordinate form of inspectre_fact_nodes_build_fp: dtheta = theta - pi/2. */
void inspectre_fact_nodes_build(struct effsource_equatorial_ctx *ctx, int mMode,
        struct coordinate *xField, korb_params *orbpar,
        double a, double p, double e, int NB, int NK, int KG, int nMax,
        double omegaPhi, double omegaR,
        inspectre_fact_nodes *out)
{
    const inspectre_field_point fp = { xField->r, xField->theta - M_PI_2,
                                       xField->phi };
    inspectre_fact_nodes_build_fp(ctx, mMode, &fp, orbpar, a, p, e, NB, NK,
                                 KG, nMax, omegaPhi, omegaR, out);
}

void inspectre_fact_nodes_free(inspectre_fact_nodes *s)
{
    free(s->chat); free(s->khat);
    s->chat = s->khat = NULL;
    s->NB = 0;
}

void inspectre_fact_nodes_integrate(const inspectre_fact_nodes *s, int nMode,
        double *nModePhiS, double *nModeDPhiS, double *nModesrc)
{
    const int NB = s->NB, KG = s->KG, nKmax = s->nKmax;
    const size_t kW = 2 * (size_t)nKmax + 1;
    double res[12] = { 0.0 };

    /* The accumulators are long double because this sum cancels hard. The
       channels carry the same mutual cancellation in Fourier space that they do
       pointwise: measured sum|term| / |result| for src is 2.6e9 at n = 0,
       2.1e10 at n = 8 and 1.7e12 at n = 32 over ~1500 terms, so a double
       accumulator loses 4.8e-08, 4.8e-07 and 5.9e-05 respectively. PhiS
       cancels only by ~5 and is unaffected either way.

       This fixes the accumulation, not the inputs: chat and khat are FFTW
       outputs and so are double to begin with, which leaves a floor of order
       cond * DBL_EPSILON that only long-double transforms (fftw3l) would
       remove. */
    for (int c = 0; c < INSP_FACT_NCOMP; c++)
    {
        /* A channel: direct read of the base-grid FFT bin */
        size_t oA = 2 * ((size_t)(0 * INSP_FACT_NCOMP + c) * NB
                         + insp_fact_wrap(nMode, NB));
        long double xr = (long double)s->chat[oA];
        long double xi = (long double)s->chat[oA + 1];

        /* L, P1..P5: convolution against the trimmed kernel coefficients */
        for (int ch = 1; ch < INSP_FACT_NCH; ch++)
        {
            const int kq = ch - 1;
            for (int k = -KG; k <= KG; k++)
            {
                int j = nMode - k;
                if (j < -nKmax || j > nKmax) continue;
                size_t oc = 2 * ((size_t)(ch * INSP_FACT_NCOMP + c) * NB
                                 + insp_fact_wrap(k, NB));
                size_t ok = 2 * ((size_t)kq * kW + (size_t)(j + nKmax));
                long double cr = (long double)s->chat[oc];
                long double ci = (long double)s->chat[oc + 1];
                long double kr = (long double)s->khat[ok];
                long double ki = (long double)s->khat[ok + 1];
                xr += cr * kr - ci * ki;
                xi += cr * ki + ci * kr;
            }
        }

        if (c == 0)      { res[0] = (double)xr;         res[1] = (double)xi; }
        else if (c < 5)  { res[2 * c] = (double)xr;     res[2 * c + 1] = (double)xi; }
        else             { res[10] = (double)xr;        res[11] = (double)xi; }
    }

    insp_store_results(res, nModePhiS, nModeDPhiS, nModesrc);
}
