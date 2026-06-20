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
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>

#include "../include/inspectre.h"

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

    return e * sin(psi) / p * (X2 + a * a + 2.0 * X * a * E - 2.0 * X2 / p * (3 + e * cos(psi)));
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

void inspectre_eval_at_lambda(struct effsource_equatorial_ctx *ctx, int mMode,
                              struct coordinate *xField, double lambda,
                              korb_params *orbpar, double a, double p, double e,
                              double *PhiS, double *dPhiS, double *ddPhiS,
                              double *src)
{
    /* particle position from the orbit at this Mino time */
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
    double Tr = korb_tfromla(orbpar->Vr, *orbpar);

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
            if (st && st != GSL_EROUND)
                fprintf(stderr, "[qag] c=%d status=%d (%s)\n", c, st, gsl_strerror(st));
            res[c] = result / Tr;
        }

        gsl_integration_workspace_free(w);
    }
    else /* sample-based methods integrate the precomputed samples */
    {
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
        {
            /* cubic-spline quadrature: fit each component vs t and integrate
               the spline analytically over [t0, t_{N-1}]. */
            gsl_interp_accel *acc = gsl_interp_accel_alloc();
            gsl_spline *sp = gsl_spline_alloc(gsl_interp_cspline, nSamples);
            double t0 = tSamples[0], tN = tSamples[nSamples - 1];
            for (int c = 0; c < 12; c++)
            {
                gsl_spline_init(sp, tSamples, &comp[c * nSamples], nSamples);
                res[c] = gsl_spline_eval_integ(sp, t0, tN, acc) / Tr;
                gsl_interp_accel_reset(acc);
            }
            gsl_spline_free(sp);
            gsl_interp_accel_free(acc);
        }
        else if (mode == INSPECTRE_INTEG_SIMPSON)
        {
            /* composite Simpson over consecutive interval pairs, written in the
               general unequal-spacing form (reduces to the 1/3 rule on the
               uniform-t grid the driver is fed). If the interval count
               (nSamples-1) is odd, the trailing single interval is closed with
               the trapezoid rule. */
            for (int c = 0; c < 12; c++) res[c] = 0.0;
            int i;
            for (i = 0; i + 2 < nSamples; i += 2)
            {
                double h0 = tSamples[i + 1] - tSamples[i];
                double h1 = tSamples[i + 2] - tSamples[i + 1];
                double w  = (h0 + h1) / 6.0;
                for (int c = 0; c < 12; c++)
                {
                    double *f = &comp[c * nSamples + i];
                    res[c] += w * ((2.0 - h1 / h0) * f[0]
                                 + (h0 + h1) * (h0 + h1) / (h0 * h1) * f[1]
                                 + (2.0 - h0 / h1) * f[2]);
                }
            }
            if (i + 1 < nSamples)   /* odd interval count: trapezoid on the last */
            {
                double dt = tSamples[i + 1] - tSamples[i];
                for (int c = 0; c < 12; c++)
                    res[c] += 0.5 * (comp[c * nSamples + i]
                                   + comp[c * nSamples + i + 1]) * dt;
            }
            for (int c = 0; c < 12; c++) res[c] /= Tr;
        }
        else /* INSPECTRE_INTEG_TIMESERIES: trapezoid over the samples */
        {
            for (int c = 0; c < 12; c++) res[c] = 0.0;
            for (int i = 1; i < nSamples; i++)
            {
                double dt = tSamples[i] - tSamples[i - 1];
                for (int c = 0; c < 12; c++)
                    res[c] += 0.5 * (comp[c * nSamples + i - 1]
                                   + comp[c * nSamples + i]) * dt;
            }
            for (int c = 0; c < 12; c++) res[c] /= Tr;
        }

        free(comp);
    }

    insp_store_results(res, nModePhiS, nModeDPhiS, nModesrc);
}
