/* Verification driver for inspectre_lib:
 *   (a) eval_at_time(korb_tfromla(lambda0)) reproduces eval_at_lambda(lambda0)
 *   (b) QAG n-mode amplitude agrees with the timeseries trapezoid amplitude
 *
 * Build: see test/Makefile target `verify_lib`.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_errno.h>

#include "../include/inspectre.h"

int main(int argc, char *argv[])
{
    gsl_set_error_handler_off();

    double a = 0.9, p = 10.0, e = 0.1, x = 1.0;
    int mMode = 2, nMode = 0;
    if (argc >= 5) { a = atof(argv[1]); p = atof(argv[2]); e = atof(argv[3]); x = atof(argv[4]); }

    int eccentric = e > 0.0 ? 1 : 0;
    int inclined  = fabs(1.0 - x) <= 1e-14 ? 0 : 1;

    korb_params orbpar;
    korb_getparams(eccentric, inclined, a, p, e, x, 1.0e-15, &orbpar);

    struct effsource_equatorial_ctx *ctx = effsource_equatorial_create(1.0, a);

    /* field point outside the orbit radial range (rMax = p/(1-e)) so the
       n-mode integrand is smooth and QAG does not sample the puncture. */
    struct coordinate xField = { .t = 0.0, .phi = 0.0, .r = 15.0, .theta = M_PI_2 };

    double omegaPhi = orbpar.wphi, omegaR = orbpar.wr;
    double Vr = orbpar.Vr;
    double Tr = korb_tfromla(Vr, orbpar);

    /* ---- (a) t-wrapper round trip ---- */
    double lambda0 = 0.3 * Vr;
    double t0 = korb_tfromla(lambda0, orbpar);

    double PhiS_a[2], dPhiS_a[8], ddPhiS_a[20], src_a[2];
    double PhiS_b[2], dPhiS_b[8], ddPhiS_b[20], src_b[2];

    inspectre_eval_at_lambda(ctx, mMode, &xField, lambda0, &orbpar, a, p, e,
                             PhiS_a, dPhiS_a, ddPhiS_a, src_a);
    inspectre_eval_at_time(ctx, mMode, &xField, t0, &orbpar, a, p, e,
                           PhiS_b, dPhiS_b, ddPhiS_b, src_b);

    printf("=== (a) t-wrapper round trip (lambda0 = %.6g, t0 = %.6g) ===\n", lambda0, t0);
    printf("PhiS  lambda: % .15e % .15e\n", PhiS_a[0], PhiS_a[1]);
    printf("PhiS  time:   % .15e % .15e\n", PhiS_b[0], PhiS_b[1]);
    printf("src   lambda: % .15e % .15e\n", src_a[0], src_a[1]);
    printf("src   time:   % .15e % .15e\n", src_b[0], src_b[1]);
    printf("max|PhiS diff| = %.3e   max|src diff| = %.3e\n\n",
           fmax(fabs(PhiS_a[0]-PhiS_b[0]), fabs(PhiS_a[1]-PhiS_b[1])),
           fmax(fabs(src_a[0]-src_b[0]), fabs(src_a[1]-src_b[1])));

    /* ---- build a timeseries (uniform in lambda) for the trapezoid path ---- */
    int N = 4000;
    double *tS = malloc(N * sizeof(double));
    double *fS = malloc(2 * N * sizeof(double));
    double *dS = malloc(8 * N * sizeof(double));
    double *sS = malloc(2 * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        double lam = (double)i / (double)(N - 1) * Vr;
        double PhiS[2], dPhiS[8], ddPhiS[20], src[2];
        inspectre_eval_at_lambda(ctx, mMode, &xField, lam, &orbpar, a, p, e,
                                 PhiS, dPhiS, ddPhiS, src);
        tS[i] = korb_tfromla(lam, orbpar);
        fS[2*i] = PhiS[0]; fS[2*i+1] = PhiS[1];
        for (int k = 0; k < 8; k++) dS[8*i+k] = dPhiS[k];
        sS[2*i] = src[0]; sS[2*i+1] = src[1];
    }

    double tsPhiS[2], tsDPhiS[8], tsSrc[2];
    double qgPhiS[2], qgDPhiS[8], qgSrc[2];

    inspectre_integrate_nmode(INSPECTRE_INTEG_TIMESERIES, ctx, mMode, nMode,
            &xField, &orbpar, a, p, e, omegaPhi, omegaR, 1e-10, 1e-10,
            tS, fS, dS, sS, N, tsPhiS, tsDPhiS, tsSrc);

    inspectre_integrate_nmode(INSPECTRE_INTEG_QAG, ctx, mMode, nMode,
            &xField, &orbpar, a, p, e, omegaPhi, omegaR, 1e-10, 1e-8,
            NULL, NULL, NULL, NULL, 0, qgPhiS, qgDPhiS, qgSrc);

    printf("=== (b) n-mode amplitude: QAG vs timeseries (N=%d, Tr=%.6g) ===\n", N, Tr);
    printf("PhiS  QAG:  % .12e % .12e\n", qgPhiS[0], qgPhiS[1]);
    printf("PhiS  ts:   % .12e % .12e\n", tsPhiS[0], tsPhiS[1]);
    printf("src   QAG:  % .12e % .12e\n", qgSrc[0], qgSrc[1]);
    printf("src   ts:   % .12e % .12e\n", tsSrc[0], tsSrc[1]);
    printf("|PhiS diff| = %.3e   |src diff| = %.3e\n",
           fmax(fabs(qgPhiS[0]-tsPhiS[0]), fabs(qgPhiS[1]-tsPhiS[1])),
           fmax(fabs(qgSrc[0]-tsSrc[0]), fabs(qgSrc[1]-tsSrc[1])));

    free(tS); free(fS); free(dS); free(sS);
    effsource_equatorial_free(ctx);
    korb_freepar(orbpar);
    return 0;
}
