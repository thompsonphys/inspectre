/* Numerical integration test driver for inspectre_lib.
 *
 * Reads run parameters from a key=value config file, builds a timeseries that is
 * uniform in coordinate time t (via a lambda(t) cubic spline), then integrates
 * the n-mode amplitude with every method named in `methods` and reports each
 * result against the QAG reference. Report-only: always exits 0.
 *
 * Build: see test/Makefile target `numtest`.
 * Usage: numtest [config-path]   (default: test/inspectre.cfg)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

#include "../include/inspectre.h"

/* ---- config ------------------------------------------------------------- */

typedef struct {
    double a, p, e, x;
    int    mMode, nMode;
    double rField, theta;
    int    N;
    double epsabs, epsrel;
    char   methods[256];
} config;

static void config_defaults(config *cfg)
{
    cfg->a = 0.9; cfg->p = 10.0; cfg->e = 0.1; cfg->x = 1.0;
    cfg->mMode = 2; cfg->nMode = 0;
    cfg->rField = 15.0; cfg->theta = M_PI_2;
    cfg->N = 4000;
    cfg->epsabs = 1e-10; cfg->epsrel = 1e-8;
    strcpy(cfg->methods, "qag,trap,simpson,spline");
}

/* trim leading/trailing whitespace in place, returning the trimmed start */
static char *trim(char *s)
{
    while (*s && isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) *--end = '\0';
    return s;
}

/* parse a key=value config file; '#' begins a comment. Unknown keys warn. */
static int config_load(config *cfg, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "numtest: cannot open config '%s'\n", path); return -1; }

    char line[512];
    while (fgets(line, sizeof line, f))
    {
        char *hash = strchr(line, '#');
        if (hash) *hash = '\0';
        char *eq = strchr(line, '=');
        if (!eq) { if (*trim(line)) {} continue; }   /* blank/comment-only ok */

        *eq = '\0';
        char *key = trim(line);
        char *val = trim(eq + 1);
        if (!*key) continue;

        if      (!strcmp(key, "a"))       cfg->a = atof(val);
        else if (!strcmp(key, "p"))       cfg->p = atof(val);
        else if (!strcmp(key, "e"))       cfg->e = atof(val);
        else if (!strcmp(key, "x"))       cfg->x = atof(val);
        else if (!strcmp(key, "mMode"))   cfg->mMode = atoi(val);
        else if (!strcmp(key, "nMode"))   cfg->nMode = atoi(val);
        else if (!strcmp(key, "rField"))  cfg->rField = atof(val);
        else if (!strcmp(key, "theta"))   cfg->theta = atof(val);
        else if (!strcmp(key, "N"))       cfg->N = atoi(val);
        else if (!strcmp(key, "epsabs"))  cfg->epsabs = atof(val);
        else if (!strcmp(key, "epsrel"))  cfg->epsrel = atof(val);
        else if (!strcmp(key, "methods")) { strncpy(cfg->methods, val, sizeof cfg->methods - 1);
                                            cfg->methods[sizeof cfg->methods - 1] = '\0'; }
        else fprintf(stderr, "numtest: unknown config key '%s' (ignored)\n", key);
    }
    fclose(f);
    return 0;
}

/* map a method name to an INSPECTRE_INTEG_* mode; -1 if unknown */
static int method_mode(const char *name)
{
    if (!strcmp(name, "qag"))     return INSPECTRE_INTEG_QAG;
    if (!strcmp(name, "trap"))    return INSPECTRE_INTEG_TIMESERIES;
    if (!strcmp(name, "simpson")) return INSPECTRE_INTEG_SIMPSON;
    if (!strcmp(name, "spline"))  return INSPECTRE_INTEG_SPLINE;
    return -1;
}

/* ---- driver ------------------------------------------------------------- */

int main(int argc, char *argv[])
{
    gsl_set_error_handler_off();

    const char *cfgpath = (argc >= 2) ? argv[1] : "test/inspectre.cfg";
    config cfg;
    config_defaults(&cfg);
    if (config_load(&cfg, cfgpath) != 0) return 1;

    double a = cfg.a, p = cfg.p, e = cfg.e, x = cfg.x;
    int mMode = cfg.mMode, nMode = cfg.nMode;
    int N = cfg.N;

    printf("=== numtest (config: %s) ===\n", cfgpath);
    printf("a=%g p=%g e=%g x=%g  mMode=%d nMode=%d  rField=%g theta=%g  N=%d\n",
           a, p, e, x, mMode, nMode, cfg.rField, cfg.theta, N);
    printf("methods = %s\n\n", cfg.methods);

    int eccentric = e > 0.0 ? 1 : 0;
    int inclined  = fabs(1.0 - x) <= 1e-14 ? 0 : 1;

    korb_params orbpar;
    korb_getparams(eccentric, inclined, a, p, e, x, 1.0e-15, &orbpar);

    struct effsource_equatorial_ctx *ctx = effsource_equatorial_create(1.0, a);
    struct coordinate xField = { .t = 0.0, .phi = 0.0, .r = cfg.rField, .theta = cfg.theta };

    double omegaPhi = orbpar.wphi, omegaR = orbpar.wr;
    double Vr = orbpar.Vr;
    double Tr = korb_tfromla(Vr, orbpar);

    /* ---- build a timeseries uniform in coordinate time t ----
       1. sample lambda uniformly on [0, Vr] and tabulate t(lambda) (monotone);
       2. spline lambda(t) and read it off a uniform-t grid for the source. */
    double *latT  = malloc(N * sizeof(double));   /* t(lambda) tabulation     */
    double *latL  = malloc(N * sizeof(double));   /* the lambda samples       */
    for (int i = 0; i < N; i++) {
        double lam = (double)i / (double)(N - 1) * Vr;
        latL[i] = lam;
        latT[i] = korb_tfromla(lam, orbpar);
    }
    gsl_interp_accel *latAcc = gsl_interp_accel_alloc();
    gsl_spline *latSpline = gsl_spline_alloc(gsl_interp_cspline, N);
    gsl_spline_init(latSpline, latT, latL, N);

    double *tS = malloc(N * sizeof(double));
    double *fS = malloc(2 * N * sizeof(double));
    double *dS = malloc(8 * N * sizeof(double));
    double *sS = malloc(2 * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        double t   = (double)i / (double)(N - 1) * Tr;
        double lam = gsl_spline_eval(latSpline, t, latAcc);
        double PhiS[2], dPhiS[8], ddPhiS[20], src[2];
        inspectre_eval_at_lambda(ctx, mMode, &xField, lam, &orbpar, a, p, e,
                                 PhiS, dPhiS, ddPhiS, src);
        tS[i] = t;
        fS[2*i] = PhiS[0]; fS[2*i+1] = PhiS[1];
        for (int k = 0; k < 8; k++) dS[8*i+k] = dPhiS[k];
        sS[2*i] = src[0]; sS[2*i+1] = src[1];
    }
    gsl_spline_free(latSpline);
    gsl_interp_accel_free(latAcc);
    free(latT); free(latL);

    /* ---- QAG reference (always computed for the diff column) ---- */
    double refPhiS[2], refDPhiS[8], refSrc[2];
    inspectre_integrate_nmode(INSPECTRE_INTEG_QAG, ctx, mMode, nMode,
            &xField, &orbpar, a, p, e, omegaPhi, omegaR, cfg.epsabs, cfg.epsrel,
            NULL, NULL, NULL, NULL, 0, refPhiS, refDPhiS, refSrc);

    printf("=== n-mode amplitude per method (Tr=%.6g, diff vs QAG) ===\n", Tr);
    printf("%-8s  %-25s  %-25s  %-11s  %-11s\n",
           "method", "PhiS (re, im)", "src (re, im)", "|PhiS-QAG|", "|src-QAG|");

    /* ---- run each configured method ---- */
    char list[256];
    strncpy(list, cfg.methods, sizeof list - 1);
    list[sizeof list - 1] = '\0';
    for (char *tok = strtok(list, ", \t"); tok; tok = strtok(NULL, ", \t"))
    {
        int mode = method_mode(tok);
        if (mode < 0) { fprintf(stderr, "numtest: unknown method '%s' (skipped)\n", tok); continue; }

        double PhiS[2], dPhiS[8], src[2];
        inspectre_integrate_nmode(mode, ctx, mMode, nMode,
                &xField, &orbpar, a, p, e, omegaPhi, omegaR, cfg.epsabs, cfg.epsrel,
                tS, fS, dS, sS, N, PhiS, dPhiS, src);

        double dPhi = fmax(fabs(PhiS[0]-refPhiS[0]), fabs(PhiS[1]-refPhiS[1]));
        double dSrc = fmax(fabs(src[0]-refSrc[0]),   fabs(src[1]-refSrc[1]));
        printf("%-8s  % .9e % .9e  % .9e % .9e  %.4e  %.4e\n",
               tok, PhiS[0], PhiS[1], src[0], src[1], dPhi, dSrc);
    }

    free(tS); free(fS); free(dS); free(sS);
    effsource_equatorial_free(ctx);
    korb_freepar(orbpar);
    return 0;
}
