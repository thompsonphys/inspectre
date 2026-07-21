/* Source-integration parameter scan for inspectre_lib.
 *
 * Scans a grid of field points (rField, theta). At each point it computes the
 * n-mode SOURCE amplitude with the adaptive GSL QAG quadrature and with the
 * fixed-grid methods (trapezoid / Simpson / spline) over a sweep of time
 * resolutions N. Error is measured against a high-accuracy QAG reference (QAG at
 * a tighter tolerance). Both the error and the wall-time of each method are
 * written to a CSV file.
 *
 * Only the source component (nModesrc) is reported; PhiS and its derivatives are
 * evaluated by inspectre_eval_at_lambda along with the source (they cannot be
 * split) but are not written out.
 *
 * Build: see test/Makefile target `scantest`.
 * Usage: scantest [config-path]   (default: test/scan.cfg)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

#include "../include/inspectre.h"

/* ---- config ------------------------------------------------------------- */

#define MAX_NVALUES 64

typedef struct {
    double a, p, e, x;
    int    mMode, nMode;
    double rFieldMin, rFieldMax;
    int    rFieldCount;
    double thetaMin, thetaMax;
    int    thetaCount;
    int    Nvalues[MAX_NVALUES];
    int    nN;
    double epsabs, epsrel;
    double refEpsabs, refEpsrel;
    char   methods[256];
    int    repeats;
    char   outfile[256];
} config;

static void config_defaults(config *cfg)
{
    cfg->a = 0.5; cfg->p = 10.0; cfg->e = 0.3; cfg->x = 1.0;
    cfg->mMode = 2; cfg->nMode = 2;
    cfg->rFieldMin = 8.0; cfg->rFieldMax = 14.0; cfg->rFieldCount = 7;
    cfg->thetaMin = M_PI_2 - 0.08; cfg->thetaMax = M_PI_2 + 0.08; cfg->thetaCount = 5;
    cfg->Nvalues[0] = 400; cfg->Nvalues[1] = 512; cfg->Nvalues[2] = 1024;
    cfg->Nvalues[3] = 2048; cfg->Nvalues[4] = 4096; cfg->Nvalues[5] = 8192;
    cfg->Nvalues[6] = 16384; cfg->nN = 7;
    cfg->epsabs = 1e-10; cfg->epsrel = 1e-9;
    cfg->refEpsabs = 1e-13; cfg->refEpsrel = 1e-12;
    strcpy(cfg->methods, "trap,simpson,spline");
    cfg->repeats = 3;
    strcpy(cfg->outfile, "data/scan_source.csv");
}

/* trim leading/trailing whitespace in place, returning the trimmed start */
static char *trim(char *s)
{
    while (*s && isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) *--end = '\0';
    return s;
}

/* parse a comma/space separated list of ints into cfg->Nvalues */
static void parse_nvalues(config *cfg, const char *val)
{
    char buf[256];
    strncpy(buf, val, sizeof buf - 1);
    buf[sizeof buf - 1] = '\0';
    cfg->nN = 0;
    for (char *tok = strtok(buf, ", \t"); tok && cfg->nN < MAX_NVALUES;
         tok = strtok(NULL, ", \t"))
    {
        int n = atoi(tok);
        if (n >= 2) cfg->Nvalues[cfg->nN++] = n;
        else fprintf(stderr, "scantest: ignoring invalid N '%s'\n", tok);
    }
}

/* parse a key=value config file; '#' begins a comment. Unknown keys warn. */
static int config_load(config *cfg, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "scantest: cannot open config '%s'\n", path); return -1; }

    char line[512];
    while (fgets(line, sizeof line, f))
    {
        char *hash = strchr(line, '#');
        if (hash) *hash = '\0';
        char *eq = strchr(line, '=');
        if (!eq) continue;                    /* blank/comment-only ok */

        *eq = '\0';
        char *key = trim(line);
        char *val = trim(eq + 1);
        if (!*key) continue;

        if      (!strcmp(key, "a"))           cfg->a = atof(val);
        else if (!strcmp(key, "p"))           cfg->p = atof(val);
        else if (!strcmp(key, "e"))           cfg->e = atof(val);
        else if (!strcmp(key, "x"))           cfg->x = atof(val);
        else if (!strcmp(key, "mMode"))       cfg->mMode = atoi(val);
        else if (!strcmp(key, "nMode"))       cfg->nMode = atoi(val);
        else if (!strcmp(key, "rFieldMin"))   cfg->rFieldMin = atof(val);
        else if (!strcmp(key, "rFieldMax"))   cfg->rFieldMax = atof(val);
        else if (!strcmp(key, "rFieldCount")) cfg->rFieldCount = atoi(val);
        else if (!strcmp(key, "thetaMin"))    cfg->thetaMin = atof(val);
        else if (!strcmp(key, "thetaMax"))    cfg->thetaMax = atof(val);
        else if (!strcmp(key, "thetaCount"))  cfg->thetaCount = atoi(val);
        else if (!strcmp(key, "Nvalues"))     parse_nvalues(cfg, val);
        else if (!strcmp(key, "epsabs"))      cfg->epsabs = atof(val);
        else if (!strcmp(key, "epsrel"))      cfg->epsrel = atof(val);
        else if (!strcmp(key, "refEpsabs"))   cfg->refEpsabs = atof(val);
        else if (!strcmp(key, "refEpsrel"))   cfg->refEpsrel = atof(val);
        else if (!strcmp(key, "methods")) { strncpy(cfg->methods, val, sizeof cfg->methods - 1);
                                            cfg->methods[sizeof cfg->methods - 1] = '\0'; }
        else if (!strcmp(key, "repeats"))     cfg->repeats = atoi(val);
        else if (!strcmp(key, "outfile")) { strncpy(cfg->outfile, val, sizeof cfg->outfile - 1);
                                            cfg->outfile[sizeof cfg->outfile - 1] = '\0'; }
        else fprintf(stderr, "scantest: unknown config key '%s' (ignored)\n", key);
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
    if (!strcmp(name, "minospline")) return INSPECTRE_INTEG_MINO_SPLINE;
    return -1;
}

/* monotonic wall-clock seconds */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* grid value i of count points spanning [lo, hi] inclusive (lo if count<=1) */
static double grid_val(double lo, double hi, int count, int i)
{
    if (count <= 1) return lo;
    return lo + (hi - lo) * (double)i / (double)(count - 1);
}

/* ---- driver ------------------------------------------------------------- */

int main(int argc, char *argv[])
{
    gsl_set_error_handler_off();

    const char *cfgpath = (argc >= 2) ? argv[1] : "test/scan.cfg";
    config cfg;
    config_defaults(&cfg);
    if (config_load(&cfg, cfgpath) != 0) return 1;

    double a = cfg.a, p = cfg.p, e = cfg.e, x = cfg.x;
    int mMode = cfg.mMode, nMode = cfg.nMode;
    int repeats = cfg.repeats > 0 ? cfg.repeats : 1;

    /* largest N drives the sample-buffer allocation (reused across the sweep) */
    int Nmax = 0;
    for (int k = 0; k < cfg.nN; k++) if (cfg.Nvalues[k] > Nmax) Nmax = cfg.Nvalues[k];
    if (cfg.nN == 0 || Nmax < 2) { fprintf(stderr, "scantest: no valid Nvalues\n"); return 1; }

    printf("=== scantest (config: %s) ===\n", cfgpath);
    printf("a=%g p=%g e=%g x=%g  mMode=%d nMode=%d\n", a, p, e, x, mMode, nMode);
    printf("rField: [%g, %g] x%d   theta: [%g, %g] x%d\n",
           cfg.rFieldMin, cfg.rFieldMax, cfg.rFieldCount,
           cfg.thetaMin, cfg.thetaMax, cfg.thetaCount);
    printf("Nvalues:");
    for (int k = 0; k < cfg.nN; k++) printf(" %d", cfg.Nvalues[k]);
    printf("\nmethods = %s  repeats = %d  outfile = %s\n\n",
           cfg.methods, repeats, cfg.outfile);

    int eccentric = e > 0.0 ? 1 : 0;
    int inclined  = fabs(1.0 - x) <= 1e-14 ? 0 : 1;

    /* orbit is independent of the field point: build context + params once */
    korb_params orbpar;
    korb_getparams(eccentric, inclined, a, p, e, x, 1.0e-15, &orbpar);
    struct effsource_equatorial_ctx *ctx = effsource_equatorial_create(1.0, a);

    double omegaPhi = orbpar.wphi, omegaR = orbpar.wr;
    double Vr = orbpar.Vr;
    double Tr = korb_tfromla(Vr, orbpar);

    /* Jacobian self-check: the Mino-time methods rely on dt/dlambda = T_r + T_th.
       Its integral over [0, Vr] must reproduce Tr. Verify before trusting it. */
    {
        const int Mchk = 200000;
        double jint = 0.0, prevJ = 0.0;
        for (int i = 0; i < Mchk; i++) {
            double lam = (double)i / (double)(Mchk - 1) * Vr;
            double J = korb_Tr(korb_psifromla(lam, orbpar), orbpar)
                     + korb_Tth(korb_chifromla(lam, orbpar), orbpar);
            if (i > 0) jint += 0.5 * (prevJ + J) * (Vr / (double)(Mchk - 1));
            prevJ = J;
        }
        double jerr = fabs(jint / Tr - 1.0);
        printf("Jacobian self-check: |int(dt/dlam)/Tr - 1| = %.3e  [%s]\n\n",
               jerr, jerr < 1e-8 ? "PASS" : "FAIL");
    }

    /* sample buffers, sized for the largest N and reused for every grid point */
    double *latT = malloc(Nmax * sizeof(double));   /* t(lambda) tabulation */
    double *latL = malloc(Nmax * sizeof(double));   /* the lambda samples   */
    double *tS = malloc(Nmax * sizeof(double));
    double *fS = malloc(2 * Nmax * sizeof(double));
    double *dS = malloc(8 * Nmax * sizeof(double));
    double *sS = malloc(2 * Nmax * sizeof(double));
    double *fftRe = malloc(Nmax * sizeof(double));  /* FFT all-n amplitude bins */
    double *fftIm = malloc(Nmax * sizeof(double));

    /* parse the fixed-grid method list once */
    char methodlist[256];
    strncpy(methodlist, cfg.methods, sizeof methodlist - 1);
    methodlist[sizeof methodlist - 1] = '\0';

    FILE *out = fopen(cfg.outfile, "w");
    if (!out) { fprintf(stderr, "scantest: cannot open outfile '%s'\n", cfg.outfile); return 1; }
    fprintf(out, "# inspectre source-integration scan. err_* vs high-accuracy QAG reference.\n");
    fprintf(out, "# qag/qagmino rows: N=0. Uniform fixed-grid total cost = build_s + time_s.\n");
    fprintf(out, "# self-contained methods (qagmino, minospline): build_s=0, time_s is total.\n");
    fprintf(out, "rField,theta,N,method,src_re,src_im,err_re,err_im,err_max,build_s,time_s\n");

    /* scratch outputs we don't report */
    double PhiS[2], dPhiS[8];

    for (int ir = 0; ir < cfg.rFieldCount; ir++)
    {
        double rField = grid_val(cfg.rFieldMin, cfg.rFieldMax, cfg.rFieldCount, ir);
        printf("Running data for r = %.10g\n", rField);
        for (int it = 0; it < cfg.thetaCount; it++)
        {
            double theta = grid_val(cfg.thetaMin, cfg.thetaMax, cfg.thetaCount, it);
            struct coordinate xField = { .t = 0.0, .phi = 0.0, .r = rField, .theta = theta };

            /* ---- 1. high-accuracy QAG ground truth ---- */
            double src_ref[2];
            double t_ref = 1e300;
            for (int rep = 0; rep < repeats; rep++) {
                double t0 = now_sec();
                inspectre_integrate_nmode(INSPECTRE_INTEG_QAG, ctx, mMode, nMode,
                        &xField, &orbpar, a, p, e, omegaPhi, omegaR,
                        cfg.refEpsabs, cfg.refEpsrel,
                        NULL, NULL, NULL, NULL, 0, PhiS, dPhiS, src_ref);
                double dt = now_sec() - t0;
                if (dt < t_ref) t_ref = dt;
            }
            fprintf(out, "%.10g,%.10g,%d,%s,% .12e,% .12e,%.3e,%.3e,%.3e,%.3e,%.6e\n",
                    rField, theta, 0, "qag_ref",
                    src_ref[0], src_ref[1], 0.0, 0.0, 0.0, 0.0, t_ref);

            /* ---- 2. adaptive QAG under test ---- */
            double src_qag[2];
            double t_qag = 1e300;
            for (int rep = 0; rep < repeats; rep++) {
                double t0 = now_sec();
                inspectre_integrate_nmode(INSPECTRE_INTEG_QAG, ctx, mMode, nMode,
                        &xField, &orbpar, a, p, e, omegaPhi, omegaR,
                        cfg.epsabs, cfg.epsrel,
                        NULL, NULL, NULL, NULL, 0, PhiS, dPhiS, src_qag);
                double dt = now_sec() - t0;
                if (dt < t_qag) t_qag = dt;
            }
            {
                double er = fabs(src_qag[0] - src_ref[0]);
                double ei = fabs(src_qag[1] - src_ref[1]);
                fprintf(out, "%.10g,%.10g,%d,%s,% .12e,% .12e,%.3e,%.3e,%.3e,%.3e,%.6e\n",
                        rField, theta, 0, "qag",
                        src_qag[0], src_qag[1], er, ei, fmax(er, ei), 0.0, t_qag);
            }

            /* ---- 2b. adaptive QAG in Mino time (no Brent inversion) ---- */
            {
                double src_qm[2];
                double t_qm = 1e300;
                for (int rep = 0; rep < repeats; rep++) {
                    double t0 = now_sec();
                    inspectre_integrate_nmode(INSPECTRE_INTEG_QAG_MINO, ctx, mMode, nMode,
                            &xField, &orbpar, a, p, e, omegaPhi, omegaR,
                            cfg.epsabs, cfg.epsrel,
                            NULL, NULL, NULL, NULL, 0, PhiS, dPhiS, src_qm);
                    double dt = now_sec() - t0;
                    if (dt < t_qm) t_qm = dt;
                }
                double er = fabs(src_qm[0] - src_ref[0]);
                double ei = fabs(src_qm[1] - src_ref[1]);
                fprintf(out, "%.10g,%.10g,%d,%s,% .12e,% .12e,%.3e,%.3e,%.3e,%.3e,%.6e\n",
                        rField, theta, 0, "qagmino",
                        src_qm[0], src_qm[1], er, ei, fmax(er, ei), 0.0, t_qm);
            }

            /* ---- 2c. panel Gauss-Legendre (self-contained, N-independent:
               node count set by the breakpoint geometry, order 16, 40 levels,
               resolved up to |nMode|). Reported N is the node count. ---- */
            if (strstr(cfg.methods, "panelgl"))
            {
                inspectre_panel_nodes ps;
                double src_pg[2] = { 0.0, 0.0 };
                double build_s = 1e300, integ_s = 1e300;
                int nNodes = 0, nBad = 0;
                for (int rep = 0; rep < repeats; rep++) {
                    double t0 = now_sec();
                    inspectre_panel_nodes_build(ctx, mMode, &xField, &orbpar,
                            a, p, e, 16, 40, abs(nMode) > 8 ? abs(nMode) : 8,
                            omegaPhi, omegaR, &ps);
                    double dt = now_sec() - t0;
                    if (dt < build_s) build_s = dt;
                    nNodes = ps.n;
                    nBad   = ps.nNonFinite;
                    t0 = now_sec();
                    inspectre_panel_nodes_integrate(&ps, mMode, nMode,
                            omegaPhi, omegaR, PhiS, dPhiS, src_pg);
                    dt = now_sec() - t0;
                    if (dt < integ_s) integ_s = dt;
                    inspectre_panel_nodes_free(&ps);
                }
                if (nBad)
                    fprintf(stderr, "scantest: panelgl r=%g theta=%g zeroed %d "
                            "non-finite deep-node samples\n", rField, theta, nBad);
                double er = fabs(src_pg[0] - src_ref[0]);
                double ei = fabs(src_pg[1] - src_ref[1]);
                fprintf(out, "%.10g,%.10g,%d,%s,% .12e,% .12e,%.3e,%.3e,%.3e,%.3e,%.6e\n",
                        rField, theta, nNodes, "panelgl",
                        src_pg[0], src_pg[1], er, ei, fmax(er, ei), build_s, integ_s);
            }

            /* ---- 3. fixed-grid sweep ---- */
            for (int k = 0; k < cfg.nN; k++)
            {
                int N = cfg.Nvalues[k];

                /* build uniform-t samples; keep the min build time over repeats */
                double build_s = 1e300;
                for (int rep = 0; rep < repeats; rep++)
                {
                    double t0 = now_sec();
                    for (int i = 0; i < N; i++) {
                        double lam = (double)i / (double)(N - 1) * Vr;
                        latL[i] = lam;
                        latT[i] = korb_tfromla(lam, orbpar);
                    }
                    gsl_interp_accel *latAcc = gsl_interp_accel_alloc();
                    gsl_spline *latSpline = gsl_spline_alloc(gsl_interp_cspline, N);
                    gsl_spline_init(latSpline, latT, latL, N);
                    for (int i = 0; i < N; i++) {
                        double t   = (double)i / (double)(N - 1) * Tr;
                        double lam = gsl_spline_eval(latSpline, t, latAcc);
                        double lPhiS[2], ldPhiS[8], lddPhiS[20], lsrc[2];
                        inspectre_eval_at_lambda(ctx, mMode, &xField, lam, &orbpar,
                                                 a, p, e, lPhiS, ldPhiS, lddPhiS, lsrc);
                        tS[i] = t;
                        fS[2*i] = lPhiS[0]; fS[2*i+1] = lPhiS[1];
                        for (int c = 0; c < 8; c++) dS[8*i+c] = ldPhiS[c];
                        sS[2*i] = lsrc[0]; sS[2*i+1] = lsrc[1];
                    }
                    gsl_spline_free(latSpline);
                    gsl_interp_accel_free(latAcc);
                    double dt = now_sec() - t0;
                    if (dt < build_s) build_s = dt;
                }

                /* strtok mutates its buffer, so work on a fresh copy each N */
                char mlist[256];
                strncpy(mlist, methodlist, sizeof mlist - 1);
                mlist[sizeof mlist - 1] = '\0';
                for (char *tok = strtok(mlist, ", \t"); tok;
                     tok = strtok(NULL, ", \t"))
                {
                    /* FFT: self-contained, returns all n at once; extract nMode */
                    if (!strcmp(tok, "fft")) {
                        if (abs(nMode) > N / 2) {
                            fprintf(stderr, "scantest: fft N=%d cannot resolve nMode=%d "
                                            "(need N>=2|n|)\n", N, nMode);
                            continue;
                        }
                        int bin = ((nMode % N) + N) % N;
                        double integ_s = 1e300;
                        for (int rep = 0; rep < repeats; rep++) {
                            double t0 = now_sec();
                            inspectre_fft_source_nmodes(ctx, mMode, &xField, &orbpar,
                                    a, p, e, omegaPhi, omegaR, N, fftRe, fftIm);
                            double dt = now_sec() - t0;
                            if (dt < integ_s) integ_s = dt;
                        }
                        double er = fabs(fftRe[bin] - src_ref[0]);
                        double ei = fabs(fftIm[bin] - src_ref[1]);
                        fprintf(out, "%.10g,%.10g,%d,%s,% .12e,% .12e,%.3e,%.3e,%.3e,%.3e,%.6e\n",
                                rField, theta, N, "fft",
                                fftRe[bin], fftIm[bin], er, ei, fmax(er, ei), 0.0, integ_s);
                        continue;
                    }

                    /* panelgl is N-independent and reported once above */
                    if (!strcmp(tok, "panelgl")) continue;

                    int mode = method_mode(tok);
                    /* qag-family are N-independent and reported once above */
                    if (mode < 0 || mode == INSPECTRE_INTEG_QAG
                                 || mode == INSPECTRE_INTEG_QAG_MINO) {
                        if (mode < 0)
                            fprintf(stderr, "scantest: unknown method '%s' (skipped)\n", tok);
                        continue;
                    }

                    /* self-contained methods sample internally and ignore the
                       prebuilt uniform samples, so they own no shared build_s. */
                    int selfcontained = (mode == INSPECTRE_INTEG_MINO_SPLINE);

                    double src[2];
                    double integ_s = 1e300;
                    for (int rep = 0; rep < repeats; rep++) {
                        double t0 = now_sec();
                        inspectre_integrate_nmode(mode, ctx, mMode, nMode,
                                &xField, &orbpar, a, p, e, omegaPhi, omegaR,
                                cfg.epsabs, cfg.epsrel,
                                tS, fS, dS, sS, N, PhiS, dPhiS, src);
                        double dt = now_sec() - t0;
                        if (dt < integ_s) integ_s = dt;
                    }
                    double er = fabs(src[0] - src_ref[0]);
                    double ei = fabs(src[1] - src_ref[1]);
                    fprintf(out, "%.10g,%.10g,%d,%s,% .12e,% .12e,%.3e,%.3e,%.3e,%.3e,%.6e\n",
                            rField, theta, N, tok,
                            src[0], src[1], er, ei, fmax(er, ei),
                            selfcontained ? 0.0 : build_s, integ_s);
                }
            }
            fflush(out);
        }
    }

    fclose(out);
    free(latT); free(latL); free(tS); free(fS); free(dS); free(sS);
    free(fftRe); free(fftIm);
    effsource_equatorial_free(ctx);
    korb_freepar(orbpar);

    printf("wrote %s\n", cfg.outfile);
    return 0;
}
