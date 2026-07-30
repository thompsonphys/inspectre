/* Round-trip accuracy test for the inspectre n-mode integrators.
 *
 * The n-mode integrators compute the Fourier amplitude
 *     A_n = (1/Tr) integral_0^Tr exp(i(m wphi + n wr) t) S_m(t) dt.
 * If the integrator is accurate, summing the amplitudes back to the time domain
 *     S_recon(t) = sum_{n=-nMax}^{nMax} A_n exp(-i(m wphi + n wr) t)
 * reproduces the directly-evaluated source S_direct(t) (the same S_m(t) the
 * amplitudes were projected from). The reconstruction error vs nMax is the
 * accuracy signal -- it should fall as nMax grows, plateauing at the
 * integrator's tolerance floor.
 *
 * At each field point (rField, theta) we build the amplitudes once per
 * integrator (qag, qagmino, minospline -- one call per n; fft -- all n in one
 * transform), then for each nMax in the sweep reconstruct on a uniform t-grid
 * and compare against the cached direct source. Summary error (max + relative
 * L2) is written to CSV and printed; an optional per-time-point dump supports
 * notebook plotting of recon vs direct curves.
 *
 * Build: see test/Makefile target `recontest`.
 * Usage: recontest [config-path]   (default: test/recon.cfg; run from repo root
 *        so relative data/ paths resolve, matching scansweep).
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

#define MAX_NMAXVALUES 64

typedef struct {
    double a, p, e, x;
    int    mMode;
    double rFieldMin, rFieldMax;
    int    rFieldCount;
    double thetaMin, thetaMax;
    int    thetaCount;
    int    nMaxValues[MAX_NMAXVALUES];
    int    nNMax;
    double epsabs, epsrel;
    int    nRecon;
    int    minoSamples;
    int    fftN;
    char   methods[256];
    int    dumpTimeseries;
    int    dumpMesh;
    int    dumpSpline;
    int    splineDense;
    char   outfile[256];
    char   tsfile[256];
    char   modesfile[256];
    char   meshfile[256];
    char   splinefile[256];
} config;

static void config_defaults(config *cfg)
{
    cfg->a = 0.5; cfg->p = 10.0; cfg->e = 0.3; cfg->x = 1.0;
    cfg->mMode = 2;
    /* default grid sits OUTSIDE the radial range [peri, apo] so the source is
       smooth, the spectrum decays, and QAG stays cheap (see test/recon.cfg). */
    cfg->rFieldMin = 16.0; cfg->rFieldMax = 24.0; cfg->rFieldCount = 3;
    cfg->thetaMin = M_PI_2; cfg->thetaMax = M_PI_2; cfg->thetaCount = 1;
    cfg->nMaxValues[0] = 4;  cfg->nMaxValues[1] = 8;
    cfg->nMaxValues[2] = 16; cfg->nMaxValues[3] = 32; cfg->nNMax = 4;
    cfg->epsabs = 1e-12; cfg->epsrel = 1e-11;
    cfg->nRecon = 512;
    cfg->minoSamples = 4096;
    cfg->fftN = 4096;
    strcpy(cfg->methods, "qag,qagmino,minospline,fft");
    cfg->dumpTimeseries = 0;
    cfg->dumpMesh = 0;
    cfg->dumpSpline = 0;
    cfg->splineDense = 4096;
    strcpy(cfg->outfile, "data/recon.csv");
    strcpy(cfg->tsfile, "data/recon_timeseries.csv");
    strcpy(cfg->modesfile, "data/recon_modes.csv");
    strcpy(cfg->meshfile, "data/recon_mesh.csv");
    strcpy(cfg->splinefile, "data/recon_spline.csv");
}

/* trim leading/trailing whitespace in place, returning the trimmed start */
static char *trim(char *s)
{
    while (*s && isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) *--end = '\0';
    return s;
}

/* parse a comma/space separated list of ints into cfg->nMaxValues */
static void parse_nmaxvalues(config *cfg, const char *val)
{
    char buf[256];
    strncpy(buf, val, sizeof buf - 1);
    buf[sizeof buf - 1] = '\0';
    cfg->nNMax = 0;
    for (char *tok = strtok(buf, ", \t"); tok && cfg->nNMax < MAX_NMAXVALUES;
         tok = strtok(NULL, ", \t"))
    {
        int n = atoi(tok);
        if (n >= 1) cfg->nMaxValues[cfg->nNMax++] = n;
        else fprintf(stderr, "recontest: ignoring invalid nMax '%s'\n", tok);
    }
}

/* parse a key=value config file; '#' begins a comment. Unknown keys warn. */
static int config_load(config *cfg, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "recontest: cannot open config '%s'\n", path); return -1; }

    char line[512];
    while (fgets(line, sizeof line, f))
    {
        char *hash = strchr(line, '#');
        if (hash) *hash = '\0';
        char *eq = strchr(line, '=');
        if (!eq) continue;

        *eq = '\0';
        char *key = trim(line);
        char *val = trim(eq + 1);
        if (!*key) continue;

        if      (!strcmp(key, "a"))             cfg->a = atof(val);
        else if (!strcmp(key, "p"))             cfg->p = atof(val);
        else if (!strcmp(key, "e"))             cfg->e = atof(val);
        else if (!strcmp(key, "x"))             cfg->x = atof(val);
        else if (!strcmp(key, "mMode"))         cfg->mMode = atoi(val);
        else if (!strcmp(key, "rFieldMin"))     cfg->rFieldMin = atof(val);
        else if (!strcmp(key, "rFieldMax"))     cfg->rFieldMax = atof(val);
        else if (!strcmp(key, "rFieldCount"))   cfg->rFieldCount = atoi(val);
        else if (!strcmp(key, "thetaMin"))      cfg->thetaMin = atof(val);
        else if (!strcmp(key, "thetaMax"))      cfg->thetaMax = atof(val);
        else if (!strcmp(key, "thetaCount"))    cfg->thetaCount = atoi(val);
        else if (!strcmp(key, "nMaxValues"))    parse_nmaxvalues(cfg, val);
        else if (!strcmp(key, "epsabs"))        cfg->epsabs = atof(val);
        else if (!strcmp(key, "epsrel"))        cfg->epsrel = atof(val);
        else if (!strcmp(key, "nRecon"))        cfg->nRecon = atoi(val);
        else if (!strcmp(key, "minoSamples"))   cfg->minoSamples = atoi(val);
        else if (!strcmp(key, "fftN"))          cfg->fftN = atoi(val);
        else if (!strcmp(key, "dumpTimeseries")) cfg->dumpTimeseries = atoi(val);
        else if (!strcmp(key, "dumpMesh"))      cfg->dumpMesh = atoi(val);
        else if (!strcmp(key, "dumpSpline"))    cfg->dumpSpline = atoi(val);
        else if (!strcmp(key, "splineDense"))   cfg->splineDense = atoi(val);
        else if (!strcmp(key, "methods")) { strncpy(cfg->methods, val, sizeof cfg->methods - 1);
                                            cfg->methods[sizeof cfg->methods - 1] = '\0'; }
        else if (!strcmp(key, "outfile")) { strncpy(cfg->outfile, val, sizeof cfg->outfile - 1);
                                            cfg->outfile[sizeof cfg->outfile - 1] = '\0'; }
        else if (!strcmp(key, "tsfile")) { strncpy(cfg->tsfile, val, sizeof cfg->tsfile - 1);
                                           cfg->tsfile[sizeof cfg->tsfile - 1] = '\0'; }
        else if (!strcmp(key, "modesfile")) { strncpy(cfg->modesfile, val, sizeof cfg->modesfile - 1);
                                              cfg->modesfile[sizeof cfg->modesfile - 1] = '\0'; }
        else if (!strcmp(key, "meshfile")) { strncpy(cfg->meshfile, val, sizeof cfg->meshfile - 1);
                                             cfg->meshfile[sizeof cfg->meshfile - 1] = '\0'; }
        else if (!strcmp(key, "splinefile")) { strncpy(cfg->splinefile, val, sizeof cfg->splinefile - 1);
                                               cfg->splinefile[sizeof cfg->splinefile - 1] = '\0'; }
        else fprintf(stderr, "recontest: unknown config key '%s' (ignored)\n", key);
    }
    fclose(f);
    return 0;
}

/* grid value i of count points spanning [lo, hi] inclusive (lo if count<=1) */
static double grid_val(double lo, double hi, int count, int i)
{
    if (count <= 1) return lo;
    return lo + (hi - lo) * (double)i / (double)(count - 1);
}

/* recognized integrator names for this harness */
static int is_known_method(const char *name)
{
    return !strcmp(name, "qag") || !strcmp(name, "qagmino")
        || !strcmp(name, "minospline") || !strcmp(name, "fft")
        || !strcmp(name, "trapezoid")
        || !strcmp(name, "minotrap") || !strcmp(name, "minosimpson");
}

/* ---- driver ------------------------------------------------------------- */

int main(int argc, char *argv[])
{
    gsl_set_error_handler_off();

    const char *cfgpath = (argc >= 2) ? argv[1] : "test/recon.cfg";
    config cfg;
    config_defaults(&cfg);
    if (config_load(&cfg, cfgpath) != 0) return 1;

    double a = cfg.a, p = cfg.p, e = cfg.e, x = cfg.x;
    int mMode = cfg.mMode;

    if (cfg.nNMax == 0) { fprintf(stderr, "recontest: no valid nMaxValues\n"); return 1; }
    if (cfg.nRecon < 2) { fprintf(stderr, "recontest: nRecon must be >= 2\n"); return 1; }

    /* largest nMax drives the amplitude-buffer size and the fft resolution */
    int nMaxMax = 0;
    for (int k = 0; k < cfg.nNMax; k++)
        if (cfg.nMaxValues[k] > nMaxMax) nMaxMax = cfg.nMaxValues[k];

    /* fft needs N >= 2*nMaxMax to resolve every requested mode */
    int fftN = cfg.fftN;
    if (fftN < 2 * nMaxMax + 1) {
        fprintf(stderr, "recontest: fftN=%d too small for nMaxMax=%d; bumping to %d\n",
                fftN, nMaxMax, 2 * nMaxMax + 2);
        fftN = 2 * nMaxMax + 2;
    }

    printf("=== recontest (config: %s) ===\n", cfgpath);
    printf("a=%g p=%g e=%g x=%g  mMode=%d\n", a, p, e, x, mMode);
    printf("rField: [%g, %g] x%d   theta: [%g, %g] x%d\n",
           cfg.rFieldMin, cfg.rFieldMax, cfg.rFieldCount,
           cfg.thetaMin, cfg.thetaMax, cfg.thetaCount);
    printf("nMaxValues:");
    for (int k = 0; k < cfg.nNMax; k++) printf(" %d", cfg.nMaxValues[k]);
    printf("\nmethods = %s  nRecon = %d  minoSamples = %d  fftN = %d\n",
           cfg.methods, cfg.nRecon, cfg.minoSamples, fftN);
    printf("epsabs = %g  epsrel = %g  outfile = %s\n\n",
           cfg.epsabs, cfg.epsrel, cfg.outfile);

    int eccentric = e > 0.0 ? 1 : 0;
    int inclined  = fabs(1.0 - x) <= 1e-14 ? 0 : 1;

    /* orbit is independent of the field point: build context + params once */
    korb_params orbpar;
    korb_getparams(eccentric, inclined, a, p, e, x, 1.0e-15, &orbpar);
    inspectre_orbit_circular_fix(&orbpar);
    struct effsource_equatorial_ctx *ctx = effsource_equatorial_create(1.0, a);

    double omegaPhi = orbpar.wphi, omegaR = orbpar.wr;
    double Vr = orbpar.Vr;
    double Tr = korb_tfromla(Vr, orbpar);

    /* parse the method list once; record which integrators are requested */
    int do_qag = 0, do_qagmino = 0, do_minospline = 0, do_fft = 0, do_trapezoid = 0;
    int do_minotrap = 0, do_minosimpson = 0;
    {
        char mlist[256];
        strncpy(mlist, cfg.methods, sizeof mlist - 1);
        mlist[sizeof mlist - 1] = '\0';
        for (char *tok = strtok(mlist, ", \t"); tok; tok = strtok(NULL, ", \t")) {
            if      (!strcmp(tok, "qag"))         do_qag = 1;
            else if (!strcmp(tok, "qagmino"))     do_qagmino = 1;
            else if (!strcmp(tok, "minospline"))  do_minospline = 1;
            else if (!strcmp(tok, "fft"))         do_fft = 1;
            else if (!strcmp(tok, "trapezoid"))   do_trapezoid = 1;
            else if (!strcmp(tok, "minotrap"))    do_minotrap = 1;
            else if (!strcmp(tok, "minosimpson")) do_minosimpson = 1;
            else fprintf(stderr, "recontest: unknown method '%s' (skipped)\n", tok);
        }
    }
    (void)is_known_method;

    /* amplitude tables: index n in [-nMaxMax, nMaxMax] -> offset n + nMaxMax */
    int nAmp = 2 * nMaxMax + 1;
    double *qagRe = malloc(nAmp * sizeof(double)), *qagIm = malloc(nAmp * sizeof(double));
    double *qmRe  = malloc(nAmp * sizeof(double)), *qmIm  = malloc(nAmp * sizeof(double));
    double *msRe  = malloc(nAmp * sizeof(double)), *msIm  = malloc(nAmp * sizeof(double));
    double *fftAmpRe = malloc(nAmp * sizeof(double)), *fftAmpIm = malloc(nAmp * sizeof(double));
    double *fftRe = malloc(fftN * sizeof(double)), *fftIm = malloc(fftN * sizeof(double));
    double *trapRe = malloc(nAmp * sizeof(double)), *trapIm = malloc(nAmp * sizeof(double));
    double *mtRe = malloc(nAmp * sizeof(double)), *mtIm = malloc(nAmp * sizeof(double));
    double *msimRe = malloc(nAmp * sizeof(double)), *msimIm = malloc(nAmp * sizeof(double));

    /* reconstruction time grid + cached direct source */
    int nRecon = cfg.nRecon;
    double *tGrid = malloc(nRecon * sizeof(double));
    double *dirRe = malloc(nRecon * sizeof(double)), *dirIm = malloc(nRecon * sizeof(double));
    /* lambda(t) spline reused to evaluate the direct source on the t-grid */
    double *latT = malloc(nRecon * sizeof(double)), *latL = malloc(nRecon * sizeof(double));

    /* trapezoid samples: the same uniform-t source evals as the direct grid,
       stored in the interleaved sample layout, on a CLOSED periodic grid of
       nRecon+1 points (last = first's periodic image at t=Tr) so the trapezoid
       rule closes the wrap interval. No extra source evals: points 0..nRecon-1
       are exactly the direct-source grid. */
    int nTrap = nRecon + 1;
    double *tsT = malloc(nTrap * sizeof(double));
    double *tsF = malloc(2 * nTrap * sizeof(double));
    double *tsD = malloc(8 * nTrap * sizeof(double));
    double *tsS = malloc(2 * nTrap * sizeof(double));

    FILE *out = fopen(cfg.outfile, "w");
    if (!out) { fprintf(stderr, "recontest: cannot open outfile '%s'\n", cfg.outfile); return 1; }
    fprintf(out, "# inspectre n-mode round-trip reconstruction accuracy.\n");
    fprintf(out, "# err_* : L-inf/L2 of (S_recon - S_direct) over the t-grid; "
                 "Sdirect_l2 normalizes for a relative error.\n");
    fprintf(out, "rField,theta,method,nMax,err_max,err_l2,Sdirect_l2\n");

    FILE *ts = NULL;
    FILE *modes = NULL;
    if (cfg.dumpTimeseries) {
        ts = fopen(cfg.tsfile, "w");
        if (!ts) { fprintf(stderr, "recontest: cannot open tsfile '%s'\n", cfg.tsfile); return 1; }
        fprintf(ts, "rField,theta,method,nMax,t,recon_re,recon_im,direct_re,direct_im\n");

        modes = fopen(cfg.modesfile, "w");
        if (!modes) { fprintf(stderr, "recontest: cannot open modesfile '%s'\n", cfg.modesfile); return 1; }
        fprintf(modes, "rField,theta,method,n,A_re,A_im\n");
    }

    /* diagnostics for the minospline graded mesh: node placement (mesh) and the
       cubic-spline interpolant of the raw, n-independent source (spline). Both
       need the minospline samples, so force the build below when requested. */
    int wantMesh   = cfg.dumpMesh;
    int wantSpline = cfg.dumpSpline;
    int buildMino  = do_minospline || do_minotrap || do_minosimpson
                     || wantMesh || wantSpline;
    FILE *mesh = NULL, *spline = NULL;
    if (wantMesh) {
        mesh = fopen(cfg.meshfile, "w");
        if (!mesh) { fprintf(stderr, "recontest: cannot open meshfile '%s'\n", cfg.meshfile); return 1; }
        fprintf(mesh, "# minospline graded-mesh node geometry + raw (n-independent) source.\n");
        fprintf(mesh, "rField,theta,node_i,lambda,t,J,src_re,src_im,phiS_re,phiS_im\n");
    }
    if (wantSpline) {
        spline = fopen(cfg.splinefile, "w");
        if (!spline) { fprintf(stderr, "recontest: cannot open splinefile '%s'\n", cfg.splinefile); return 1; }
        fprintf(spline, "# dense eval of the cubic spline through the graded-mesh raw source "
                        "(compare vs the dense direct source in the timeseries dump).\n");
        fprintf(spline, "rField,theta,lambda,t,spline_src_re,spline_src_im,is_node\n");
    }

    double PhiS[2], dPhiS[8]; /* scratch outputs we don't report */

    for (int ir = 0; ir < cfg.rFieldCount; ir++)
    {
        double rField = grid_val(cfg.rFieldMin, cfg.rFieldMax, cfg.rFieldCount, ir);
        printf("Running data for r = %.10g\n", rField);
        for (int it = 0; it < cfg.thetaCount; it++)
        {
            double theta = grid_val(cfg.thetaMin, cfg.thetaMax, cfg.thetaCount, it);
            struct coordinate xField = { .t = 0.0, .phi = 0.0, .r = rField, .theta = theta };

            /* ---- direct time-domain source on the reconstruction grid ----
               Sample at uniform t in [0, Tr); build a lambda(t) spline (t(lambda)
               is the natural parametrization) and evaluate the source at lambda. */
            for (int i = 0; i < nRecon; i++) {
                double lam = (double)i / (double)(nRecon - 1) * Vr;
                latL[i] = lam;
                latT[i] = korb_tfromla(lam, orbpar);
            }
            gsl_interp_accel *latAcc = gsl_interp_accel_alloc();
            gsl_spline *latSpline = gsl_spline_alloc(gsl_interp_cspline, nRecon);
            gsl_spline_init(latSpline, latT, latL, nRecon);
            double Sdir_l2 = 0.0;
            for (int j = 0; j < nRecon; j++) {
                double t   = (double)j / (double)nRecon * Tr;   /* [0, Tr), periodic */
                double lam = gsl_spline_eval(latSpline, t, latAcc);
                double lPhiS[2], ldPhiS[8], lddPhiS[20], lsrc[2];
                inspectre_eval_at_lambda(ctx, mMode, &xField, lam, &orbpar,
                                         a, p, e, lPhiS, ldPhiS, lddPhiS, lsrc);
                tGrid[j] = t;
                dirRe[j] = lsrc[0];
                dirIm[j] = lsrc[1];
                Sdir_l2 += lsrc[0] * lsrc[0] + lsrc[1] * lsrc[1];
                /* stash all 12 components for the trapezoid sample-based path */
                tsT[j] = t;
                tsF[2*j] = lPhiS[0]; tsF[2*j+1] = lPhiS[1];
                for (int c = 0; c < 8; c++) tsD[8*j+c] = ldPhiS[c];
                tsS[2*j] = lsrc[0]; tsS[2*j+1] = lsrc[1];
            }
            Sdir_l2 = sqrt(Sdir_l2 / (double)nRecon);
            gsl_spline_free(latSpline);
            gsl_interp_accel_free(latAcc);
            /* closed periodic wrap point at t = Tr: copy the t=0 sample */
            tsT[nRecon] = Tr;
            tsF[2*nRecon] = tsF[0]; tsF[2*nRecon+1] = tsF[1];
            for (int c = 0; c < 8; c++) tsD[8*nRecon+c] = tsD[c];
            tsS[2*nRecon] = tsS[0]; tsS[2*nRecon+1] = tsS[1];

            /* ---- amplitude tables up to nMaxMax (once per integrator) ----
               minospline: build the n-independent graded mesh + raw source once,
               then frequency-shift + spline-integrate per n (no per-n re-eval). */
            inspectre_mino_samples ms;
            if (buildMino)
                inspectre_mino_samples_build(ctx, mMode, &xField, &orbpar,
                                             a, p, e, cfg.minoSamples, &ms);

            long qagLimitHits = 0, qmLimitHits = 0;
            inspectre_qag_limit_reset();
            if (do_qag || do_qagmino || do_minospline || do_trapezoid
                || do_minotrap || do_minosimpson) {
                for (int n = -nMaxMax; n <= nMaxMax; n++) {
                    int idx = n + nMaxMax;
                    double src[2];
                    if (do_qag) {
                        long c0 = inspectre_qag_limit_count();
                        inspectre_integrate_nmode(INSPECTRE_INTEG_QAG, ctx, mMode, n,
                                &xField, &orbpar, a, p, e, omegaPhi, omegaR,
                                cfg.epsabs, cfg.epsrel, NULL, NULL, NULL, NULL, 0,
                                PhiS, dPhiS, src);
                        qagRe[idx] = src[0]; qagIm[idx] = src[1];
                        if (inspectre_qag_limit_count() > c0) qagLimitHits++;
                    }
                    if (do_qagmino) {
                        long c0 = inspectre_qag_limit_count();
                        inspectre_integrate_nmode(INSPECTRE_INTEG_QAG_MINO, ctx, mMode, n,
                                &xField, &orbpar, a, p, e, omegaPhi, omegaR,
                                cfg.epsabs, cfg.epsrel, NULL, NULL, NULL, NULL, 0,
                                PhiS, dPhiS, src);
                        qmRe[idx] = src[0]; qmIm[idx] = src[1];
                        if (inspectre_qag_limit_count() > c0) qmLimitHits++;
                    }
                    if (do_minospline) {
                        inspectre_mino_samples_integrate(&ms, mMode, n,
                                omegaPhi, omegaR, PhiS, dPhiS, src);
                        msRe[idx] = src[0]; msIm[idx] = src[1];
                    }
                    if (do_minotrap) {
                        inspectre_mino_samples_integrate_rule(&ms, mMode, n,
                                omegaPhi, omegaR, INSPECTRE_INTEG_TIMESERIES,
                                PhiS, dPhiS, src);
                        mtRe[idx] = src[0]; mtIm[idx] = src[1];
                    }
                    if (do_minosimpson) {
                        inspectre_mino_samples_integrate_rule(&ms, mMode, n,
                                omegaPhi, omegaR, INSPECTRE_INTEG_SIMPSON,
                                PhiS, dPhiS, src);
                        msimRe[idx] = src[0]; msimIm[idx] = src[1];
                    }
                    if (do_trapezoid) {
                        inspectre_integrate_nmode(INSPECTRE_INTEG_TIMESERIES, ctx, mMode, n,
                                &xField, &orbpar, a, p, e, omegaPhi, omegaR,
                                cfg.epsabs, cfg.epsrel, tsT, tsF, tsD, tsS, nTrap,
                                PhiS, dPhiS, src);
                        trapRe[idx] = src[0]; trapIm[idx] = src[1];
                    }
                }
            }
            if (qagLimitHits)
                printf("  ** qag: %ld/%d modes hit the QAG iteration limit at "
                       "r=%.6g th=%.6g (results may be inaccurate)\n",
                       qagLimitHits, nAmp, rField, theta);
            if (qmLimitHits)
                printf("  ** qagmino: %ld/%d modes hit the QAG iteration limit at "
                       "r=%.6g th=%.6g (results may be inaccurate)\n",
                       qmLimitHits, nAmp, rField, theta);
            if (do_fft) {
                inspectre_fft_source_nmodes(ctx, mMode, &xField, &orbpar,
                        a, p, e, omegaPhi, omegaR, fftN, fftRe, fftIm);
                for (int n = -nMaxMax; n <= nMaxMax; n++) {
                    int bin = ((n % fftN) + fftN) % fftN;
                    fftAmpRe[n + nMaxMax] = fftRe[bin];
                    fftAmpIm[n + nMaxMax] = fftIm[bin];
                }
            }

            /* ---- reconstruct + compare for each method and nMax ---- */
            struct { const char *name; int on; double *re, *im; } methods[] = {
                { "qag",        do_qag,        qagRe, qagIm },
                { "qagmino",    do_qagmino,    qmRe,  qmIm  },
                { "minospline", do_minospline, msRe,  msIm  },
                { "fft",        do_fft,        fftAmpRe, fftAmpIm },
                { "trapezoid",  do_trapezoid,  trapRe, trapIm },
                { "minotrap",   do_minotrap,   mtRe,  mtIm  },
                { "minosimpson",do_minosimpson,msimRe,msimIm},
            };
            const int nMethods = (int)(sizeof methods / sizeof methods[0]);

            for (int m = 0; m < nMethods; m++) {
                if (!methods[m].on) continue;
                double *Are = methods[m].re, *Aim = methods[m].im;
                for (int k = 0; k < cfg.nNMax; k++) {
                    int nMax = cfg.nMaxValues[k];
                    double err_max = 0.0, err_l2 = 0.0;
                    for (int j = 0; j < nRecon; j++) {
                        double t = tGrid[j];
                        /* S_recon(t) = sum_n A_n exp(-i(m wphi + n wr) t) */
                        double sre = 0.0, sim = 0.0;
                        for (int n = -nMax; n <= nMax; n++) {
                            int idx = n + nMaxMax;
                            double w = (double)mMode * omegaPhi + (double)n * omegaR;
                            double c = cos(w * t), s = sin(w * t);
                            /* (Are + i Aim)(c - i s) */
                            sre += Are[idx] * c + Aim[idx] * s;
                            sim += Aim[idx] * c - Are[idx] * s;
                        }
                        double er = sre - dirRe[j];
                        double ei = sim - dirIm[j];
                        double mag = sqrt(er * er + ei * ei);
                        if (mag > err_max) err_max = mag;
                        err_l2 += er * er + ei * ei;

                        if (ts)
                            fprintf(ts, "%.10g,%.10g,%s,%d,%.10g,% .12e,% .12e,% .12e,% .12e\n",
                                    rField, theta, methods[m].name, nMax,
                                    t, sre, sim, dirRe[j], dirIm[j]);
                    }
                    err_l2 = sqrt(err_l2 / (double)nRecon);

                    fprintf(out, "%.10g,%.10g,%s,%d,%.6e,%.6e,%.6e\n",
                            rField, theta, methods[m].name, nMax,
                            err_max, err_l2, Sdir_l2);
                    printf("  r=%.6g th=%.6g %-10s nMax=%-4d  max=%.3e  "
                           "relL2=%.3e\n",
                           rField, theta, methods[m].name, nMax, err_max,
                           Sdir_l2 > 0.0 ? err_l2 / Sdir_l2 : err_l2);
                }
            }

            /* ---- dump the n-mode amplitudes A_n for the max nMax (=nMaxMax),
               for every enabled method (gated by dumpTimeseries) ---- */
            if (modes) {
                for (int m = 0; m < nMethods; m++) {
                    if (!methods[m].on) continue;
                    for (int n = -nMaxMax; n <= nMaxMax; n++) {
                        int idx = n + nMaxMax;
                        fprintf(modes, "%.10g,%.10g,%s,%d,% .12e,% .12e\n",
                                rField, theta, methods[m].name, n,
                                methods[m].re[idx], methods[m].im[idx]);
                    }
                }
                fflush(modes);
            }

            /* ---- minospline mesh + spline diagnostics (n-independent) ---- */
            if (mesh) {
                for (int i = 0; i < ms.n; i++)
                    fprintf(mesh, "%.10g,%.10g,%d,% .12e,% .12e,% .12e,% .12e,% .12e,% .12e,% .12e\n",
                            rField, theta, i, ms.lam[i], ms.t[i], ms.J[i],
                            ms.raw[10 * ms.n + i], ms.raw[11 * ms.n + i],
                            ms.raw[0 * ms.n + i],  ms.raw[1 * ms.n + i]);
                fflush(mesh);
            }
            if (spline) {
                /* cubic spline through the raw source nodes (same nodes fed to the
                   quadrature for the src component, minus per-n shift + Jacobian),
                   evaluated on a dense uniform-lambda grid for a true-vs-interpolant
                   comparison against the dense direct source. */
                gsl_interp_accel *sacc = gsl_interp_accel_alloc();
                gsl_spline *spRe = gsl_spline_alloc(gsl_interp_cspline, ms.n);
                gsl_spline *spIm = gsl_spline_alloc(gsl_interp_cspline, ms.n);
                gsl_spline_init(spRe, ms.lam, &ms.raw[10 * ms.n], ms.n);
                gsl_spline_init(spIm, ms.lam, &ms.raw[11 * ms.n], ms.n);
                int D = cfg.splineDense < 2 ? 2 : cfg.splineDense;
                int ni = 0;  /* index into the node list, to flag knots */
                for (int j = 0; j < D; j++) {
                    double lam = (double)j / (double)(D - 1) * ms.Vr;
                    double sre = gsl_spline_eval(spRe, lam, sacc);
                    double sim = gsl_spline_eval(spIm, lam, sacc);
                    int is_node = 0;
                    while (ni < ms.n && ms.lam[ni] < lam) ni++;
                    if (ni < ms.n && ms.lam[ni] == lam) is_node = 1;
                    fprintf(spline, "%.10g,%.10g,% .12e,% .12e,% .12e,% .12e,%d\n",
                            rField, theta, lam, korb_tfromla(lam, orbpar),
                            sre, sim, is_node);
                }
                gsl_spline_free(spRe);
                gsl_spline_free(spIm);
                gsl_interp_accel_free(sacc);
                fflush(spline);
            }

            if (buildMino) inspectre_mino_samples_free(&ms);

            fflush(out);
            if (ts) fflush(ts);
        }
    }

    fclose(out);
    if (ts) fclose(ts);
    if (modes) fclose(modes);
    if (mesh) fclose(mesh);
    if (spline) fclose(spline);
    free(qagRe); free(qagIm); free(qmRe); free(qmIm);
    free(msRe); free(msIm); free(fftAmpRe); free(fftAmpIm);
    free(fftRe); free(fftIm); free(trapRe); free(trapIm);
    free(mtRe); free(mtIm); free(msimRe); free(msimIm);
    free(tsT); free(tsF); free(tsD); free(tsS);
    free(tGrid); free(dirRe); free(dirIm); free(latT); free(latL);
    effsource_equatorial_free(ctx);
    korb_freepar(orbpar);

    printf("\nwrote %s%s%s%s\n", cfg.outfile,
           cfg.dumpTimeseries ? " (+ timeseries + modes)" : "",
           cfg.dumpMesh ? " (+ mesh)" : "",
           cfg.dumpSpline ? " (+ spline)" : "");
    return 0;
}
