#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_odeiv2.h>

#include "../include/inspectre.h"

double getDistance(double t, double rField, double thetaField, double rParticle){
    double rSqr = rParticle*rParticle + rField * rField;
    double angDist = 2.0 * rParticle * rField * sin(thetaField);
    return sqrt(rSqr - angDist);
}

int main(int argc, char *argv[])
{

    /* initialization */
    int numTimePts, numPeriods, mMode, nMode, eccentric, inclined;
    double PhiS[2], dPhiS[8], ddPhiS[20], src[2];
    double nModePhiS[2], nModeDPhiS[8], nModesrc[2];
    double lam, psi, t, r_p, phi_p, ur, a, p, e, x, err;
    double MinoPeriodR, deltaLambda, omegaPhi, omegaR;
    double rField, thetaField, rMin, rMax;
    double instantaneousDistance;
    struct coordinate xField;
    korb_params orbpar;
    struct effsource_equatorial_ctx *ctx;

    /* Disable the GSL error handler so that it doesn't abort due to roundoff errors */
    gsl_set_error_handler_off();

    if (argc < 2)
    {
        printf("please input parameters in order: a, p, e, x, number of time points, number of periods, m-mode and n-mode\n");
        return 0;
    }

    /* set orbit parameters from inputs */
    a = strtod(argv[1], NULL);
    p = strtod(argv[2], NULL);
    e = strtod(argv[3], NULL);
    x = strtod(argv[4], NULL);

    rMin = p / (1.0 + e);
    rMax = p / (1.0 - e);

    /* set timesteps from input */
    numTimePts = (int)strtod(argv[5], NULL);
    numPeriods = (int)strtod(argv[6], NULL);

    /* set m-mode and n-mode from input */
    mMode = (int)strtod(argv[7], NULL);
    nMode = (int)strtod(argv[8], NULL);

    /* set spatial sampling from input */
    rField = strtod(argv[9], NULL);
    thetaField = strtod(argv[10], NULL);

    eccentric = e > 0.0 ? 1 : 0;
    inclined = fabs(1.0 - x) <= 1e-14 ? 0 : 1;

    err = 1.0e-15;

    /* All orbital characteristics are calculated and stored inside "orbpar" */
    korb_getparams(eccentric, inclined, a, p, e, x, err, &orbpar);
    inspectre_orbit_circular_fix(&orbpar);

    /* effective-source context (M = 1, spin a) */
    ctx = effsource_equatorial_create(1.0, a);

    /* Get the Mino time radial period and sample uniformly over numPeriods
       full periods for numTimePts */
    MinoPeriodR = (double)numPeriods * orbpar.Vr;
    deltaLambda = MinoPeriodR / ((double)numTimePts - 1.0);

    /* Get frequencies for n-mode decomposition */
    omegaPhi = orbpar.wphi;
    omegaR = orbpar.wr;

    xField.t = 0.0;
    xField.phi = 0.0;
    xField.r = rField;
    xField.theta = thetaField;

    /* setting up output files */
    FILE *trajFile, *fieldFile, *nfieldFile;

    trajFile = fopen("data/trajectory_adaptive.dat", "w");
    fprintf(trajFile, "# lambda\ttime\tradius\tphi\tfourVel\n");
    fieldFile = fopen("data/field_adaptive.dat", "w");
    fprintf(fieldFile, "# time\tfieldRe\tfieldIm\tsrcRe\tsrcIm\tdrFieldRe\tdrFieldIm\tdthFieldRe\tdthFieldIm\n");
    nfieldFile = fopen("data/nmode_field_adaptive.dat", "w");
    fprintf(nfieldFile, "# time\tfieldRe\tfieldIm\tsrcRe\tsrcIm\tdrFieldRe\tdrFieldIm\tdthFieldRe\tdthFieldIm\n");

    // looping over Mino time steps
    // for (int i = 0; i < numTimePts; i++)
    lam = 0.0;
    // deltaLambda = 1;
    printf("deltaLambda = %.15g\n", deltaLambda);
    while ( lam < MinoPeriodR)
    {
        // find BL coordinate positions
        psi = korb_psifromla(lam, orbpar);
        t = korb_tfromla(lam, orbpar);
        r_p = korb_rfrompsi(psi, orbpar);
        phi_p = korb_phifromla(lam, orbpar);

        // set four-velocity (for trajectory output)
        ur = fourVel(psi, a, p, e, orbpar.E);

        instantaneousDistance = getDistance(t, xField.r, xField.theta, r_p) / xField.r;

        fprintf(trajFile, "%.15f\t%.15f\t%.15f\t%.15f\t%.15f\n",
                lam, t, r_p, phi_p, ur);

        // effective source + puncture at this field point and Mino time
        inspectre_eval_at_lambda(ctx, mMode, &xField, lam, &orbpar, a, p, e,
                                 PhiS, dPhiS, ddPhiS, src);

        fprintf(fieldFile, "%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g\n", t, PhiS[0],PhiS[1],src[0],src[1],dPhiS[2],dPhiS[3],dPhiS[4],dPhiS[5]);

        /* get n-mode integrand timeseries data */
        generateNModeIntegrands(t, omegaPhi, omegaR, mMode, nMode, PhiS, dPhiS, src, nModePhiS, nModeDPhiS, nModesrc);

        fprintf(nfieldFile, "%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g\n", t, nModePhiS[0],nModePhiS[1],nModesrc[0],nModesrc[1],nModeDPhiS[2],nModeDPhiS[3],nModeDPhiS[4],nModeDPhiS[5]);

        double rescale = sqrt(instantaneousDistance*instantaneousDistance*instantaneousDistance);
        lam += deltaLambda * rescale;
    }

    fclose(trajFile);
    fclose(fieldFile);
    fclose(nfieldFile);

    effsource_equatorial_free(ctx);
    korb_freepar(orbpar);

    return 0;
}
