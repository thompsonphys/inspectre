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

int main(int argc, char *argv[])
{

    /* initialization */
    int numTimePts, numPeriods, mMode, nMode, eccentric, inclined;
    double PhiS[2], dPhiS[8], ddPhiS[20], src[2];
    double nModePhiS[2], nModeDPhiS[8], nModesrc[2];
    double lam, psi, t, r_p, phi_p, ur, a, p, e, x, err;
    double MinoPeriodR, deltaLambda, omegaPhi, omegaR;
    struct coordinate xParticle;
    struct coordinate xField;
    korb_params orbpar;

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

    /* set timesteps from input */
    numTimePts = (int)strtod(argv[7], NULL);
    numPeriods = (int)strtod(argv[8], NULL);

    /* set m-mode and n-mode from input */
    mMode = (int)strtod(argv[9], NULL);
    nMode = (int)strtod(argv[10], NULL);

    /* set field point from inputs */
    xField.r = strtod(argv[5], NULL);
    xField.theta = strtod(argv[6], NULL);
    xField.t = 0.0;
    xField.phi = 0.0;

    eccentric = e > 0.0 ? 1 : 0;
    inclined = fabs(1.0 - x) <= 1e-14 ? 0 : 1;

    err = 1.0e-15;

    /* All orbital characteristics are calculated and stored inside "orbpar" */
    korb_getparams(eccentric, inclined, a, p, e, x, err, &orbpar);
    inspectre_orbit_circular_fix(&orbpar);

    /* Get the Mino time radial period and sample uniformly over numPeriods
       full periods for numTimePts */
    MinoPeriodR = 2.0 * M_PI / orbpar.Yr;
    deltaLambda = (double)numPeriods * MinoPeriodR / ((double)numTimePts - 1.0);

    /* Get frequencies for n-mode decomposition */
    omegaPhi = orbpar.wphi;
    omegaR = orbpar.wr;

    /* setting up output files */
    FILE *fp, *esd, *nmd;
    fp = fopen("data/traj_source.dat", "w");
    fprintf(fp, "# lambda\ttime\tradius\tphi\tfourVel\tReField\tImField\tReEffSrc\tImEffSrc\n");
    esd = fopen("data/puncture_derivs.dat", "w");
    fprintf(esd, "# time\tReDtField\tImDtField\tReDrField\tImDrField\tReDthField\tImDthField\tReDphiField\tImDphiField\n");
    nmd = fopen("data/nmode_data.dat", "w");
    fprintf(nmd, "# time\tReField\tImField\tReDtField\tImDtField\tReDrField\tImDrField\tReDthField\tImDthField\tReDphiField\tImDphiField\tReEffSrc\tImEffSrc\n");

    // looping over Mino time steps
    for (int i = 0; i < numTimePts; i++)
    {
        lam = (double)i * deltaLambda;

        // find BL coordinate positions
        psi = korb_psifromla(lam, orbpar);
        t = korb_tfromla(lam, orbpar);
        r_p = korb_rfrompsi(psi, orbpar);
        phi_p = korb_phifromla(lam, orbpar);

        // set four-velocity
        ur = fourVel(psi, a, p, e, orbpar.E);

        // set effective source particle struct
        xParticle.t = 0.0;
        xParticle.r = r_p;
        xParticle.theta = M_PI_2; // enforce equatorial orbits for now
        xParticle.phi = phi_p;

        // compute effective source coefficients
        effsource_set_particle(&xParticle, orbpar.E, orbpar.Lz, ur);

        // calculate m-mode field
        effsource_calc_m(mMode, &xField, PhiS, dPhiS, ddPhiS, src);

        fprintf(fp, "%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\n",
                lam, t, r_p, phi_p, ur, PhiS[0], PhiS[1], src[0], src[1]);

        fprintf(esd, "%.15g\t%.15g\t%.15g\t%.15g\t%.15g\t%.15g\t%.15g\t%.15g\t%.15g\n",
                t, dPhiS[0], dPhiS[1], dPhiS[2], dPhiS[3], dPhiS[4], dPhiS[5], dPhiS[6], dPhiS[7]);

        /* get n-mode integrand timeseries data */
        generateNModeIntegrands(t, omegaPhi, omegaR, mMode, nMode, PhiS, dPhiS, src, nModePhiS, nModeDPhiS, nModesrc);

        fprintf(nmd, "%.15g\t%.15g\t%.15g\t%.15g\t%.15g\t%.15g\t%.15g\t%.15g\t%.15g\t%.15g\t%.15g\t%.15g\t%.15g\n",
                t,
                nModePhiS[0],
                nModePhiS[1],
                nModeDPhiS[0],
                nModeDPhiS[1],
                nModeDPhiS[2],
                nModeDPhiS[3],
                nModeDPhiS[4],
                nModeDPhiS[5],
                nModeDPhiS[6],
                nModeDPhiS[7],
                nModesrc[0],
                nModesrc[1]);
    }

    fclose(fp);
    fclose(esd);
    fclose(nmd);

    korb_freepar(orbpar);

    return 0;
}
