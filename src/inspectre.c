#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_odeiv2.h>

#include "../lib/kerrgeodesics/korb.h"
#include "../lib/effectivesource/effsource.h"

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
