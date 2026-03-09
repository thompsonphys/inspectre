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
    int numRadialPts, numThetaPts;
    double PhiS[2], dPhiS[8], ddPhiS[20], src[2];
    double nModePhiS[2], nModeDPhiS[8], nModesrc[2];
    double lam, psi, t, r_p, phi_p, ur, a, p, e, x, err;
    double MinoPeriodR, deltaLambda, omegaPhi, omegaR;
    double rMin, rMax, rMinField, rMaxField, deltaRadius;
    double thetaMin, thetaMax, deltaTheta;
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
    numTimePts = (int)strtod(argv[5], NULL);
    numPeriods = (int)strtod(argv[6], NULL);

    /* set m-mode and n-mode from input */
    mMode = (int)strtod(argv[7], NULL);
    nMode = (int)strtod(argv[8], NULL);

    /* set spatial sampling from input */
    numRadialPts = (int)strtod(argv[9], NULL);
    numThetaPts = (int)strtod(argv[10], NULL);

    eccentric = e > 0.0 ? 1 : 0;
    inclined = fabs(1.0 - x) <= 1e-14 ? 0 : 1;

    err = 1.0e-15;

    if (eccentric == 1)
    {
        rMin = p / (1.0 + e);
        rMax = p / (1.0 - e);
    }
    else
    {
        rMax = p;
        rMin = p;
    }

    /* just hack this in here */
    rMinField = rMin - 1.0;
    rMaxField = rMax + 1.0;
    deltaRadius = (rMaxField - rMinField) / ((double)numRadialPts - 1.0);

    thetaMin = M_PI_2 - 0.08;
    thetaMax = M_PI_2 + 0.08;
    deltaTheta = (thetaMax - thetaMin) / ((double)numThetaPts - 1.0);

    /* All orbital characteristics are calculated and stored inside "orbpar" */
    korb_getparams(eccentric, inclined, a, p, e, x, err, &orbpar);

    /* Get the Mino time radial period and sample uniformly over numPeriods
       full periods for numTimePts */
    MinoPeriodR = orbpar.Vr;
    deltaLambda = (double)numPeriods * MinoPeriodR / ((double)numTimePts - 1.0);

    /* Get frequencies for n-mode decomposition */
    omegaPhi = orbpar.wphi;
    omegaR = orbpar.wr;

    xField.t = 0.0;
    xField.phi = 0.0;

    /* setting up output files */
    FILE *trajFile, *fieldReFile, *fieldImFile, *sourceReFile, *sourceImFile, *fieldDrReFile, *fieldDrImFile, *fieldDthReFile, *fieldDthImFile;
    FILE *nfieldReFile, *nfieldImFile, *nsourceReFile, *nsourceImFile, *nfieldDrReFile, *nfieldDrImFile, *nfieldDthReFile, *nfieldDthImFile;

    trajFile = fopen("data/trajectory.dat", "w");
    fprintf(trajFile, "# lambda\ttime\tradius\tphi\tfourVel\n");
    fieldReFile = fopen("data/puncture_re.dat", "w");
    fprintf(fieldReFile, "# time\tPts\n");
    fieldImFile = fopen("data/puncture_im.dat", "w");
    fprintf(fieldImFile, "# time\tPts\n");
    sourceReFile = fopen("data/source_re.dat", "w");
    fprintf(sourceReFile, "# time\tPts\n");
    sourceImFile = fopen("data/source_im.dat", "w");
    fprintf(sourceImFile, "# time\tPts\n");
    fieldDrReFile = fopen("data/puncture_dr_re.dat", "w");
    fprintf(fieldDrReFile, "# time\tPts\n");
    fieldDrImFile = fopen("data/puncture_dr_im.dat", "w");
    fprintf(fieldDrImFile, "# time\tPts\n");
    fieldDthReFile = fopen("data/puncture_dth_re.dat", "w");
    fprintf(fieldDthReFile, "# time\tPts\n");
    fieldDthImFile = fopen("data/puncture_dth_im.dat", "w");
    fprintf(fieldDthImFile, "# time\tPts\n");

    nfieldReFile = fopen("data/n_puncture_re.dat", "w");
    fprintf(nfieldReFile, "# time\tPts\n");
    nfieldImFile = fopen("data/n_puncture_im.dat", "w");
    fprintf(nfieldImFile, "# time\tPts\n");
    nsourceReFile = fopen("data/n_source_re.dat", "w");
    fprintf(nsourceReFile, "# time\tPts\n");
    nsourceImFile = fopen("data/n_source_im.dat", "w");
    fprintf(nsourceImFile, "# time\tPts\n");
    nfieldDrReFile = fopen("data/n_puncture_dr_re.dat", "w");
    fprintf(nfieldDrReFile, "# time\tPts\n");
    nfieldDrImFile = fopen("data/n_puncture_dr_im.dat", "w");
    fprintf(nfieldDrImFile, "# time\tPts\n");
    nfieldDthReFile = fopen("data/n_puncture_dth_re.dat", "w");
    fprintf(nfieldDthReFile, "# time\tPts\n");
    nfieldDthImFile = fopen("data/n_puncture_dth_im.dat", "w");
    fprintf(nfieldDthImFile, "# time\tPts\n");

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

        fprintf(trajFile, "%.15f\t%.15f\t%.15f\t%.15f\t%.15f\n",
                lam, t, r_p, phi_p, ur);

        fprintf(fieldReFile, "%.15g", t);
        fprintf(fieldImFile, "%.15g", t);
        fprintf(sourceReFile, "%.15g", t);
        fprintf(sourceImFile, "%.15g", t);
        fprintf(fieldDrReFile, "%.15g", t);
        fprintf(fieldDrImFile, "%.15g", t);
        fprintf(fieldDthReFile, "%.15g", t);
        fprintf(fieldDthImFile, "%.15g", t);

        fprintf(nfieldReFile, "%.15g", t);
        fprintf(nfieldImFile, "%.15g", t);
        fprintf(nsourceReFile, "%.15g", t);
        fprintf(nsourceImFile, "%.15g", t);
        fprintf(nfieldDrReFile, "%.15g", t);
        fprintf(nfieldDrImFile, "%.15g", t);
        fprintf(nfieldDthReFile, "%.15g", t);
        fprintf(nfieldDthImFile, "%.15g", t);

        // compute effective source coefficients
        effsource_set_particle(&xParticle, orbpar.E, orbpar.Lz, ur);

        // looping over theta field points
        for (int j = 0; j < numThetaPts; j++)
        {
            xField.theta = thetaMin + (double)j * deltaTheta;

            // looping over radial field points
            for (int k = 0; k < numRadialPts; k++)
            {
                xField.r = rMinField + (double)k * deltaRadius;

                // calculate m-mode field
                effsource_calc_m(mMode, &xField, PhiS, dPhiS, ddPhiS, src);

                fprintf(fieldReFile, ",%.15g", PhiS[0]);
                fprintf(fieldImFile, ",%.15g", PhiS[1]);
                fprintf(sourceReFile, ",%.15g", src[0]);
                fprintf(sourceImFile, ",%.15g", src[1]);
                fprintf(fieldDrReFile, ",%.15g", dPhiS[2]);
                fprintf(fieldDrImFile, ",%.15g", dPhiS[3]);
                fprintf(fieldDthReFile, ",%.15g", dPhiS[4]);
                fprintf(fieldDthImFile, ",%.15g", dPhiS[5]);

                /* get n-mode integrand timeseries data */
                generateNModeIntegrands(t, omegaPhi, omegaR, mMode, nMode, PhiS, dPhiS, src, nModePhiS, nModeDPhiS, nModesrc);

                fprintf(nfieldReFile, ",%.15g", nModePhiS[0]);
                fprintf(nfieldImFile, ",%.15g", nModePhiS[1]);
                fprintf(nsourceReFile, ",%.15g", nModesrc[0]);
                fprintf(nsourceImFile, ",%.15g", nModesrc[1]);
                fprintf(nfieldDrReFile, ",%.15g", nModeDPhiS[2]);
                fprintf(nfieldDrImFile, ",%.15g", nModeDPhiS[3]);
                fprintf(nfieldDthReFile, ",%.15g", nModeDPhiS[4]);
                fprintf(nfieldDthImFile, ",%.15g", nModeDPhiS[5]);
            }
        }
        fprintf(fieldReFile, "\n");
        fprintf(fieldImFile, "\n");
        fprintf(sourceReFile, "\n");
        fprintf(sourceImFile, "\n");
        fprintf(fieldDrReFile, "\n");
        fprintf(fieldDrImFile, "\n");
        fprintf(fieldDthReFile, "\n");
        fprintf(fieldDthImFile, "\n");
        fprintf(nfieldReFile, "\n");
        fprintf(nfieldImFile, "\n");
        fprintf(nsourceReFile, "\n");
        fprintf(nsourceImFile, "\n");
        fprintf(nfieldDrReFile, "\n");
        fprintf(nfieldDrImFile, "\n");
        fprintf(nfieldDthReFile, "\n");
        fprintf(nfieldDthImFile, "\n");
    }

    fclose(trajFile);
    fclose(fieldReFile);
    fclose(fieldImFile);
    fclose(sourceReFile);
    fclose(sourceImFile);
    fclose(fieldDrReFile);
    fclose(fieldDrImFile);
    fclose(fieldDthReFile);
    fclose(fieldDthImFile);
    fclose(nfieldReFile);
    fclose(nfieldImFile);
    fclose(nsourceReFile);
    fclose(nsourceImFile);
    fclose(nfieldDrReFile);
    fclose(nfieldDrImFile);
    fclose(nfieldDthReFile);
    fclose(nfieldDthImFile);

    korb_freepar(orbpar);

    return 0;
}
