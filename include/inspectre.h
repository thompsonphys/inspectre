#ifndef INSPECTRE_H
#define INSPECTRE_H

#include "../lib/kerrgeodesics/korb.h"

double fourVelocityXFunction(double spin, double semiLatusRectum, double eccentricity);
double fourVelocity(double psi, double spin, double semiLatusRectum, double eccentricity, double energy);
double frequencyShiftReal(double time, double omegaPhi, double omegaR, double fieldRE, double fieldIM, int mMode, int nMode);
double frequencyShiftImag(double time, double omegaPhi, double omegaR, double fieldRE, double fieldIM, int mMode, int nMode);
void generateNModeIntegrandAtTime(double time, double omegaPhi, double omegaR, int mMode, int nMode, double *field, double *nModeField);
void generateKerrOrbitalParameters(korb_params *orbitalParameters, spin, semiLatusRectum, eccentricity, x);

#endif