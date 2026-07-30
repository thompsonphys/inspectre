from kerrgeodesics import KerrOrbit
from .source import EffectiveSource
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from copy import deepcopy

# Optional low-level C bindings (src/inspectre_lib.c). Built separately via the
# project setup.py; the pure-Python methods below work without it.
try:
    import inspectre_c
except ImportError:
    inspectre_c = None


def set_array(duration, num_pts, buffer=0):
    assert buffer >= 0

    delta = duration / num_pts

    # duration inclusive time array
    time_array = np.arange(-buffer, num_pts + buffer + 1) * delta
    if buffer == 0:
        mask = time_array == time_array
    else:
        mask = (time_array >= 0.0) & (time_array <= duration)
    return delta, time_array, mask


# inheritance was breaking, doing this the hard way
class Inspectre:

    def __init__(
        self,
        spin=0.0,
        semilatus_rectum=10.0,
        eccentricity=0.0,
        x=1.0,
        mass=1.0,
        err=1e-15,
        mode="equatorial",
        **kwargs,
    ):

        self.orbit = KerrOrbit(
            spin=spin,
            semilatus_rectum=semilatus_rectum,
            eccentricity=eccentricity,
            x=x,
            err=err,
        )
        self.es = EffectiveSource(mass=mass, spin=spin, mode=mode)

        self._trajectory_exists = False
        self._trajectory_resampled = False
        self.BUFFER = 10
        self._old_traj = None

    def __del__(self):
        del self.orbit
        del self.es

    # -- orbital constants --------------------------------------------------

    @property
    def energy(self):
        return self.orbit.energy

    @property
    def angular_momentum(self):
        return self.orbit.angular_momentum

    @property
    def carter_constant(self):
        return self.orbit.carter_constant

    # -- orbital frequencies (Boyer-Lindquist t) ----------------------------

    @property
    def epicyclic_frequency(self):
        a, p = self.spin, self.semilatus_rectum
        R = 1.0 - 6.0 / p + 8.0 * a / p**1.5 - 3.0 * a * a / (p * p)
        return self.orbit.omega_phi * np.sqrt(max(R, 0.0))

    @property
    def omega_r(self):
        if self.params.eccentric == 0:
            return self.epicyclic_frequency
        return self.orbit.omega_r

    @property
    def omega_theta(self):
        return self.orbit.omega_theta

    @property
    def omega_phi(self):
        return self.orbit.omega_phi

    # -- Mino-time frequencies ----------------------------------------------

    @property
    def upsilon_r(self):
        if self.params.eccentric == 0:
            return self.gamma * self.epicyclic_frequency
        return self.orbit.upsilon_r

    @property
    def upsilon_theta(self):
        return self.orbit.upsilon_theta

    @property
    def upsilon_phi(self):
        return self.orbit.upsilon_phi

    @property
    def gamma(self):
        return self.orbit.gamma

    # -- Mino-time periods --------------------------------------------------

    @property
    def mino_period_r(self):
        if self.params.eccentric == 0:
            return 2.0 * np.pi / self.upsilon_r
        return self.orbit.mino_period_r

    @property
    def mino_period_theta(self):
        return self.orbit.mino_period_theta

    # -- orbital element access ---------------------------------------------

    @property
    def spin(self):
        return self.orbit.spin

    @property
    def semilatus_rectum(self):
        return self.orbit.semilatus_rectum

    @property
    def eccentricity(self):
        return self.orbit.eccentricity

    @property
    def x(self):
        return self.orbit.x

    @property
    def params(self):
        """Direct access to the underlying korb_params struct."""
        return self.orbit.params

    # -- trajectory evaluation ----------------------------------------------

    def psi_from_lambda(self, lam):
        """chi_r from Mino time."""
        return self.orbit.psi_from_lambda(lam)

    def r_from_lambda(self, lam):
        """Boyer-Lindquist r from lambda through chi_r."""
        return self.orbit.r_from_lambda(lam)

    def t_from_lambda(self, lam):
        """Boyer-Lindquist t from Mino time."""
        return self.orbit.t_from_lambda(lam)

    def phi_from_lambda(self, lam):
        """Boyer-Lindquist phi from Mino time."""
        return self.orbit.phi_from_lambda(lam)

    def theta_from_lambda(self, lam):
        """Boyer-Lindquist theta from Mino time."""
        return self.orbit.theta_from_lambda(lam)

    def four_velocity_equatorial(self, lam):
        """Compute u^r for equatorial orbits from lambda through chi_r."""
        return self.orbit.four_velocity_equatorial(lam)

    # -- Set Source Functions -----------------------------------------------

    def set_particle(self, r_p, theta_p, phi_p, u_r):
        self.es.set_particle(
            r_p, theta_p, phi_p, self.energy, self.angular_momentum, u_r
        )

    def phi_s(self, r_field, theta_field, phi_field):
        return self.es.phi_s(r_field, theta_field, phi_field)

    def phi_s_m(self, m, r_field, theta_field):
        return self.es.phi_s_m(m, r_field, theta_field)

    def source(self, r_field, theta_field, phi_field):
        return self.es.source(r_field, theta_field, phi_field)

    def source_m(self, m, r_field, theta_field):
        return self.es.source_m(m, r_field, theta_field)
    
    # -- Populate source along trajectory -----------------------------------

    def generate_equatorial_trajectory(self, num_periods=1, num_pts=100):

        self._trajectory_resampled = False
        if hasattr(self, "_old_traj") and isinstance(self._old_traj, dict):
            del self._old_traj

        total_duration = num_periods * self.mino_period_r

        delta_lambda, lambda_values, lambda_mask = set_array(
            total_duration, num_pts, buffer=self.BUFFER
        )

        traj_data = []
        for lam in lambda_values:
            traj_data.append(
                [
                    self.t_from_lambda(lam),
                    self.r_from_lambda(lam),
                    self.theta_from_lambda(lam),
                    self.phi_from_lambda(lam),
                    self.four_velocity_equatorial(lam),
                ]
            )

        t_vals, r_vals, theta_vals, phi_vals, ur_vals = np.array(traj_data).T

        # store full trajectory, including buffer, for later potential interpolation
        self._trajectory_full = {
            "metadata": {
                "periods": num_periods,
                "delta_lambda": delta_lambda,
                "lambda_mask": lambda_mask,
            },
            "lambda": lambda_values,
            "t": t_vals,
            "r": r_vals,
            "theta": theta_vals,
            "phi": phi_vals,
            "ur": ur_vals,
        }

        # give requested range here
        self.trajectory = {
            "metadata": {"periods": num_periods, "delta_lambda": delta_lambda},
            "lambda": lambda_values[lambda_mask],
            "t": t_vals[lambda_mask],
            "r": r_vals[lambda_mask],
            "theta": theta_vals[lambda_mask],
            "phi": phi_vals[lambda_mask],
            "ur": ur_vals[lambda_mask],
        }
        self._trajectory_exists = True

    def check_trajectory(self):
        if not self._trajectory_exists:
            raise ValueError("Please first generate a trajectory.")

    def resample_trajectory(self, new_indep_array=None, num_pts=100, indep_var="t"):
        # resample either in lambda or t, let user decide independent variable

        self.check_trajectory()

        if new_indep_array is None:

            old_duration = self.trajectory[indep_var][-1]

            new_delta, new_indep_var, _ = set_array(old_duration, num_pts, buffer=0)

            self.trajectory[indep_var] = new_indep_var
            old_indep_var_full = self._trajectory_full[indep_var]

            # spline over the trajectory with boundary buffer
            for k, coord in self._trajectory_full.items():
                if k in set([indep_var, "metadata"]):
                    continue

                self.trajectory[k] = CubicSpline(old_indep_var_full, coord)(
                    new_indep_var
                )

            self.trajectory["metadata"][f"delta_{indep_var}"] = new_delta
        else:
            start = self.trajectory[indep_var][0]
            end = self.trajectory[indep_var][-1]
            assert (new_indep_array[0] >= start) and (new_indep_array[-1] <= end)

            self.trajectory[indep_var] = new_indep_array
            old_indep_var_full = self._trajectory_full[indep_var]

            # spline over the trajectory with boundary buffer
            for k, coord in self._trajectory_full.items():
                if k in set([indep_var, "metadata"]):
                    continue

                self.trajectory[k] = CubicSpline(old_indep_var_full, coord)(
                    new_indep_array
                )

            self.trajectory["metadata"][f"delta_{indep_var}"] = None

    def puncture_along_trajectory(self, r_field, theta_field, phi_field):

        self.check_trajectory()

        field_values = []
        zipped_coords = zip(
            self.trajectory["r"],
            self.trajectory["theta"],
            self.trajectory["phi"],
            self.trajectory["ur"],
        )
        for r_p, theta_p, phi_p, ur in zipped_coords:
            self.set_particle(r_p, theta_p, phi_p, ur)
            field_values.append(self.phi_s(r_field, theta_field, phi_field))

        return np.array([self.trajectory["t"], field_values]).T

    def puncture_mmode_along_trajectory(self, m, r_field, theta_field):

        self.check_trajectory()

        field_values = []
        zipped_coords = zip(
            self.trajectory["t"],
            self.trajectory["r"],
            self.trajectory["theta"],
            self.trajectory["phi"],
            self.trajectory["ur"],
        )
        for t_p, r_p, theta_p, phi_p, ur in zipped_coords:
            self.set_particle(r_p, theta_p, phi_p, ur)
            re, im = self.phi_s_m(m, r_field, theta_field)
            field_values.append([t_p, re, im])

        return np.array(field_values)

    def puncture_mn_integrand_along_trajectory(self, m, n, r_field, theta_field):

        m_mode_data = self.puncture_mmode_along_trajectory(m, r_field, theta_field)

        t, re, im = m_mode_data.T
        angular_frequency = m * self.omega_phi + n * self.omega_r

        sin_omega_t = np.sin(t * angular_frequency)
        cos_omega_t = np.cos(t * angular_frequency)

        return np.array(
            [
                t,
                re * cos_omega_t - im * sin_omega_t,
                im * cos_omega_t + re * sin_omega_t,
            ]
        ).T

    # -- Fast C bindings (inspectre_c) --------------------------------------
    #
    # These call the compiled routines in src/inspectre_lib.c directly, passing
    # the raw korb_params / effsource_equatorial_ctx handles that this object
    # already owns. They are equivalent to the pure-Python methods above but
    # avoid the per-point Python overhead. Require the inspectre_c extension.

    def _require_c(self):
        if inspectre_c is None:
            raise RuntimeError(
                "inspectre_c extension not built; install it with "
                "`pip install -e .` from the inspectre project root."
            )

    @property
    def _ctx(self):
        """Raw effsource_equatorial_ctx pointer (borrowed; owned by self.es)."""
        return self.es._ef._ctx

    @property
    def _orbpar(self):
        """Raw korb_params pointer (borrowed; owned by self.orbit)."""
        return self.orbit._params

    def eval_at_lambda_fast(self, m, lam, r_field, theta_field, phi_field=0.0):
        """C eval of puncture + effective source at Mino time lambda.

        Returns (PhiS[2], dPhiS[8], ddPhiS[20], src[2]) as lists (Re/Im interleaved).
        """
        self._require_c()
        xF = self.es.make_coordinate(0.0, r_field, theta_field, phi_field)
        return inspectre_c.eval_at_lambda(
            self._ctx, m, xF, lam, self._orbpar,
            self.spin, self.semilatus_rectum, self.eccentricity,
        )

    def eval_at_time_fast(self, m, t, r_field, theta_field, phi_field=0.0):
        """C eval of puncture + effective source at coordinate time t.

        Brent-inverts lambda(t) internally. Returns
        (PhiS[2], dPhiS[8], ddPhiS[20], src[2]) as lists (Re/Im interleaved).
        """
        self._require_c()
        xF = self.es.make_coordinate(0.0, r_field, theta_field, phi_field)
        return inspectre_c.eval_at_time(
            self._ctx, m, xF, t, self._orbpar,
            self.spin, self.semilatus_rectum, self.eccentricity,
        )

    def lambda_from_t(self, t):
        """Invert t = t(lambda) for lambda via the C Brent solver."""
        self._require_c()
        return inspectre_c.inspectre_lambda_from_t(t, self._orbpar)

    # Integration modes that integrate caller-supplied per-sample arrays rather
    # than sampling the source themselves. The C driver dereferences the sample
    # pointers directly, so calling these with NULL arrays segfaults the process
    # (and the Jupyter kernel) -- build the samples here before dispatching.
    _SAMPLE_BASED_MODES = None  # populated lazily once inspectre_c is importable

    def integrate_nmode_fast(self, m, n, r_field, theta_field, phi_field=0.0,
                             mode=None, nSamples=257, epsabs=1e-10, epsrel=1e-10):
        """C n-mode Fourier amplitude over one radial period.

        `mode` defaults to the self-contained MINO_SPLINE quadrature; pass another
        INSPECTRE_INTEG_* constant from inspectre_c to switch. The sample-based
        modes (TIMESERIES, SIMPSON, SPLINE) integrate a precomputed timeseries; we
        build it here on a closed uniform-t grid of `nSamples` points over one
        radial period. Returns (nModePhiS[2], nModeDPhiS[8], nModesrc[2]) as lists.
        """
        self._require_c()
        if mode is None:
            mode = inspectre_c.INSPECTRE_INTEG_MINO_SPLINE
        if Inspectre._SAMPLE_BASED_MODES is None:
            Inspectre._SAMPLE_BASED_MODES = {
                inspectre_c.INSPECTRE_INTEG_TIMESERIES,
                inspectre_c.INSPECTRE_INTEG_SIMPSON,
                inspectre_c.INSPECTRE_INTEG_SPLINE,
            }
        xF = self.es.make_coordinate(0.0, r_field, theta_field, phi_field)

        tSamples = fieldSamples = derivSamples = srcSamples = None
        if mode in Inspectre._SAMPLE_BASED_MODES:
            tSamples, fieldSamples, derivSamples, srcSamples = \
                self._build_nmode_samples(m, r_field, theta_field, phi_field,
                                          nSamples)

        return inspectre_c.integrate_nmode(
            mode, self._ctx, m, n, xF, self._orbpar,
            self.spin, self.semilatus_rectum, self.eccentricity,
            self.omega_phi, self.omega_r, epsabs, epsrel,
            tSamples=tSamples, fieldSamples=fieldSamples,
            derivSamples=derivSamples, srcSamples=srcSamples,
            nSamples=nSamples,
        )

    def _build_nmode_samples(self, m, r_field, theta_field, phi_field, nSamples):
        """Closed uniform-t timeseries of the puncture/source for sample-based
        n-mode quadrature.

        Returns (tSamples[N], fieldSamples[2N], derivSamples[8N], srcSamples[2N])
        as flat lists in the interleaved layout the C driver expects. The grid is
        the closed period [0, Tr] (t[0]=0, t[N-1]=Tr is the periodic image of t=0)
        so the trapezoid/Simpson rules close the wrap interval -- matching the
        reference sampling in test/recontest.c.
        """
        Tr = 2.0 * np.pi / self.omega_r
        tSamples = [0.0] * nSamples
        fieldSamples = [0.0] * (2 * nSamples)
        derivSamples = [0.0] * (8 * nSamples)
        srcSamples = [0.0] * (2 * nSamples)
        for i in range(nSamples):
            t = i / (nSamples - 1) * Tr
            PhiS, dPhiS, _ddPhiS, src = self.eval_at_time_fast(
                m, t, r_field, theta_field, phi_field)
            tSamples[i] = t
            fieldSamples[2 * i], fieldSamples[2 * i + 1] = PhiS[0], PhiS[1]
            for c in range(8):
                derivSamples[8 * i + c] = dPhiS[c]
            srcSamples[2 * i], srcSamples[2 * i + 1] = src[0], src[1]
        return tSamples, fieldSamples, derivSamples, srcSamples

    def panel_nmodes_fast(self, m, n_list, r_field, theta_field, phi_field=0.0,
                          order=16, max_levels=40):
        """All requested n-mode amplitudes from one panel-GL node build.

        Splits the radial period at the analytic closest-approach breakpoints,
        refines panels geometrically toward them, and applies Gauss-Legendre of
        `order` points per panel (`max_levels` caps the refinement depth). The
        node build is the only source-evaluating step; each n then costs a
        weighted phase sum. Returns a list of (nModePhiS[2], nModeDPhiS[8],
        nModesrc[2]) tuples, one per entry of n_list.
        """
        self._require_c()
        xF = self.es.make_coordinate(0.0, r_field, theta_field, phi_field)
        n_max = max(abs(int(n)) for n in n_list)
        s = inspectre_c.panel_nodes_build(
            self._ctx, m, xF, self._orbpar,
            self.spin, self.semilatus_rectum, self.eccentricity,
            self.omega_phi, self.omega_r,
            order=order, maxLevels=max_levels, nMax=n_max)
        try:
            return [inspectre_c.panel_nodes_integrate(
                        s, m, int(n), self.omega_phi, self.omega_r)
                    for n in n_list]
        finally:
            inspectre_c.panel_nodes_free(s)

    def fact_nmodes_fast(self, m, n_list, r_field, theta_field, phi_field=0.0,
                         NB=None, NK=1 << 20, KG=400):
        """All requested n-mode amplitudes from one kernel-factorization build.

        Pure-C port of inspectre.factorization: samples the seven-channel
        calc_m_split on an NB-point uniform-t grid (defaults to 4*n_max),
        FFTs channels and the six scalar kernels, then assembles each n by
        convolution. Returns a list of (nModePhiS[2], nModeDPhiS[8],
        nModesrc[2]) tuples, one per entry of n_list.
        """
        self._require_c()
        xF = self.es.make_coordinate(0.0, r_field, theta_field, phi_field)
        n_max = max(abs(int(n)) for n in n_list)
        if NB is None:
            NB = max(256, 4 * n_max)
        s = inspectre_c.fact_nodes_build(
            self._ctx, m, xF, self._orbpar,
            self.spin, self.semilatus_rectum, self.eccentricity,
            self.omega_phi, self.omega_r,
            NB=NB, NK=NK, KG=KG, nMax=n_max)
        try:
            return [inspectre_c.fact_nodes_integrate(s, int(n))
                    for n in n_list]
        finally:
            inspectre_c.fact_nodes_free(s)

    def fft_source_nmodes_fast(self, m, r_field, theta_field, phi_field=0.0, N=256):
        """C all-n source amplitudes from one FFTW transform.

        Returns (outRe[N], outIm[N]); bin k holds mode n = k if k<=N/2 else k-N.
        """
        self._require_c()
        xF = self.es.make_coordinate(0.0, r_field, theta_field, phi_field)
        return inspectre_c.fft_source_nmodes(
            self._ctx, m, xF, self._orbpar,
            self.spin, self.semilatus_rectum, self.eccentricity,
            self.omega_phi, self.omega_r, N,
        )

    def source_along_trajectory(self, r_field, theta_field, phi_field):

        self.check_trajectory()

        field_values = []
        zipped_coords = zip(
            self.trajectory["r"],
            self.trajectory["theta"],
            self.trajectory["phi"],
            self.trajectory["ur"],
        )
        for r_p, theta_p, phi_p, ur in zipped_coords:
            self.set_particle(r_p, theta_p, phi_p, ur)
            field_values.append(self.source(r_field, theta_field, phi_field))

        return np.array([self.trajectory["t"], field_values]).T

    def source_mmode_along_trajectory(self, m, r_field, theta_field, full_traj=False):

        self.check_trajectory()

        field_values = []
        if full_traj:
            traj_dict = self._trajectory_full
        else:
            traj_dict = self.trajectory

        zipped_coords = zip(
            traj_dict["t"],
            traj_dict["r"],
            traj_dict["theta"],
            traj_dict["phi"],
            traj_dict["ur"],
        )
        for t_p, r_p, theta_p, phi_p, ur in zipped_coords:
            self.set_particle(r_p, theta_p, phi_p, ur)
            re, im = self.source_m(m, r_field, theta_field)
            field_values.append([t_p, re, im])

        return np.array(field_values)

    def source_mn_integrand_along_trajectory(
        self, m, n, r_field, theta_field, full_traj=False
    ):

        m_mode_data = self.source_mmode_along_trajectory(
            m, r_field, theta_field, full_traj=full_traj
        )

        t, re, im = m_mode_data.T
        angular_frequency = m * self.omega_phi + n * self.omega_r

        sin_omega_t = np.sin(t * angular_frequency)
        cos_omega_t = np.cos(t * angular_frequency)

        return np.array(
            [
                t,
                re * cos_omega_t - im * sin_omega_t,
                im * cos_omega_t + re * sin_omega_t,
            ]
        ).T
