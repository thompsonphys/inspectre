from kerrgeodesics import KerrOrbit
from .source import EffectiveSource
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from copy import deepcopy


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
        semilatus_rect=10.0,
        eccentricity=0.0,
        x=1.0,
        mass=1.0,
        err=1e-15,
        mode="equatorial",
        **kwargs,
    ):

        self.orbit = KerrOrbit(
            spin=spin,
            semilatus_rect=semilatus_rect,
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
    def omega_r(self):
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
        return self.orbit.mino_period_r

    @property
    def mino_period_theta(self):
        return self.orbit.mino_period_theta

    # -- orbital element access ---------------------------------------------

    @property
    def spin(self):
        return self.orbit.spin

    @property
    def semilatus_rect(self):
        return self.orbit.semilatus_rect

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
