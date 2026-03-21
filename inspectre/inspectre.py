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


class Inspectre(KerrOrbit, EffectiveSource):

    def __init__(
        self,
        spin,
        semilatus_rectum,
        eccentricity,
        x=1.0,
        mass=1.0,
        err=1e-15,
        mode="equatorial",
        **kwargs,
    ):
        super().__init__(
            spin=spin,
            semilatus_rectum=semilatus_rectum,
            eccentricity=eccentricity,
            x=x,
            mass=mass,
            err=err,
            mode=mode,
            **kwargs,
        )

        self._trajectory_exists = False
        self._trajectory_resampled = False
        self.BUFFER = 10

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
            "lambda":lambda_values,
            "t":t_vals,
            "r":r_vals,
            "theta":theta_vals,
            "phi":phi_vals,
            "ur":ur_vals
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

    def resample_trajectory(self, num_pts=100, indep_var = "t"):
        # resample either in lambda or t, let user decide independent variable

        self.check_trajectory()

        old_duration = self.trajectory[indep_var][-1]

        new_delta, new_indep_var, _ = set_array(old_duration, num_pts, buffer=0)

        self.trajectory[indep_var] = new_indep_var
        old_indep_var_full = self._trajectory_full[indep_var]

        # spline over the trajectory with boundary buffer
        for k, coord in self._trajectory_full.items():
            if k in set([indep_var, "metadata"]):
                continue

            self.trajectory[k] = CubicSpline(old_indep_var_full, coord)(new_indep_var)

        self.trajectory["metadata"][f"delta_{indep_var}"] = new_delta

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
            self.set_particle(
                r_p, theta_p, phi_p, self.energy, self.angular_momentum, ur
            )
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
            self.set_particle(
                r_p, theta_p, phi_p, self.energy, self.angular_momentum, ur
            )
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
            self.set_particle(
                r_p, theta_p, phi_p, self.energy, self.angular_momentum, ur
            )
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
            self.set_particle(
                r_p, theta_p, phi_p, self.energy, self.angular_momentum, ur
            )
            re, im = self.source_m(m, r_field, theta_field)
            field_values.append([t_p, re, im])

        return np.array(field_values)

    def source_mn_integrand_along_trajectory(self, m, n, r_field, theta_field, full_traj=False):

        m_mode_data = self.source_mmode_along_trajectory(m, r_field, theta_field, full_traj=full_traj)

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
