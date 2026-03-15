from .geodesics import KerrOrbit
from .source import EffectiveSource
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from copy import deepcopy


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

    def generate_equatorial_trajectory(self, num_periods=1, num_pts=100):

        self._trajectory_resampled = False
        if hasattr(self, "_old_traj") and isinstance(self._old_traj, dict):
            del self._old_traj

        total_duration = num_periods * self.mino_period_r

        # recompute delta_lambda to be endpoint inclusive
        delta_lambda = total_duration / num_pts

        # include endpoint
        lambda_values = np.arange(0, total_duration + delta_lambda, step=delta_lambda)

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

        self.trajectory = {
            "metadata": {"periods": num_periods, "delta_lambda": delta_lambda},
            "lambda": lambda_values,
            "t": t_vals,
            "r": r_vals,
            "theta": theta_vals,
            "phi": phi_vals,
            "ur": ur_vals,
        }
        self._trajectory_exists = True

    def check_trajectory(self):
        if not self._trajectory_exists:
            raise ValueError("Please first generate a trajectory.")

    def resample_trajectory(self, num_pts=100):

        self.check_trajectory()

        if self._trajectory_resampled:
            # copy back old trajectory to re-do sampling
            self.trajectory = deepcopy(self._old_traj)
            del self._old_traj
            self._trajectory_resampled = False

        self._old_traj = deepcopy(self.trajectory)

        old_time = self.trajectory["t"]
        new_delta_t = old_time[-1] / num_pts

        new_times = np.arange(old_time[0], old_time[-1] + new_delta_t, step=new_delta_t)

        self.trajectory["t"] = new_times

        for k, coord in self.trajectory.items():
            if k == "t" or k == "metadata":
                continue

            self.trajectory[k] = IUS(old_time, coord, k=3)(new_times)

        del self.trajectory["metadata"]["delta_lambda"]

        self.trajectory["metadata"]["delta_t"] = new_delta_t

        self._trajectory_resampled = True

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

    def source_mmode_along_trajectory(self, m, r_field, theta_field):

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
            re, im = self.source_m(m, r_field, theta_field)
            field_values.append([t_p, re, im])

        return np.array(field_values)
    

    def source_mn_integrand_along_trajectory(self, m, n, r_field, theta_field):

        m_mode_data = self.source_mmode_along_trajectory(m, r_field, theta_field)

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
