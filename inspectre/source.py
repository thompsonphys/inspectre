import numpy as np


class EffectiveSource:
    """High-level wrapper around the effsource SWIG modules.

    Parameters
    ----------
    spin : float
        Black hole spin parameter (a/M).
    mass : float
        Black hole mass, default 1.0.
    mode : str
        Which effective-source module to use: "circular" or "equatorial".
    """

    def __init__(self, spin, mass=1.0, mode="equatorial"):
        if mode == "circular":
            import effsource_circular as _ef
        elif mode == "equatorial":
            import effsource_equatorial as _ef
        else:
            raise ValueError(f"mode must be 'circular' or 'equatorial', got {mode!r}")

        self._ef = _ef
        self._ef.disable_gsl_error_handler()
        self._ef.effsource_init(mass, spin)

    @staticmethod
    def make_coordinate(t, r, theta, phi):
        """Create a coordinate struct. Uses whichever effsource module is available."""
        try:
            import effsource_equatorial as _ef
        except ImportError:
            import effsource_circular as _ef
        return _ef.make_coordinate(t=t, r=r, theta=theta, phi=phi)

    def set_particle(self, r, theta, phi, energy, lz, ur):
        """Set the particle position and orbital parameters.

        Parameters
        ----------
        r, theta, phi : float
            Particle position (Boyer-Lindquist coordinates, t is set to 0).
        energy : float
            Specific orbital energy E.
        lz : float
            Specific angular momentum L_z.
        ur : float
            Radial component of four-velocity u^r.
        """
        xp = self._ef.make_coordinate(t=0.0, r=r, theta=theta, phi=phi)
        self._ef.effsource_set_particle(xp, energy, lz, ur)

    def calc_m(self, m, r_field, theta_field):
        """Compute the m-mode effective source at a field point.

        Parameters
        ----------
        m : int
            Azimuthal mode number.
        r_field, theta_field : float
            Field point position (t=0, phi=0).

        Returns
        -------
        dict with keys:
            PhiS : ndarray shape (2,) — [Re, Im] singular field
            dPhiS : ndarray shape (4, 2) — first derivatives, rows [t, r, theta, phi]
            d2PhiS : ndarray shape (10, 2) — second derivatives (Re/Im interleaved)
            src : ndarray shape (2,) — [Re, Im] effective source
        """
        xf = self._ef.make_coordinate(t=0.0, r=r_field, theta=theta_field, phi=0.0)
        PhiS, dPhiS, d2PhiS, src = self._ef.calc_m(m, xf)

        return {
            "PhiS": np.array(PhiS),
            "dPhiS": np.array(dPhiS).reshape(4, 2),
            "d2PhiS": np.array(d2PhiS).reshape(10, 2),
            "src": np.array(src),
        }

    def calc(self, r_field, theta_field, phi_field=0.0):
        """Compute the full (non-decomposed) effective source at a field point.

        Parameters
        ----------
        r_field, theta_field : float
            Field point position (t=0).
        phi_field: float, optional
            Field point azimuthal position

        Returns
        -------
        dict with keys:
            PhiS : float — singular field
            dPhiS : ndarray shape (4,) — first derivatives [t, r, theta, phi]
            d2PhiS : ndarray shape (10,) — second derivatives
            src : float — effective source
        """
        xf = self._ef.make_coordinate(
            t=0.0, r=r_field, theta=theta_field, phi=phi_field
        )
        PhiS, dPhiS, d2PhiS, src = self._ef.calc(xf)

        return {
            "PhiS": PhiS,
            "dPhiS": np.array(dPhiS),
            "d2PhiS": np.array(d2PhiS),
            "src": src,
        }
