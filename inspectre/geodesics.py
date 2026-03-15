import math
import kerrgeodesics as _kg


class KerrOrbit:
    """High-level wrapper around the kerrgeodesics SWIG module.

    Parameters
    ----------
    spin : float
        Black hole spin parameter (a/M), |a| < 1.
    semilatus_rectum : float
        Orbital semilatus rectum (p).
    eccentricity : float
        Orbital eccentricity (e), 0 <= e < 1.
    x : float
        Inclination parameter x = sin(theta_min), default 1.0 (equatorial).
    err : float
        Fourier series convergence tolerance, default 1e-15.
    """

    def __init__(
        self,
        spin=0.0,
        semilatus_rectum=10.0,
        eccentricity=0.0,
        x=1.0,
        err=1e-15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._params = _kg.korb_params()
        self._closed = False

        eccentric = 1 if eccentricity > 0.0 else 0
        inclined = 0 if abs(1.0 - x) <= 1e-14 else 1

        _kg.korb_getparams(
            eccentric,
            inclined,
            spin,
            semilatus_rectum,
            eccentricity,
            x,
            err,
            self._params,
        )

    # -- orbital constants --------------------------------------------------

    @property
    def energy(self):
        return self._params.E

    @property
    def angular_momentum(self):
        return self._params.Lz

    @property
    def carter_constant(self):
        return self._params.Q

    # -- orbital frequencies (Boyer-Lindquist t) ----------------------------

    @property
    def omega_r(self):
        return self._params.wr

    @property
    def omega_theta(self):
        return self._params.wth

    @property
    def omega_phi(self):
        return self._params.wphi

    # -- Mino-time frequencies ----------------------------------------------

    @property
    def upsilon_r(self):
        return self._params.Yr

    @property
    def upsilon_theta(self):
        return self._params.Yth

    @property
    def upsilon_phi(self):
        return self._params.Yphi

    @property
    def gamma(self):
        return self._params.Ga

    # -- Mino-time periods --------------------------------------------------

    @property
    def mino_period_r(self):
        return 2.0 * math.pi / self._params.Yr

    @property
    def mino_period_theta(self):
        return 2.0 * math.pi / self._params.Yth

    # -- orbital element access ---------------------------------------------

    @property
    def spin(self):
        return self._params.a

    @property
    def semilatus_rectum(self):
        return self._params.p

    @property
    def eccentricity(self):
        return self._params.e

    @property
    def x(self):
        return self._params.x

    @property
    def params(self):
        """Direct access to the underlying korb_params struct."""
        return self._params

    # -- trajectory evaluation ----------------------------------------------

    def psi_from_lambda(self, lam):
        """chi_r from Mino time."""
        return _kg.korb_psifromla(lam, self._params)

    def r_from_lambda(self, lam):
        """Boyer-Lindquist r from lambda through chi_r."""
        psi = self.psi_from_lambda(lam)
        return _kg.korb_rfrompsi(psi, self._params)

    def t_from_lambda(self, lam):
        """Boyer-Lindquist t from Mino time."""
        return _kg.korb_tfromla(lam, self._params)

    def phi_from_lambda(self, lam):
        """Boyer-Lindquist phi from Mino time."""
        return _kg.korb_phifromla(lam, self._params)

    def theta_from_lambda(self, lam):
        """Boyer-Lindquist theta from Mino time."""
        return _kg.korb_thfromla(lam, self._params)

    def four_velocity_equatorial(self, lam):
        """Compute u^r for equatorial orbits from lambda through chi_r."""
        psi = self.psi_from_lambda(lam)
        return _kg.korb_getfourvel_equatorial(psi, self._params)

    # -- cleanup ------------------------------------------------------------

    def close(self):
        if not self._closed:
            _kg.korb_freepar(self._params)
            self._closed = True

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
