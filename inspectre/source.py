import weakref

import numpy as np
from effsource_equatorial import (
    make_coordinate,
    EffsourceEquatorialContext,
    disable_gsl_error_handler,
)
from effsource_circular import EffsourceContext

try:
    import effsource_original
except ImportError:
    effsource_original = None

try:
    import effsource_original_agm
except ImportError:
    effsource_original_agm = None

ORBITS = ("equatorial", "circular")
IMPLS = ("refactored", "original", "original_agm")
ORIGINAL_IMPLS = {"original": effsource_original,
                  "original_agm": effsource_original_agm}
_ORIGINAL_LIVE = {}


class _OriginalAdapter:
    """Global-scope effsource code, one live instance per impl per process.

    Each impl is its own shared object with its own particle statics, so
    different impls do not collide and may be live simultaneously.

    Method names match EffsourceEquatorialContext for the subset the original
    API implements. The offset, split and extended methods are absent.
    """

    def __init__(self, impl, mass, spin):
        lib = ORIGINAL_IMPLS[impl]
        if lib is None:
            raise RuntimeError(
                f"effsource_{impl} extension not built; rebuild effectivesource "
                "with `pip install -e ../effectivesource`."
            )
        ref = _ORIGINAL_LIVE.get(impl)
        live = ref() if ref is not None else None
        if live is not None:
            raise RuntimeError(
                f"impl={impl!r} is global state, and an EffectiveSource is "
                f"already alive with mass={live.mass}, spin={live.spin}. "
                "Release it before creating another."
            )
        self.impl, self.lib = impl, lib
        self.mass, self.spin = mass, spin
        lib.effsource_init(mass, spin)
        _ORIGINAL_LIVE[impl] = weakref.ref(self)

    def set_particle(self, x_p, E, L, ur):
        """(coordinate, E, L, ur) -> None."""
        self.lib.effsource_set_particle(x_p, E, L, ur)

    def calc_PhiS(self, x):
        """coordinate -> PhiS."""
        buf = self.lib.doubleArray(1)
        self.lib.effsource_PhiS(x, buf.cast())
        return buf[0]

    def calc_PhiS_m(self, m, x):
        """(m, coordinate) -> (Re, Im)."""
        buf = self.lib.doubleArray(2)
        self.lib.effsource_PhiS_m(m, x, buf.cast())
        return buf[0], buf[1]

    def calc(self, x):
        """coordinate -> (PhiS, dPhiS[4], d2PhiS[10], src)."""
        _PhiS = self.lib.doubleArray(1)
        _dPhiS = self.lib.doubleArray(4)
        _d2PhiS = self.lib.doubleArray(10)
        _src = self.lib.doubleArray(1)
        self.lib.effsource_calc(
            x, _PhiS.cast(), _dPhiS.cast(), _d2PhiS.cast(), _src.cast()
        )
        return (
            _PhiS[0],
            [_dPhiS[i] for i in range(4)],
            [_d2PhiS[i] for i in range(10)],
            _src[0],
        )

    def calc_m(self, m, x):
        """(m, coordinate) -> (PhiS[2], dPhiS[8], d2PhiS[20], src[2])."""
        _PhiS = self.lib.doubleArray(2)
        _dPhiS = self.lib.doubleArray(8)
        _d2PhiS = self.lib.doubleArray(20)
        _src = self.lib.doubleArray(2)
        self.lib.effsource_calc_m(
            m, x, _PhiS.cast(), _dPhiS.cast(), _d2PhiS.cast(), _src.cast()
        )
        return (
            [_PhiS[i] for i in range(2)],
            [_dPhiS[i] for i in range(8)],
            [_d2PhiS[i] for i in range(20)],
            [_src[i] for i in range(2)],
        )


class EffectiveSource:
    """High-level wrapper around the effsource SWIG modules.

    Parameters
    ----------
    spin : float
        Black hole spin parameter (a/M).
    mass : float
        Black hole mass, default 1.0.
    orbit : str
        Orbit geometry: "equatorial" or "circular".
    impl : str
        Which implementation of that geometry: "refactored" (the context-based
        code in use), "original" (as published at upstream 07a31ce), or
        "original_agm" (original with K and E evaluated from the complementary
        parameter by AGM). The two original impls are global state, one live
        instance of each per process.
    """

    def __init__(self, mass=1.0, spin=0.0, orbit="equatorial",
                 impl="refactored", **kwargs):
        if "mode" in kwargs:
            raise TypeError(
                "EffectiveSource no longer takes mode=; it took two unrelated "
                "axes at once. Use orbit= for the geometry "
                f"({' / '.join(ORBITS)}) and impl= for the implementation "
                f"({' / '.join(IMPLS)}). mode= now means only the quadrature "
                "selector on the integrate_nmode routines."
            )
        if orbit not in ORBITS:
            raise ValueError(f"orbit must be one of {ORBITS}, got {orbit!r}")
        if impl not in IMPLS:
            raise ValueError(f"impl must be one of {IMPLS}, got {impl!r}")

        self.orbit = orbit
        self.impl = impl
        self.mass = mass
        self.spin = spin

        if impl != "refactored" and orbit != "equatorial":
            raise NotImplementedError(
                f"orbit={orbit!r} has no {impl!r} build. The original circular "
                "code is effectivesource/kerr-circular.c, which no extension "
                "currently compiles."
            )

        if impl != "refactored":
            self._ef = _OriginalAdapter(impl, mass, spin)
            self._ef.lib.disable_gsl_error_handler()
        elif orbit == "circular":
            self._ef = EffsourceContext(mass, spin)
            disable_gsl_error_handler()
        else:
            self._ef = EffsourceEquatorialContext(mass, spin)
            disable_gsl_error_handler()

    @staticmethod
    def make_coordinate(t, r, theta, phi):
        """Create a coordinate struct. Uses whichever effsource module is available."""

        return make_coordinate(t=t, r=r, theta=theta, phi=phi)

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
        xp = self.make_coordinate(t=0.0, r=r, theta=theta, phi=phi)
        self._ef.set_particle(xp, energy, lz, ur)

    def phi_s(self, r_field, theta_field, phi_field):
        """Compute the puncture at a field point.

        Parameters
        ----------
        r_field, theta_field, phi_field : float
            Field point position (t=0).

        Returns
        -------
        float
        """
        xf = self.make_coordinate(t=0.0, r=r_field, theta=theta_field, phi=phi_field)
        PhiS = self._ef.calc_PhiS(xf)

        return PhiS

    def phi_s_m(self, m, r_field, theta_field):
        """Compute the puncture at a field point.

        Parameters
        ----------
        m : int
            Azimuthal mode number.
        r_field, theta_field : float
            Field point position (t=0, phi=0).

        Returns
        -------
        PhiS : ndarray shape (2,) — [Re, Im] singular field
        """
        xf = self.make_coordinate(t=0.0, r=r_field, theta=theta_field, phi=0.0)
        PhiS_m = self._ef.calc_PhiS_m(m, xf)

        return PhiS_m

    def phi_s_m_offset(self, m, dr, dtheta, branch="auto"):
        """Compute the m-mode puncture at an offset from the particle.

        Parameters
        ----------
        m : int
            Azimuthal mode number. The "ei_table" branch caps at m = 20;
            "legendre" has no ceiling.
        dr, dtheta : float
            Field point offset from the particle (r - r_p, theta - theta_p).
        branch : str
            "auto", "ei_table" or "legendre". "auto" picks "legendre" once
            C1 = alpha/beta exceeds 0.5 m^-1.2, the median crossing of the two
            branches' error curves.

        Returns
        -------
        (Re, Im, branch_used)
        """
        if self.orbit != "equatorial" or self.impl != "refactored":
            raise NotImplementedError(
                f"phi_s_m_offset needs orbit='equatorial', impl='refactored'; "
                f"got orbit={self.orbit!r}, impl={self.impl!r}."
            )
        from .mmode import phi_s_m

        return phi_s_m(self._ef, m, dr, dtheta, branch=branch)

    def calc_m_offset(self, m, dr, dtheta, branch="auto"):
        """Compute the m-mode puncture, derivatives and source at an offset.

        Parameters
        ----------
        m : int
            Azimuthal mode number. The "ei_table" branch caps at m = 20.
        dr, dtheta : float
            Field point offset from the particle (r - r_p, theta - theta_p).
        branch : str
            "auto", "ei_table" or "legendre". "auto" picks "legendre" once
            C1 = alpha/beta exceeds 0.45 m^-1.1, the median crossing of the two
            branches' error curves.

        Returns
        -------
        dict with keys PhiS (2,), dPhiS (4, 2), d2PhiS (10, 2), src (2,),
        branch (str). Mixed second derivatives are NAN, as in the C.
        """
        if self.orbit != "equatorial" or self.impl != "refactored":
            raise NotImplementedError(
                f"calc_m_offset needs orbit='equatorial', impl='refactored'; "
                f"got orbit={self.orbit!r}, impl={self.impl!r}."
            )
        from .mmode import calc_m

        PhiS, dPhiS, d2PhiS, src, used = calc_m(self._ef, m, dr, dtheta,
                                                branch=branch)
        return {
            "PhiS": np.array(PhiS),
            "dPhiS": np.array(dPhiS).reshape(4, 2),
            "d2PhiS": np.array(d2PhiS).reshape(10, 2),
            "src": np.array(src),
            "branch": used,
        }

    def source(self, r_field, theta_field, phi_field):
        """Compute the effective source at a field point.

        Parameters
        ----------
        r_field, theta_field, phi_field : float
            Field point position (t=0).

        Returns
        -------
        float
        """
        xf = self.make_coordinate(t=0.0, r=r_field, theta=theta_field, phi=phi_field)
        _, _, _, src = self._ef.calc(xf)

        return src

    def source_m(self, m, r_field, theta_field):
        """Compute the effective source at a field point.

        Parameters
        ----------
        m : int
            Azimuthal mode number.
        r_field, theta_field : float
            Field point position (t=0, phi=0).

        Returns
        -------
        PhiS : ndarray shape (2,) — [Re, Im] singular field
        """
        xf = self.make_coordinate(t=0.0, r=r_field, theta=theta_field, phi=0.0)
        _, _, _, src = self._ef.calc_m(m, xf)

        return src

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
        xf = self.make_coordinate(t=0.0, r=r_field, theta=theta_field, phi=0.0)
        PhiS, dPhiS, d2PhiS, src = self._ef.calc_m(m, xf)

        return {
            "PhiS": np.array(PhiS),
            "dPhiS": np.array(dPhiS).reshape(4, 2),
            "d2PhiS": np.array(d2PhiS).reshape(10, 2),
            "src": np.array(src),
        }

    def calc(self, r_field, theta_field, phi_field):
        """Compute the full (non-decomposed) effective source at a field point.

        Parameters
        ----------
        r_field, theta_field, phi_field : float
            Field point position (t=0).

        Returns
        -------
        dict with keys:
            PhiS : float — singular field
            dPhiS : ndarray shape (4,) — first derivatives [t, r, theta, phi]
            d2PhiS : ndarray shape (10,) — second derivatives
            src : float — effective source
        """
        xf = self.make_coordinate(t=0.0, r=r_field, theta=theta_field, phi=phi_field)
        PhiS, dPhiS, d2PhiS, src = self._ef.calc(xf)

        return {
            "PhiS": PhiS,
            "dPhiS": np.array(dPhiS),
            "d2PhiS": np.array(d2PhiS),
            "src": src,
        }
