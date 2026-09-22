"""Kerr equatorial separatrix, for keeping orbit sweeps clear of plunge."""

import math


def r_isco(a):
    """spin -> ISCO radius; a > 0 prograde, a < 0 retrograde."""
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0)
                                               + (1.0 - a) ** (1.0 / 3.0))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    s = -1.0 if a >= 0.0 else 1.0
    return 3.0 + z2 + s * math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


def r_mb(a):
    """spin -> marginally bound circular radius."""
    return 2.0 - a + 2.0 * math.sqrt(1.0 - a)


def p_sep(a, e):
    """(spin, eccentricity) -> separatrix semi-latus rectum.

    Exact at e = 0 (ISCO) and e = 1 (twice the marginally bound radius), linear
    in e between; the true separatrix is slightly convex, so this reads a little
    low at intermediate e and callers should apply a margin.
    """
    return r_isco(a) + e * (2.0 * r_mb(a) - r_isco(a))


def is_bound(a, p, e, margin=1.15, pad=0.3):
    """(spin, p, eccentricity) -> True if p clears p_sep by margin*p_sep + pad."""
    return p >= margin * p_sep(a, e) + pad
