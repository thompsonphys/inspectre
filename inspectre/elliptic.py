# ellipitic coordinates for field sampling
from numpy import cosh, sinh, cos, sin, arange, pi, array


def elliptic_to_cartesian(semimajor_axis, mu, nu, x_shift=0.0):
    # note these are skewed in y to adjust for normalization
    # of \cos\theta, so not real elliptic...

    x = semimajor_axis * cosh(mu) * cos(nu) + x_shift
    y = sinh(mu) * sin(nu) / semimajor_axis

    return x, y


def ray_sampling(p, e, delta_mu=0.05, delta_nu=0.05):

    # let's sample rays in elliptic coordinates that approach the radial
    # libration region along hyperbolae, why not
    # the "Cartesian" coordinates are (r, \cos\theta) in SpECTRE

    r_min = p / (1.0 + e)
    r_max = p / (1.0 - e)

    r_mid = (r_min + r_max) / 2.0
    r_width = r_max - r_min
    semimajor_axis = r_width / 2.0

    rays_in_mu_nu = [
        (mu, nu)
        for mu in arange(0.01, 0.7, delta_mu)
        for nu in arange(0, 2.0 * pi, delta_nu)
    ]

    rays_in_x_y = [
        elliptic_to_cartesian(semimajor_axis, mu, nu, x_shift=r_mid)
        for mu, nu in rays_in_mu_nu
    ]

    return array(rays_in_x_y)
