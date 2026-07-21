"""Kernel-factored n-modes of the m-mode effective source.

Every calc_m output splits exactly (effectivesource calc_m_split) into seven
channels

    X_m(t) = A(t) + L(t) ln a(t) + sum_{q=1..5} Pq(t) / a(t)^q,
    a(t)   = alpha20(t) dr(t)^2 + alpha02(t) dtheta^2,

with A, L, P1..P5 analytic (narrow) along the worldline at a legal field point
(the P-side hidden poles are extracted in closed form; see
docs/nmode-kernel-factorization.md sec 6).  The wide-spectrum content lives in
six universal scalar kernels {ln a, 1/a, 1/a^2, ..., 1/a^5} that cost only orbit
quantities.  The n-modes follow by convolution

    X_n = A_hat(n) + (L_hat * Kln)(n) + sum_q (Pq_hat * Kq)(n),

Kln, Kq = FFTs of {ln a, 1/a^q}.  Truth for comparison: Inspectre.panel_nmodes_fast
(PANEL_GL).  Because recombination is exact and convolution is linear, X_n equals
the true n-mode regardless of the (non-unique) A<->Pq split.

This module lifts the ad-hoc cells of notebooks/kernel_factorization_toy.ipynb
into reusable functions, now consuming the clean 7-channel calc_m_split.
"""
import math

import numpy as np

# output name -> (calc_m_split block index, Re component, Im component)
OUTPUTS = {"PhiS": (0, 0, 1), "dPhiS_r": (1, 2, 3), "src": (3, 0, 1)}
# panel_nmodes_fast tuple layout is (PhiS, dPhiS, src) -- note src index 2 not 3
PANEL = {"PhiS": (0, 0, 1), "dPhiS_r": (1, 2, 3), "src": (2, 0, 1)}
CHANNELS = ("A", "L", "P1", "P2", "P3", "P4", "P5")


def orbit_quantities(insp):
    """(Tr, omega_r, omega_phi) for the current orbit."""
    return insp.t_from_lambda(insp.mino_period_r), insp.omega_r, insp.omega_phi


def _worldline(insp, t_grid):
    for t in t_grid:
        lam = insp.lambda_from_t(t)
        yield (t, insp.r_from_lambda(lam), insp.phi_from_lambda(lam),
               insp.four_velocity_equatorial(lam))


def extract_channels(insp, m, r_f, th_f, NB=4096, output="src"):
    """Sample the seven split channels of ``output`` on a uniform-t grid.

    Returns a dict with the fold-applied channel series ``chan[name]`` (name in
    CHANNELS), orbit series (r_p, a20, a02, dr_base, alpha_base) and scalars.
    """
    Tr, om_r, om_ph = orbit_quantities(insp)
    dth = th_f - math.pi / 2
    blk, cre, cim = OUTPUTS[output]
    t_grid = np.arange(NB) * (Tr / NB)
    chan = {c: np.empty(NB, dtype=complex) for c in CHANNELS}
    r_p = np.empty(NB)
    a20 = np.empty(NB)
    a02 = np.empty(NB)
    for j, (t, rp, phip, ur) in enumerate(_worldline(insp, t_grid)):
        insp.set_particle(rp, math.pi / 2, phip, ur)
        blocks = insp.es._ef.calc_m_split(m, r_f - rp, dth)
        a20[j], a02[j], _, _ = insp.es._ef.get_alpha()
        r_p[j] = rp
        fold = np.exp(1j * m * om_ph * t)
        for ch, name in enumerate(CHANNELS):
            chan[name][j] = fold * complex(blocks[blk][ch][cre],
                                           blocks[blk][ch][cim])
    dr_base = r_f - r_p
    alpha_base = a20 * dr_base ** 2 + a02 * dth * dth
    return dict(t_grid=t_grid, r_p=r_p, a20=a20, a02=a02, dr_base=dr_base,
                alpha_base=alpha_base, dth=dth, Tr=Tr, om_r=om_r, om_ph=om_ph,
                chan=chan, output=output, m=m, r_f=r_f, th_f=th_f, NB=NB)


def _resample_real(x, n_dense):
    """Spectral (zero-pad) resample of a real periodic series to n_dense."""
    xh = np.fft.fft(x)
    out = np.zeros(n_dense, dtype=complex)
    h = len(x) // 2
    out[:h] = xh[:h]
    out[-h:] = xh[-h:]
    return np.real(np.fft.ifft(out)) * (n_dense / len(x))


def build_kernels(R, NK=1 << 20):
    """FFT coefficients of the six scalar kernels {ln a, 1/a, ..., 1/a^5}.

    Uses the orbit series in ``R`` (r_p, a20, a02) spectrally resampled to NK.
    Returns dict with K[name] (name in {L, P1..P5}, length NK) and NK, alpha_d.
    """
    r_pd = _resample_real(R["r_p"], NK)
    a20d = _resample_real(R["a20"], NK)
    a02d = _resample_real(R["a02"], NK)
    alpha_d = a20d * (R["r_f"] - r_pd) ** 2 + a02d * R["dth"] ** 2
    K = {"L": np.fft.ifft(np.log(alpha_d))}
    inv = 1.0 / alpha_d
    p = inv.copy()
    for q in range(1, 6):
        K[f"P{q}"] = np.fft.ifft(p)
        p = p * inv
    return dict(K=K, NK=NK, alpha_d=alpha_d)


def nmodes_by_convolution(R, kernels, n_list, KG=400):
    """n-modes X_n by the seven-channel convolution (A direct + L,P1..P5)."""
    NB, NK = R["NB"], kernels["NK"]
    K = kernels["K"]
    c = {name: np.fft.ifft(R["chan"][name]) for name in CHANNELS}
    ks = np.arange(-KG, KG + 1)
    cc = {name: c[name][ks % NB] for name in ("L", "P1", "P2", "P3", "P4", "P5")}

    def one(n):
        s = c["A"][n % NB]
        for name in ("L", "P1", "P2", "P3", "P4", "P5"):
            s += np.sum(cc[name] * K[name][(n - ks) % NK])
        return s

    return np.array([one(int(n)) for n in n_list])


def panel_truth(insp, m, r_f, th_f, n_list, output="src"):
    """PANEL_GL reference amplitudes for ``output`` over n_list."""
    blk, cre, cim = PANEL[output]
    res = dict(zip(n_list, insp.panel_nmodes_fast(m, list(n_list), r_f, th_f)))
    return np.array([complex(res[n][blk][cre], res[n][blk][cim])
                     for n in n_list])


def compare_to_panel(insp, m, r_f, th_f, n_list, output="src", NB=4096,
                     NK=1 << 20, KG=400):
    """Convolution vs PANEL_GL for ``output``; returns a report dict.

    Keys: n_list, X_conv, X_panel, rel, rel_le64, rel_all, R, kernels.
    """
    R = extract_channels(insp, m, r_f, th_f, NB=NB, output=output)
    kernels = build_kernels(R, NK=NK)
    X_conv = nmodes_by_convolution(R, kernels, n_list, KG=KG)
    X_pan = panel_truth(insp, m, r_f, th_f, n_list, output=output)
    rel = np.abs(X_conv - X_pan) / np.maximum(np.abs(X_pan), 1e-300)
    nl = np.asarray(n_list)
    return dict(n_list=nl, X_conv=X_conv, X_panel=X_pan, rel=rel,
                rel_le64=float(rel[nl <= 64].max()),
                rel_all=float(rel.max()), R=R, kernels=kernels)
