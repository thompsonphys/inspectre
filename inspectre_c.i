%module inspectre_c

/* Low-level SWIG bindings for the inspectre C library (src/inspectre_lib.c).

   These functions take pointers to types owned by the two sibling SWIG modules:
     - korb_params *                 (module: kerrgeodesics)
     - struct effsource_equatorial_ctx *, struct coordinate *  (module: effsource_equatorial)
   SWIG's shared Python runtime type table (all three modules build with the same
   SWIG runtime version) makes those pointers interchangeable across the
   independently-built extensions, so objects created by KerrOrbit /
   EffsourceEquatorialContext pass straight through here. */

%{
/* inspectre.h transitively includes korb.h + effsource{,_equatorial}.h via its
   own relative #include lines; including only it avoids double-including korb.h
   (which has no include guard). */
#include "inspectre.h"
%}

%include "carrays.i"
%array_class(double, doubleArray);

/* korb.h declares `double complex` prototypes; neutralize the C99 keyword for
   SWIG's parser exactly as kerrgeodesics.i does (this directive is not emitted
   into the generated C, so the real <complex.h> is still used at compile time). */
#define complex

/* Learn the foreign types without regenerating their proxy classes. effsource.h
   must precede effsource_equatorial.h (defines struct coordinate). */
%import "korb.h"
%import "effsource.h"
%import "effsource_equatorial.h"

/* GSL-internal integrand callbacks (void *params) — not useful from Python. */
%ignore inspectre_nmode_params;
%ignore inspectre_nmode_integrand;
%ignore inspectre_nmode_integrand_lambda;

/* Wrap the inspectre public API. */
%include "inspectre.h"

/* ------------------------------------------------------------------------- *
 * Pythonic convenience wrappers that hide the doubleArray buffers.          *
 * ------------------------------------------------------------------------- */
%pythoncode %{

def _to_array(seq):
    """Build a doubleArray from a Python sequence (None -> (None, None))."""
    if seq is None:
        return None, None
    n = len(seq)
    a = doubleArray(n)
    for i, v in enumerate(seq):
        a[i] = v
    return a, a.cast()


def _unpack(PhiS, dPhiS, ddPhiS, src):
    return (
        [PhiS[0], PhiS[1]],
        [dPhiS[i] for i in range(8)],
        [ddPhiS[i] for i in range(20)],
        [src[0], src[1]],
    )


def make_field_point(r, dtheta, phi=0.0):
    """Field point carrying dtheta = theta - pi/2 directly.

    Pass the result anywhere xField is accepted; the wrappers dispatch to the
    _fp entry points. Routing dtheta through an absolute theta instead caps its
    relative accuracy at 1.1e-16/dtheta.
    """
    fp = inspectre_field_point()
    fp.r = r
    fp.dtheta = dtheta
    fp.phi = phi
    return fp


def _fp_pick(xField, fnAbs, fnFp):
    """Select the absolute-coordinate or offset entry point for this field point."""
    return fnFp if isinstance(xField, inspectre_field_point) else fnAbs


def eval_gold_at_lambda(ctx, mMode, fp, lam, orbpar, a, p, e):
    """Reference evaluator: split channels reassembled in long double.

    Requires an inspectre_field_point. Returns (PhiS[2], dPhiS[8], src[2]).
    """
    PhiS = doubleArray(2); dPhiS = doubleArray(8); src = doubleArray(2)
    inspectre_eval_gold_at_lambda_fp(ctx, mMode, fp, lam, orbpar, a, p, e,
                                     PhiS.cast(), dPhiS.cast(), src.cast())
    return ([PhiS[0], PhiS[1]],
            [dPhiS[i] for i in range(8)],
            [src[0], src[1]])


def eval_at_lambda(ctx, mMode, xField, lam, orbpar, a, p, e):
    """Effective source + puncture at Mino time lambda.

    Returns (PhiS[2], dPhiS[8], ddPhiS[20], src[2]) as Python lists.
    """
    PhiS = doubleArray(2); dPhiS = doubleArray(8)
    ddPhiS = doubleArray(20); src = doubleArray(2)
    _f = _fp_pick(xField, inspectre_eval_at_lambda, inspectre_eval_at_lambda_fp)
    _f(ctx, mMode, xField, lam, orbpar, a, p, e,
       PhiS.cast(), dPhiS.cast(), ddPhiS.cast(), src.cast())
    return _unpack(PhiS, dPhiS, ddPhiS, src)


def eval_at_time(ctx, mMode, xField, t, orbpar, a, p, e):
    """Effective source + puncture at coordinate time t (Brent-inverts lambda(t)).

    Returns (PhiS[2], dPhiS[8], ddPhiS[20], src[2]) as Python lists.
    """
    PhiS = doubleArray(2); dPhiS = doubleArray(8)
    ddPhiS = doubleArray(20); src = doubleArray(2)
    _f = _fp_pick(xField, inspectre_eval_at_time, inspectre_eval_at_time_fp)
    _f(ctx, mMode, xField, t, orbpar, a, p, e,
       PhiS.cast(), dPhiS.cast(), ddPhiS.cast(), src.cast())
    return _unpack(PhiS, dPhiS, ddPhiS, src)


def integrate_nmode(mode, ctx, mMode, nMode, xField, orbpar, a, p, e,
                    omegaPhi, omegaR, epsabs=1e-10, epsrel=1e-10,
                    tSamples=None, fieldSamples=None, derivSamples=None,
                    srcSamples=None, nSamples=0):
    """n-mode Fourier amplitude over one radial period.

    `mode` is one of the INSPECTRE_INTEG_* constants. The self-contained modes
    (QAG, QAG_MINO, MINO_SPLINE) ignore the sample arrays — leave them None and
    pass nSamples only for MINO_SPLINE node count. The sample-based modes
    (TIMESERIES, SIMPSON, SPLINE) require tSamples + interleaved
    fieldSamples[2*N], derivSamples[8*N], srcSamples[2*N].

    Returns (nModePhiS[2], nModeDPhiS[8], nModesrc[2]) as Python lists.
    """
    _t, tP = _to_array(tSamples)
    _f, fP = _to_array(fieldSamples)
    _d, dP = _to_array(derivSamples)
    _s, sP = _to_array(srcSamples)
    nPhiS = doubleArray(2); nDPhiS = doubleArray(8); nsrc = doubleArray(2)
    _f = _fp_pick(xField, inspectre_integrate_nmode, inspectre_integrate_nmode_fp)
    _f(mode, ctx, mMode, nMode, xField, orbpar, a, p, e,
       omegaPhi, omegaR, epsabs, epsrel,
       tP, fP, dP, sP, nSamples,
       nPhiS.cast(), nDPhiS.cast(), nsrc.cast())
    return ([nPhiS[0], nPhiS[1]],
            [nDPhiS[i] for i in range(8)],
            [nsrc[0], nsrc[1]])


def fft_source_nmodes(ctx, mMode, xField, orbpar, a, p, e, omegaPhi, omegaR, N):
    """All-n source amplitudes from one FFTW transform.

    Returns (outRe[N], outIm[N]); bin k holds mode n = k if k<=N/2 else k-N.
    """
    outRe = doubleArray(N); outIm = doubleArray(N)
    _f = _fp_pick(xField, inspectre_fft_source_nmodes,
                  inspectre_fft_source_nmodes_fp)
    _f(ctx, mMode, xField, orbpar, a, p, e,
       omegaPhi, omegaR, N, outRe.cast(), outIm.cast())
    return [outRe[i] for i in range(N)], [outIm[i] for i in range(N)]


def mino_samples_build(ctx, mMode, xField, orbpar, a, p, e, nSamples):
    """Build the reusable n-independent Mino-time graded-mesh samples.

    Returns an inspectre_mino_samples handle; free it with mino_samples_free.
    """
    s = inspectre_mino_samples()
    _f = _fp_pick(xField, inspectre_mino_samples_build,
                  inspectre_mino_samples_build_fp)
    _f(ctx, mMode, xField, orbpar, a, p, e, nSamples, s)
    return s


def mino_samples_integrate(s, mMode, nMode, omegaPhi, omegaR):
    """Frequency-shift prebuilt samples for mode n and spline-integrate.

    Returns (nModePhiS[2], nModeDPhiS[8], nModesrc[2]) as Python lists.
    """
    nPhiS = doubleArray(2); nDPhiS = doubleArray(8); nsrc = doubleArray(2)
    inspectre_mino_samples_integrate(s, mMode, nMode, omegaPhi, omegaR,
                                     nPhiS.cast(), nDPhiS.cast(), nsrc.cast())
    return ([nPhiS[0], nPhiS[1]],
            [nDPhiS[i] for i in range(8)],
            [nsrc[0], nsrc[1]])


def mino_samples_free(s):
    inspectre_mino_samples_free(s)


def panel_nodes_build(ctx, mMode, xField, orbpar, a, p, e, omegaPhi, omegaR,
                      order=16, maxLevels=40, nMax=64):
    """Build panel-GL nodes split at the closest-approach breakpoints.

    nMax is the largest |n| the node set must resolve. Returns an
    inspectre_panel_nodes handle; free it with panel_nodes_free.
    """
    s = inspectre_panel_nodes()
    _f = _fp_pick(xField, inspectre_panel_nodes_build,
                  inspectre_panel_nodes_build_fp)
    _f(ctx, mMode, xField, orbpar, a, p, e,
       order, maxLevels, nMax, omegaPhi, omegaR, s)
    return s


def panel_nodes_integrate(s, mMode, nMode, omegaPhi, omegaR):
    """Frequency-shift prebuilt panel samples for mode n and sum the quadrature.

    Returns (nModePhiS[2], nModeDPhiS[8], nModesrc[2]) as Python lists.
    """
    nPhiS = doubleArray(2); nDPhiS = doubleArray(8); nsrc = doubleArray(2)
    inspectre_panel_nodes_integrate(s, mMode, nMode, omegaPhi, omegaR,
                                    nPhiS.cast(), nDPhiS.cast(), nsrc.cast())
    return ([nPhiS[0], nPhiS[1]],
            [nDPhiS[i] for i in range(8)],
            [nsrc[0], nsrc[1]])


def panel_nodes_free(s):
    inspectre_panel_nodes_free(s)


def fact_nodes_build(ctx, mMode, xField, orbpar, a, p, e, omegaPhi, omegaR,
                     NB=4096, NK=1 << 20, KG=400, nMax=512):
    """Build the kernel-factorization channel/kernel FFT coefficients.

    NB channel-split source evaluations (the only source-evaluating step);
    kernels resolve |n| <= nMax. Returns an inspectre_fact_nodes handle; free
    it with fact_nodes_free.
    """
    s = inspectre_fact_nodes()
    _f = _fp_pick(xField, inspectre_fact_nodes_build,
                  inspectre_fact_nodes_build_fp)
    _f(ctx, mMode, xField, orbpar, a, p, e,
       NB, NK, KG, nMax, omegaPhi, omegaR, s)
    return s


def fact_nodes_integrate(s, nMode):
    """n-mode amplitudes for mode n by the seven-channel convolution.

    Returns (nModePhiS[2], nModeDPhiS[8], nModesrc[2]) as Python lists.
    """
    nPhiS = doubleArray(2); nDPhiS = doubleArray(8); nsrc = doubleArray(2)
    inspectre_fact_nodes_integrate(s, nMode,
                                   nPhiS.cast(), nDPhiS.cast(), nsrc.cast())
    return ([nPhiS[0], nPhiS[1]],
            [nDPhiS[i] for i in range(8)],
            [nsrc[0], nsrc[1]])


def fact_nodes_free(s):
    inspectre_fact_nodes_free(s)
%}
