import os
import shutil
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

prefix = os.environ.get("CONDA_PREFIX", "/usr")

_here = os.path.dirname(os.path.abspath(__file__))
kerr_rel = os.path.join("lib", "kerrgeodesics")
effs_rel = os.path.join("lib", "effectivesource")
kerr_abs = os.path.join(_here, kerr_rel)
effs_abs = os.path.join(_here, effs_rel)

kerrgeodesics_ext = Extension(
    "_kerrgeodesics",
    sources=[
        os.path.join(kerr_rel, "kerrgeodesics.i"),
        os.path.join(kerr_rel, "korb.c"),
    ],
    include_dirs=[prefix + "/include", kerr_abs],
    library_dirs=[prefix + "/lib"],
    runtime_library_dirs=[prefix + "/lib"],
    libraries=["m", "gsl", "gslcblas", "fftw3"],
    extra_compile_args=["-std=gnu99", "-O3"],
)

effsource_circular_ext = Extension(
    "_effsource_circular",
    sources=[
        os.path.join(effs_rel, "effsource_circular.i"),
        os.path.join(effs_rel, "kerr-circular.c"),
    ],
    include_dirs=[prefix + "/include", effs_abs],
    library_dirs=[prefix + "/lib"],
    runtime_library_dirs=[prefix + "/lib"],
    libraries=["gsl", "gslcblas", "m"],
    extra_compile_args=["-std=gnu99", "-O3"],
)

effsource_equatorial_ext = Extension(
    "_effsource_equatorial",
    sources=[
        os.path.join(effs_rel, "effsource_equatorial.i"),
        os.path.join(effs_rel, "kerr-equatorial.c"),
        os.path.join(effs_rel, "kerr-equatorial-coeffs.c"),
        os.path.join(effs_rel, "kerr-equatorial-dtcoeffs.c"),
        os.path.join(effs_rel, "kerr-equatorial-dttcoeffs.c"),
    ],
    include_dirs=[prefix + "/include", effs_abs],
    library_dirs=[prefix + "/lib"],
    runtime_library_dirs=[prefix + "/lib"],
    libraries=["gsl", "gslcblas", "m"],
    extra_compile_args=["-std=gnu99", "-O0"],
)


class CustomBuildExt(build_ext):
    """After SWIG generates kerrgeodesics.py next to the .i file, copy it
    back to the project root so the editable install can find it."""

    def run(self):
        super().run()
        for fname in [
            "kerrgeodesics.py",
            "effsource_circular.py",
            "effsource_equatorial.py",
        ]:
            if "kerr" in fname:
                _abs = kerr_abs
            else:
                _abs = effs_abs
            src = os.path.join(_abs, fname)
            dst = os.path.join(_here, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)


setup(
    ext_modules=[kerrgeodesics_ext, effsource_circular_ext, effsource_equatorial_ext],
    cmdclass={"build_ext": CustomBuildExt},
)
