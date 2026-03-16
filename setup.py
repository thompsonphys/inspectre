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

# Copy the pre-generated SWIG .py files from submodule dirs to root so
# setuptools can find them as py-modules.
for _name, _src_dir in [
    ("kerrgeodesics.py", kerr_abs),
    ("effsource_circular.py", effs_abs),
    ("effsource_equatorial.py", effs_abs),
]:
    _src = os.path.join(_src_dir, _name)
    _dst = os.path.join(_here, _name)
    if os.path.exists(_src):
        shutil.copy2(_src, _dst)

kerrgeodesics_ext = Extension(
    "_kerrgeodesics",
    sources=[
        os.path.join(kerr_rel, "kerrgeodesics_wrap.c"),
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
        os.path.join(effs_rel, "effsource_circular_wrap.c"),
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
        os.path.join(effs_rel, "effsource_equatorial_wrap.c"),
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
        src = os.path.join(kerr_abs, "kerrgeodesics.py")
        dst = os.path.join(_here, "kerrgeodesics.py")
        if os.path.exists(src):
            shutil.copy2(src, dst)


setup(
    ext_modules=[kerrgeodesics_ext, effsource_circular_ext, effsource_equatorial_ext],
    cmdclass={"build_ext": CustomBuildExt},
)
