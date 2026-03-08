# EffectiveSourceInspectre

Code to generate Kerr geodesics and evaluate the effective source for given field point.

The repositories `KerrGeodesicsC` and `EffectiveSource` are included as submodules from my forks, with minor changes to the originals. The Makefile in the base dir will sort out most of the compilation, but you may need to point the compiler specification somewhere else (currently it uses a CONDA prefix, assuming that's how you have everything installed).

The shell script `quickrun.sh` will pass in a set of paramters to the `inspectre` executable located in `bin/` after things have compiled. Feel free to change the paramters, though note with care the order they are passed into the executable. No error checking is done on this!

Two data files are output into `data/`: `traj_source.dat` has a timeseries of trajectory data, alongside the puncture and effective source evaluated at the prescribed field point. `puncture_derivs.dat` stores the four coordinate derivatives of the puncture field, again evaluated at the field point.