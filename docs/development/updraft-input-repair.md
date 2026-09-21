# Updraft input repair (D003)

The driver previously allocated `zvervel(kbdim,klev,nw)`, initialized only
plane `nw`, and passed the entire array to `activ_updraft`, whose
`pvervel` argument has shape `(kbdim,klev)`. With the default 20 PDF bins,
the callee read the uninitialized first plane.

The repair stores a two-dimensional pressure velocity and initializes every
active cell. It preserves the existing input file, variable `W`, and sampled
indices `(85,25,47,1)`.

## Units and remaining scientific choice

Inspect the supplied file with:

```sh
ncdump -h input/GlobalTraj-CE_200801.01_activ_1step.nc
```

Its metadata states `W:units = "m s-1"` and
`W:long_name = "total vertical velocity for activation"`.
The actual `activ_updraft` interface expects pressure velocity in Pa/s and
converts it to m/s using `-pvervel/(grav*prho)`. The driver therefore supplies
`-grav*zrhoa*W`; the PDF receives the original signed velocity independent of
air density.

This retains the existing choice of total `W` as the prescribed PDF mean.
It does **not** resolve whether this box experiment should instead use the
separate `W_LARGE` field and reconstruct turbulence. That is a scientific
configuration decision, not an array-initialization repair.

## Regression evidence

```sh
.venv/bin/python -m pytest tests/test_updraft_input.py -q
```

Eight tests pass. The harness extracts the declaration, allocation, assignment
and call from the actual driver, and the unchanged `activ_updraft` and
`aero_activ_updraft_pdf` routines. Fixture modules provide only module state.
Before assignment it explicitly poisons allocated input storage with signaling
NaNs; compiler local initialization alone does not poison allocated arrays.

The preserved baseline reproduces the poisoned-plane floating-point trap with
20 bins (reported as SIGILL by this ARM64/GNU Fortran installation). The original
one-bin wiring also produces a wrong Gaussian PDF because it misinterprets m/s
as Pa/s. Corrected tests cover three signed/zero input velocities, four different
air densities, and PDF kernels with one and 20 bins. They compare finite outputs
against the analytical Gaussian with the supplied velocity as its mean.
The one-bin fixture still selects the PDF branch; it does not validate the
separate `nactivpdf=0` diagnostic-pointer path.

This isolated repair does not establish full-driver validity. The independently
identified seven-versus-17-bin aerosol-water storage overrun (D004) can corrupt
the activation call even after D003 is fixed.
