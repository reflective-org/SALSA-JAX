# Radiation reference audit (F09d / D005)

The supplied standalone case cannot currently serve as an optics reference.
The smallest useful next reference is **one checked call to
`ham_subm_interface`**, with full-precision input/output capture and an explicit
stop before activation, radiation and removal. No production source, harness or
existing document was changed for this audit.

## Initialization and data findings

- `mo_ham_init.F90:1062–1101` imports/calls `ham_rad_initialize` only with
  `HAMMOZ`. Both current build profiles define `SALSA` and `NOMPI`, not
  `HAMMOZ`; the driver nevertheless calls `ham_rad` unconditionally.
- The skipped initializer establishes refractive indices, wavelength/lookup
  interpolation parameters, reverse species maps and lookup storage.
  `ham_rad_refrac_volume` consumes `cnr/cni` from this missing initialization.
- Neither supplied optical file is readable by `ncdump -h`:

  | Input | Size | Result |
  |---|---:|---|
  | `input/lut_optical_properties_M7.nc` | 64 bytes | NetCDF: Invalid argument |
  | `input/lut_optical_properties_lw_M7.nc` | 67 bytes | NetCDF: Invalid argument |

  Their CDF signatures do not establish valid datasets; these are truncated
  records, not usable optical tables. The reader expects eight SW variables
  (`sigma/omega/asym/pp180_1/2`) and two LW variables
  (`sigma_1_lw/sigma_2_lw`), on 101 × 201 × 101 points.
- Simply adding initialization would encounter further defects.
  `TM5M7_DATADIR='./input'` is concatenated with filenames without a slash.
  The standalone initializer also needs the LW wavenumbers initialized before
  constructing `lambda`; the current `TM5M7_INIT` call occurs later inside
  the radiation calculation. Local `nr/ni` values are copied before their
  output calculations initialize them.
- Driver optical arrays begin as ones (main outputs) and zeros (diagnostics).
  Those are placeholders, not observations or reference values. The arrays
  appear only in declarations, initialization and the radiation call; no later
  box-model process consumes these optical outputs.

This is not a request to fabricate tables or broaden the current repair into
radiative transfer. Restoring optics requires authentic complete tables, their
provenance and SALSA applicability, initialization-order repairs, and separate
numerical validation.

## Why existing switches do not define the reference

`ldiag_aeropt=.FALSE.` disables only the additional diagnostic-wavelength block;
it does not disable SW/LW calculations. `naerorad=0` alone does not guard the
driver call or reset the `nrad` entries controlling these loops. Setting
`nrad(:)=0` would bypass calculations but leave the initialized optical
placeholders; those must not be reported as modelled optics.

Moreover, reading the HAMCTL namelist is compiled out without `HAMMOZ`.
Editing `namelist.echam` alone is not a supported way to configure this
standalone experiment. Capture effective compiled/runtime values.

## Checked experiments

A fresh working-tree copy at HEAD `0dd52818d421a5148a414a41e86cf8981f6869de`
included the in-progress nucleation gas-index repair recorded in its manifest.
It built successfully with `configs/fortran-checked.toml`; its unmodified-driver run
returned -4 (SIGILL, this platform's floating-point trap). Two copied-driver
experiments retained the same checked flags:

1. **Stop after the first microphysics call:** insert checks immediately after
   `CALL ham_subm_interface(...)`, before the following
   `DO jn=1, nclass` derived-number loop. All entries of `pxtm1`, `pxtte`,
   `zrwet`, `zm6dry`, `zrhop` and `zww` were finite. The explicit success
   marker printed and the process returned zero. This stops before a complete
   driver timestep and before `activ_updraft` or `ham_rad`.
2. **Omit only the radiation call:** the copied default driver completed its
   1000 steps and returned zero. This establishes feasibility of a future
   explicitly optics-excluded case, not scientific trajectory acceptance:
   no per-step full-state, conservation or initial-diagnostic validation was
   added.

Source/input hashes, exact copied-driver patches, build flags, executable/log
hashes and local experiment paths are in
[radiation audit evidence](evidence/radiation-audit-2026-09-21.json).
Neither experiment modifies the checkout's scientific sources.

## Reference scope and subsequent capture

The [first-microphysics scope](scoped-reference.md) implements the proven stop
and finite-state gate in a copied driver. It completes no full driver timestep;
optics, activation and removal are outside the scope. Complete entry/exit state
capture at full precision remains the next task. Retain checked flags and label
the manifest as a scoped microphysics capture, not a completed model run.

The interface returns tendencies as well as state, so retain `pxtm1`,
`pxtte` and `time_step_len`; subsequent comparisons must account for
`pxtm1 + pxtte*time_step_len`. Keep existing pre-call formatted diagnostic
rows out of the oracle: some precede assignment and are not initial-state
evidence. Follow this with process-specific budget tests and entry/exit capture
inside SALSA. Finite output alone is not a conservation or physics validation.
