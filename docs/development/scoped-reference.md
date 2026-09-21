# First-call microphysics reference

The full standalone driver remains unvalidated because radiation initialization
and supplied optical tables are incomplete. Core microphysics can be inspected
at a narrower, explicit boundary without changing production Fortran source.

## Run the checked scope

From the repository's own virtual environment:

```sh
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-checked.toml --scope first-microphysics \
  --capture-runtime --run --output build/first-microphysics
```

Use a new output directory every time. The tool copies the checkout and inputs,
records their hashes, and instruments only the copied driver. Optional runtime
metadata is emitted before the time loop. The scope overlay then stops after
`ham_subm_interface` returns for the first timestep, before activation updraft,
radiation, cloud activation or removal.

Before stopping, it requires finite values in all six arrays: tracer state
`pxtm1`, tendencies `pxtte`, wet radius `zrwet`, dry radius `zm6dry`, density
`zrhop`, and aerosol water `zww`. It writes a completion record only after these
checks pass. The manifest distinguishes this result from ordinary full execution;
an executable exit code alone cannot satisfy the completion contract.

The copied-source overlay and hash chain are retained. The source checkout,
original namelist, and full-driver default remain unchanged. Omitting `--scope`
selects the original full-driver run, including its current checked failure.

## What this verifies

This is one completed HAM microphysics interface call using the default case,
with bounds checks, floating-point traps and finite-output checks. Initialization
metadata records the effective process switches, including `nsnucl=0`.

It does **not** validate complete trajectories, optics, activation, deposition,
all possible configurations, or numerical agreement with JAX. It does not update
and advance driver tendencies through a second step. Existing diagnostic files
may contain initialization-only records and are not a scoped trajectory oracle.
The completion status therefore remains explicitly unvalidated scientifically.

The next task is full-precision native SALSA entry/exit capture with species,
tracer, shape and unit contracts, followed by separate coagulation, condensation,
and nucleation captures. Read the [radiation audit](radiation-reference-audit.md)
for why an optics repair is not hidden inside this scope.

The [checked integration record](evidence/first-microphysics-2026-09-21.json)
retains the actual completion marker, metadata, source/input hashes, compiler
flags and original → runtime-overlay → scope-overlay driver hash chain.
