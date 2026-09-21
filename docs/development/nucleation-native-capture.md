# Active native nucleation capture

The checked Fortran box driver now provides an active, full-precision reference
for nested nucleation and its enclosing condensation call. This closes the
single-case F13c.N1–N3/N08b gate. It does not complete the JAX SALSA timestep or
validate the full box-model trajectory.

## Explicit first-call profile

`configs/fortran-nucleation-active.toml` retains the checked compiler flags and
adds these settings:

```toml
[reference_case]
nsnucl = 2
nj3 = 1
sulfate_molecules_m3 = 1.0e14
```

The standalone build excludes HAMMOZ's namelist reader, so editing the SALSA
namelist would not activate this configuration. `reference_configuration.py`
sets the switches before `start_ham` in the **copied** driver and overrides
only the first-step sulfate concentration before the existing m⁻³→cm⁻³ and
tracer conversions. All other initialization and forcing are retained. The
harness restricts this profile to `--scope first-microphysics` and verifies the
requested switches against effective runtime metadata.

The default first-call H2SO4 concentration is only 4.0445e-10 molecules/m³,
below the activation cutoff. The explicit 1e14 concentration produces a useful
nonzero source and insertion. This is a controlled test case, not the default
driver trajectory. Production Fortran sources and scientific input files remain
unchanged. Each run records the profile, original/configured driver hashes,
configuration patch, compiler, source/input hashes and subsequent capture patches.

## Regeneration and invariance checks

Use this repository's `.venv`, GNU Fortran and NetCDF C/Fortran libraries. Each
output directory must be new:

```sh
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-nucleation-active.toml --scope first-microphysics \
  --run --output build/active-control-new
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-nucleation-active.toml --scope first-microphysics \
  --capture-salsa-state --capture-nucleation-state \
  --run --output build/active-capture-new
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-nucleation-active.toml --scope first-microphysics \
  --capture-salsa-state --capture-nucleation-state \
  --run --output build/active-repeat-new
```

`--capture-nucleation-state` automatically captures enclosing condensation and
runtime metadata. It requires an explicit profile. Nested files record actual
arguments, initialized and returned source arrays, fractions, gas indices,
boundary-layer levels, species properties and constants immediately around the
original call. Parent files bracket uptake and particle insertion. Hydration
and remapping still execute in Fortran around these boundaries.

All three recorded runs complete the strict first-microphysics scope. Control,
capture and repeat have byte-identical `scope-state.txt` (all six HAM arrays),
runtime metadata and ordinary diagnostics. The two captured runs also reproduce
native SALSA, nested nucleation and parent condensation files exactly.
The [evidence record](evidence/nucleation-native-2026-09-21.json) retains these
hash comparisons; executable success alone is not sufficient acceptance.

Export reviewed fixtures after those comparisons:

```sh
.venv/bin/python tools/nucleation_capture.py \
  --run-directory build/active-capture-new --output build/nucleation-fixture-new
.venv/bin/python tools/condensation_capture.py \
  --run-directory build/active-capture-new --output build/condensation-fixture-new
.venv/bin/python -m pytest -q tests/test_nucleation_native.py
```

The committed pair lives in `tests/fixtures/nucleation-active/` and
`tests/fixtures/condensation-active/`. Export requires completed checked capture
and manifest-matching raw hashes; loading also verifies file hashes.

## Native acceptance

The adapter reconstructs Fortran record order, checks agreement between nested
and parent inputs, validates the fixed constants, pads the absent organic gas,
and constructs the native `jj >= ppbl` mask. The fixture has one horizontal
cell and one boundary-layer level; a separate 2×2 adapter test checks ordering
and boundary masks. All nine scheme/growth combinations remain covered by the
independent process matrix, not by this one activated driver fixture.

The actual captured molecular source is **1.6066412692623088e8 molecules/m³/s**;
condensation inserts **1.0077747032660246e6 particles/m³** into bin 1. H2SO4
falls from 1e14 to **9.96019137520891e13 molecules/m³**. Nested nucleation itself
leaves its input aerosol/gas state unchanged. The extracted Fortran oracle is
compared with these captures before eager/JIT JAX comparisons. Species budgets,
finite/nonnegative outputs and unchanged unrelated state also pass.

The full suite passes **384 tests** in the repository environment. No scientific
kernel changed in this milestone, and no new performance claim is made; the
existing [process benchmarks](nucleation-process.md#cpu-benchmark) remain scoped
to their measured workloads. Grid generation, hydration, remapping and complete
native composition are still pending. D009 coagulation volume drift, D007
tiny-particle cleanup and D005 radiation remain separate reference issues.
