# Development quickstart

Run commands from the `SALSA-JAX` repository root. Create its own `.venv`; do not
reuse or copy the original SALSA environment. Python 3.11+ is required; the tested
checkpoint uses Python 3.12.12, GNU Fortran 16.1.0 and macOS ARM64.

## Python environment and leaf parity

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest -q
```

GNU Fortran and NetCDF's `ncdump` must be on `PATH` for the full test suite.
The leaf oracle compiles the original
`coagc` function and constants; it does not need NetCDF or a full HAM build.
A missing compiler is an error, not a skipped parity test. Exact package versions
are a tested environment snapshot, not evidence of support for all platforms.

```python
import jax
jax.config.update("jax_enable_x64", True)
from salsa_jax.processes.coagulation import coagulation_coefficient

coefficient = coagulation_coefficient(
    1e-8, 2e-8, 7.85e-22, 8.38e-21, 280.0, 101325.0
)
print(coefficient)  # m^3/s; diameters m, masses kg, T K, pressure Pa
```

The kernel accepts positive physical inputs and supports broadcasting/JIT.
This is a collision coefficient, not a complete aerosol timestep.

## Isolated full Fortran audit

Install Make, GNU Fortran and both NetCDF C/Fortran development libraries. Ensure
`nf-config` and `nc-config` are on `PATH`. The audit tool discovers their flags,
copies scientific inputs, and avoids modifying the root source/output files.

```sh
.venv/bin/python tools/fortran_reference.py --output build/my-reference --run
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-checked.toml --output build/my-checked-reference --run
```

Each output directory must be new. Inspect its `manifest.json`, `build.log`,
`run.log` and diagnostic tables. The manifest records the current working-tree
source bytes as well as HEAD; the command does not claim a dirty tree equals HEAD.
These run wall times include startup/I/O and are **not** kernel benchmarks.

The checked profile now passes the repaired condensation/driver sites and fails
in standalone radiation (D005); see [reference repairs](reference-repairs.md).
Earlier ordinary runs completed but remain unvalidated scientific references;
the D008 initializer repair has since changed the baseline.
Do not disable checks to declare a defect fixed. Optional `--capture-runtime`
records initialized dimensions, switches, gas mapping and real precision, even
when a later process fails; see [runtime capture](runtime-metadata.md).

To inspect only the first microphysics interface call, add
`--scope first-microphysics` to the checked command. This explicit scope checks
six output/state arrays for finite values and stops before radiation, activation
and removal. Its manifest reports a separate completion status; it does not
establish full-driver parity. See [the scoped reference](scoped-reference.md).
Add `--capture-salsa-state` to capture native inputs/outputs, mappings and
geometry at full precision; runtime metadata is then enabled automatically.
See the [capture and fixture guide](native-state-capture.md) for export commands
and the checked default-case evidence.

For the legacy root build, persistent settings belong in `Makefile.in`.
`sh configure` needs `makedepf90` and overwrites `Makefile`; the audit tool uses
the checked-in dependency graph without requiring regeneration.

## First leaf benchmark

```sh
mkdir -p outputs/benchmarks
.venv/bin/python -m benchmarks.benchmark_coagulation \
  --repetitions 1000 --trials 5 > outputs/benchmarks/coagulation.json
```

Run this after tests finish, without competing heavy jobs. It compares original
Fortran and warmed synchronized JAX on the same 1734 cases, records compilation
separately, and checks numerical results. Interpret it only as a CPU leaf result;
thread settings are recorded but not standardized. The [protocol](validation-and-benchmarks.md)
describes later process, end-to-end and ensemble gates.

Keep environments/builds/bulk outputs ignored. Commit small scientific fixtures,
source/config changes and reviewed results with provenance; do not commit raw
run directories. Follow [STATUS.md](../plans/STATUS.md) to select the next task.

## Complete coagulation

The [process guide](coagulation-process.md) documents `coagulation_step`, its
17-bin/five-material inputs, boundary fixture and original-Fortran comparisons.
Supply hydrated wet diameters; hydration is a separate process. Add
`--capture-coagulation-state` to a checked first-microphysics run to regenerate
the real process boundary. D009 volume drift in stiff native cases is retained
and exposed by tests, not repaired by the port.

```sh
.venv/bin/python -m benchmarks.benchmark_coagulation_process --repetitions 1000 --trials 5
```

Use `--single-case` for the captured default only. These timings compare the
complete isolated process, with JAX compilation reported separately; they are
not full-model performance measurements.

## Separate condensation

The [condensation guide](condensation-process.md) documents the canonical gas
slots, runtime species properties, thresholds and `nsnucl=0` scope. It includes
both sulfate and optional nonvolatile organic uptake; the separate nucleation
coupling API is described below. Regenerate its actual boundary with `--capture-condensation-state`
in the checked first-microphysics scope.

```sh
.venv/bin/python -m benchmarks.benchmark_condensation --repetitions 1000 --trials 5
```

Its oracle includes the earlier D001/D002 repairs and uses compiled legacy
constants. Positive organic cases are isolated fixtures, not a new full-driver
gas registration. All benchmarks use the repository's virtual environment.

## Supported nucleation and coupling

See the [nucleation guide](nucleation-process.md) for separate critical-rate,
growth/source and coupled APIs. Enable float64; mark `scheme` and `growth`
static under JIT. Explicit RH and boundary-layer flags augment the existing
condensation inputs. The default driver has nucleation off; active kernel
fixtures do not establish an activated full-driver trajectory.

```sh
.venv/bin/python -m pytest -q tests/test_nucleation_rates.py \
  tests/test_nucleation_sources.py tests/test_nucleation.py \
  tests/test_nucleation_process_oracle.py
.venv/bin/python -m benchmarks.benchmark_nucleation --repetitions 500 --trials 5
```

Run benchmarks after tests finish. Each native case computes two vertical cells;
JAX timing includes the same pair. Retained legacy growth formulas and the
empty-background insertion gate are documented in the process guide.

For a checked active native boundary, use the explicit test profile; default
first-call sulfate is too small to activate nucleation:

```sh
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-nucleation-active.toml --scope first-microphysics \
  --capture-salsa-state --capture-nucleation-state \
  --run --output build/active-nucleation-new
```

The [active capture guide](nucleation-native-capture.md) explains first-call
forcing, capture/control checks, fixture export and supported scope.

## Grid and hydration

The [grid/hydration guide](grid-and-hydration.md) documents `size_grid`,
`inverse_molality`, `equilibrate`, explicit species properties and solubility flags.
Enable float64 and supply number/material volume plus RH and temperature.
The initial hydration capture and composed grid → hydration → coagulation →
condensation tests pass with native `recalc=False`. Remapping and corrections
now extend native-boundary acceptance through both default passes; see below.
Run the full suite in the repository’s `.venv`. The
[current status](../plans/STATUS.md) tracks validation evidence and the remaining
diagnostic conversion, HAM coupling and longer-trajectory gates.

```sh
.venv/bin/python -m pytest -q tests/test_grid.py tests/test_inverse_molality.py \
  tests/test_hydration_oracle.py tests/test_hydration.py tests/test_hydration_native.py
.venv/bin/python -m benchmarks.benchmark_hydration --repetitions 500 --trials 5
```

Run timings after tests finish; add `--single-case` for the curated final-state
case. Add `--capture-hydration-state` to a checked first-microphysics run to
capture the initial native boundary. D010 documents invalid zero-volume inputs;
parity does not repair that native failure. See the guide for capture commands,
benchmarks and known limits.

## Remapping and post-remap correction

The [remapping guide](remapping.md) documents `remap` and `correction`, their
17-bin/five-material contracts, retained native policies and actual two-pass
capture. One complete caller pass is redistribution → tiny cleanup → hydration
→ radius/water conversion. Repeat that whole sequence twice; two consecutive
redistributions alone do not reproduce SALSA.

```sh
.venv/bin/python -m pytest -q tests/test_remapping_oracle.py \
  tests/test_remapping.py tests/test_remapping_correction.py \
  tests/test_remapping_capture.py tests/test_remapping_native.py
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-checked.toml --scope first-microphysics \
  --capture-salsa-state --capture-remapping-state \
  --run --output build/remapping-new
```

Use a new run directory. The copy-only overlay records eight boundaries across
both passes; the fixture records configuration, source and overlay provenance.
The tests retain native orphan material and tiny-particle inert-material loss.
An absent organic-gas mapping in the tiny cleanup branch is D007: affected JAX
cells report `valid=False` and NaN physical outputs, rather than claiming a
successful native result.

```sh
.venv/bin/python -m benchmarks.benchmark_remapping --repetitions 500 --trials 5
```

Run benchmarks after tests finish. This adapter times one redistribution call;
cleanup, hydration and the second pass are excluded. The measured one-call CPU
results show no speedup; see the [raw timing links](remapping.md#isolated-cpu-benchmark).
They do not establish composed or full-model performance.

## Complete native SALSA call

The [composition guide](salsa-composition.md) documents `salsa_step`, its native
mixed-unit mass interface, static JIT switches and process traces. It includes
initial cleanup, final density, recalculation and both complete correction passes.

```sh
.venv/bin/python -m pytest -q tests/test_salsa.py tests/test_salsa_native.py
.venv/bin/python -m benchmarks.benchmark_salsa --single-case --repetitions 500 --trials 5
.venv/bin/python -m benchmarks.benchmark_salsa --repetitions 500 --trials 5
```

Run benchmarks after all validation jobs finish. Single-case timing covers two
native vertical cells; the full matrix covers 54 cells in 21 static groups.
Compilation and synchronized warm execution are reported separately.

To capture a checked native call with diameter recalculation, use a fresh output
path and `configs/fortran-recalc.toml`; for active nucleation plus recalculation,
use `configs/fortran-nucleation-active-recalc.toml`. Both configure copied driver
sources before initialization. Original Fortran files and inputs remain intact.

```sh
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-recalc.toml --scope first-microphysics \
  --capture-salsa-state --capture-hydration-state --capture-coagulation-state \
  --capture-condensation-state --capture-remapping-state \
  --run --output build/salsa-recalc-new
```

Add `--capture-nucleation-state` for the active profile. Commit-sized remaining
gates cover diagnostic conversion (H03), remaining HAM metadata/conversions, longer
core trajectories (I03), and the complete box driver.

## Seven-mode initialization and HAM gas conversion

Number initialization now has an independent JAX API and actual driver capture:

```sh
.venv/bin/python -m pytest -q tests/test_initialization.py tests/test_initialization_native.py
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-checked.toml --scope first-microphysics \
  --capture-initialization-state --run --output build/initialization-new
.venv/bin/python -m pytest -q tests/test_gas_conversion.py
```

Use a fresh output directory. Read the [initialization guide](initialization.md)
for units, midpoint quadrature and native mode-literal precision. The
[gas conversion guide](ham-conversion-audit.md) distinguishes molecules/cm³
from SALSA's molecules/m³ and documents all three native conversion codes.
Material/tracer initialization and actual first-call gas conversion now pass
checked default and active-forcing captures. Reproduce both boundaries with:

```sh
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-checked.toml --scope first-microphysics \
  --capture-initialization-state --capture-ham-conversion-state \
  --capture-salsa-state --run --output build/ham-conversion-new
.venv/bin/python -m pytest -q tests/test_aerosol_initialization.py \
  tests/test_aerosol_initialization_oracle.py tests/test_ham_conversion_capture.py \
  tests/test_ham_conversion_native.py
```

Use `configs/fortran-nucleation-active.toml` and another fresh directory for the
active profile. Six checked runs (control/capture/repeat for each profile) preserve
outputs exactly. Export with `tools.ham_conversion_capture.export_fixture`;
export the number capture from that same run with
`tools.initialization_capture.export_fixture` into `number-initialization/`
inside the HAM fixture. Shared run hashes associate the complete initialization
chain. The [material guide](aerosol-initialization.md) explains units and limits;
H03 diagnostic conversion and the remaining HAM wrapper are next.
