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
both sulfate and optional nonvolatile organic uptake; no nucleation API is
exposed yet. Regenerate its actual boundary with `--capture-condensation-state`
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
