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

GNU Fortran must be on `PATH` for the tests. The leaf oracle compiles the original
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

The checked profile currently fails on known D001, an inactive organic-gas
index in condensation; see [architecture](architecture.md). The ordinary run
completes but is not an approved scientific reference until that issue is resolved.
Do not disable bounds checking to declare the defect fixed.

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
