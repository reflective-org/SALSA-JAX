# SALSA-JAX

An incremental Python/JAX port of HAM-SALSA aerosol microphysics, validated
process by process against the original Fortran implementation.

**Status:** standalone coagulation and condensation are ported to JAX for the
17-bin, five-material configuration, using supplied hydrated diameters.
Condensation includes sulfate and optional nonvolatile organic uptake with
nucleation disabled. Nucleation and the coupled SALSA model remain in the
[commit-sized roadmap](docs/plans/ULTRAPLAN.md).

## Quick start

Use a dedicated virtual environment. GNU Fortran must be on `PATH` for the
mandatory original-Fortran comparisons; the driver metadata test also needs
NetCDF's `ncdump` on `PATH`.

```sh
git clone git@github.com:reflective-org/SALSA-JAX.git
cd SALSA-JAX
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest -q
```

Enable double precision before calling scientific kernels:

```python
import jax
jax.config.update("jax_enable_x64", True)
from salsa_jax.processes.coagulation import coagulation_coefficient

coefficient = coagulation_coefficient(
    1e-8, 2e-8, 7.85e-22, 8.38e-21, 280.0, 101325.0
)  # m³/s; diameters m, masses kg, temperature K, pressure Pa
```

The first 1734-case comparison has maximum relative error below 9e-15. The
initial leaf benchmark shows no speedup; see the measured
[results and limitations](docs/development/results-2026-09-21.md).

The [complete coagulation API and validation](docs/development/coagulation-process.md)
cover 31 cases, actual model-boundary capture and repeated process calls.
Maximum process relative error is below 2e-15. CPU timing shows no speedup yet:
31 cases take about 132 µs in JAX versus 124 µs in optimized Fortran.
The port preserves a discovered native cross-regime volume-conservation defect
(D009), which becomes large in stiff stress cases; numerical parity does not
establish physical validity there.

The separate [condensation API and validation](docs/development/condensation-process.md)
cover 29 cases against the D001/D002-repaired Fortran routine. Particle number
and gas outputs match exactly; maximum volume relative error is about 6e-16.
Both species budgets close. CPU timing is slower than Fortran: 35.3 versus
6.4 µs for 29 cases. Positive organic cases are kernel fixtures; the default
full driver does not register that gas.

## Fortran reference and current limitation

This repository preserves the history and original reference revision of
[reflective-org/SALSA](https://github.com/reflective-org/SALSA).
The tag `fortran-baseline-d87dc07` identifies the original reference revision
`d87dc072084336fd351d00a2ade863304c6f313d`. The `upstream` remote in the initial
local checkout points to that repository; `origin` is SALSA-JAX. Clones only
configure `origin` unless an upstream remote is added explicitly.

Six isolated Fortran repairs address absent organic gas in condensation and
nucleation, an undefined organic fraction, activation velocity shape/units,
undersized aerosol-water storage, and overwritten component maps.
The full checked run now reaches an unresolved radiation initialization failure;
full-model scientific validation is pending. See the
[repair record](docs/development/reference-repairs.md). The first scalar kernel
is independently validated.
Source relocation waits for reference validation. Historical benchmark evidence
was generated in the original SALSA working directory before this repository
was created; its recorded source hashes remain applicable.

Full reference audits additionally need Make and NetCDF C/Fortran libraries:

```sh
.venv/bin/python tools/fortran_reference.py --output build/reference --run
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-checked.toml --output build/checked --run
```

The second command currently reproduces the known defect; do not suppress the
check to claim validation. Use a new output directory for each run.

For an explicitly narrower reference, stop after the first HAM microphysics
call under strict checks:

```sh
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-checked.toml --scope first-microphysics \
  --capture-salsa-state --run --output build/first-microphysics
```

This scope checks finite microphysics outputs and records completion before
radiation, activation and removal. It is not a validated full trajectory;
see [scope and limitations](docs/development/scoped-reference.md). The
[native capture guide](docs/development/native-state-capture.md) describes the
full-precision fixture, map validation and repeated-run evidence.

## Layout and development

- `src/salsa_jax/`: JAX kernels and constants.
- `tests/`, `tools/`, `benchmarks/`: parity tests, original-Fortran oracles and timings.
- `configs/`: reference build/run profiles.
- `docs/`: architecture, evidence, contributor workflows and plans.
- Root `.F90`, `include/`, `input/`: Fortran reference source and scientific inputs.
- `.venv/`, `build/`, `outputs/`: local generated files, excluded from Git.

Start with the [documentation index](docs/README.md),
[development quickstart](docs/development/quickstart.md), and
[current status / next tasks](docs/plans/STATUS.md).
Native entry/exit capture is verified for the default first call. Coagulation
and condensation are separately captured and ported. Next is independent
nucleation; hydration, grid generation and remapping also remain to be ported
for coupled execution. Full-driver
radiation requires valid optical tables and a separate initialization repair.

See [LICENSE.txt](LICENSE.txt) and retained source notices for attribution.
The [original Fortran overview](docs/fortran-overview.md) describes HAM-SALSA's
scientific context.
