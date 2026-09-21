# SALSA-JAX

An incremental Python/JAX port of HAM-SALSA aerosol microphysics, validated
process by process against the original Fortran implementation.

**Status:** the Brownian collision-coefficient kernel is implemented and tested.
Complete coagulation, condensation, nucleation and the coupled box model remain
in the [commit-sized roadmap](docs/plans/ULTRAPLAN.md). This repository does not
currently provide a complete JAX SALSA simulation.

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

## Fortran reference and current limitation

This repository preserves the history and original reference revision of
[reflective-org/SALSA](https://github.com/reflective-org/SALSA).
The tag `fortran-baseline-d87dc07` identifies the original reference revision
`d87dc072084336fd351d00a2ade863304c6f313d`. The `upstream` remote in the initial
local checkout points to that repository; `origin` is SALSA-JAX. Clones only
configure `origin` unless an upstream remote is added explicitly.

Four isolated Fortran repairs address absent organic gas, an undefined organic
fraction, activation velocity shape/units, and undersized aerosol-water storage.
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
The next priority is a small reproduction and repair of the checked Fortran
failure, followed by full-precision process capture and grid-geometry porting.

See [LICENSE.txt](LICENSE.txt) and retained source notices for attribution.
The [original Fortran overview](docs/fortran-overview.md) describes HAM-SALSA's
scientific context.
