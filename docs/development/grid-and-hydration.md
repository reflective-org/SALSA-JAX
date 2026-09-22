# Native grid generation and hydration

The fixed 17-bin grid, five-material inverse molality and equilibrium hydration
are independently callable JAX kernels. They retain the current standalone
Fortran arithmetic and actual initialized material properties. No production
Fortran source change accompanies this port.

## Grid geometry

```python
import jax
jax.config.update("jax_enable_x64", True)
from salsa_jax.grid import size_grid

grid = size_grid()  # default limits: 3e-9, 5e-8, 7e-7, 1e-5 m
```

`SizeGrid` contains `vlolim`, `vhilim` (m³/particle), `dpmid` (m), and
`vratiohi`, `vratiolo` (dimensionless volume ratios). Positive, increasing
`regime_limits` have shape `(...,4)`; outputs have shape `(...,17)`. The fixed
layout is 3 bins in 1a, then 4+3 in 2a, duplicated exactly in 2b. Ordering
restarts at bin 11. Mid-diameter follows the average of bounding **volumes**.

Native `REAL(ii)/REAL(nbin)` evaluates spacing fractions in binary32 before
promotion to double precision. Computing them directly in float64 changes
one-third/two-thirds boundaries materially. The port preserves these fractions,
including tiny roundoff gaps between regime boundaries; it does not snap edges.
The verbatim `set_sizebins` oracle and the captured default grid match every
field exactly in the recorded eager/JIT tests.

## Water uptake and solubility

```python
from salsa_jax.processes.hydration import (
    equilibrate, inverse_molality, solubility_mask, HydrationParameters,
)
water = jax.jit(equilibrate)(
    number, volume, relative_humidity, temperature, grid.dpmid,
    soluble_mask=solubility_mask(hydrate_insoluble=True),
    parameters=HydrationParameters(),
)
# water.core: m3/particle; water.wet_diameter: m; water.iterations: per-bin count
```

Number has shape `(...,17)` in particles/m³; volume has shape `(...,17,5)` in
material m³/air m³, ordered SO4/OC/BC/SS/DU. State leading axes match. RH is a
fraction, T is K, and environment/scalar properties broadcast over case axes.
Species masses/densities have a final length-five axis. Inputs are immutable.
Defaults are the actual native species registry, including sulfate 96.0631
g/mol, organic 180 g/mol and water 18 g/mol. `HydrationParameters` exposes species
properties, water density/molar mass, organic soluble fraction, surface tension
and the number threshold. Constants are centralized in `constants.py`.

`inverse_molality(activity)` returns five values in kg water/mol solute. Activity
must lie in (0,1). Sulfate and salt use the original polynomials, organic uptake
includes `epsoc`, and BC/dust return zero. The independently compiled actual
Fortran leaf is the reference for these expressions.

Native hydration:

- Uses the strict condition `number > nlim`, with default nlim=1 particle/m³.
  Empty/below-threshold bins receive grid-midpoint diameter and corresponding
  dry volume, even if their supplied material volumes are nonzero.
- Clamps RH to [.05,.95]. Each soluble bin starts with Kelvin factor 1.001 and
  iterates until successive diameters differ by at most one percent of the
  previous diameter. The uncapped native convergence rule is retained.
- Uses `sizeclass%lsoluble`, not a direct branch on `lhydr2b`. Initialization sets
  the first ten bins soluble and the final seven from `lhydr2b`; the default is
  true. Explicit masks also support isolated branch tests.
- Gives insoluble bins their dry diameter. Pure BC/dust with positive volume
  also stay dry when marked soluble, converging in two iterations.

Each valid soluble active bin needs positive total dry volume and finite
physical properties. The inherited **D010** zero-volume failure is reproduced
in checked Fortran. JAX returns NaN wet diameter for such a bin, or for a
nonfinite Kelvin factor, so invalid input cannot hang an uncapped JIT loop or
appear as a successful physical result. This is error signaling, not a repair
of the native scientific model. No artificial iteration cap is applied to valid
cases; the validation matrix does not establish convergence over all inputs.

## Independent reference and native capture

```sh
.venv/bin/python -m pytest -q tests/test_grid.py tests/test_inverse_molality.py \
  tests/test_hydration_oracle.py tests/test_hydration.py tests/test_hydration_native.py
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-checked.toml --scope first-microphysics \
  --capture-salsa-state --capture-hydration-state \
  --capture-coagulation-state --capture-condensation-state \
  --run --output build/hydration-capture-new
```

The hydration oracle compiles actual `zbinmol_inv` and `equilibration`, with
original precision and physical constants. It supplies explicit species data
and solubility flags; an untimed copy-only diagnostic records iteration counts.
Checked/optimized outputs agree in the 31-case matrix. Cases cover pure/mixed
materials, RH clamps, number-threshold neighbors, insoluble masks, physical
property variations and seeded mixtures. Pure inert-material and zero-volume
cases distinguish valid dry behavior from the native floating-point failure.

`--capture-hydration-state` records the first actual equilibration boundary,
after the native pre-process cleanup and before coagulation. It leaves later
optional/post-remap calls untouched. Capture/control and fresh-repeat checks
must compare full-precision HAM arrays, native/process files and diagnostics.
The reviewed fixture is `tests/fixtures/hydration-default/` and its exporter is
`tools/hydration_capture.py`.

The native acceptance test reconstructs the captured material registry and
first proves the extracted Fortran oracle reproduces that real boundary. It
then composes JAX grid → hydration → coagulation → condensation against their
captured boundaries, retaining the native `recalc=False` call order. It supplies
JAX-computed geometry and hydrated diameters. This composition ends before
remapping and is not a complete SALSA step or a box-model trajectory.

## Measured acceptance and CPU timing

All **459 tests passed at this milestone** using `.venv/bin/python -m pytest -q`
(50.13 s). Subsequent [remapping/correction acceptance](remapping.md) raises the
current suite to 541 tests.
The 31-case matrix has maximum relative core error 3.53e-16 and wet-diameter
error 2.08e-16 in eager/JIT mode, with exact per-bin iteration counts (maximum
four). The molality leaf's eager error is below 6.44e-14; JIT matches exactly.
Hydration acceptance uses rtol=3e-13, molality uses 2e-12, both with atol=0.

At the actual native initial boundary, extracted Fortran wet-diameter error is
1.13e-16 relative; JAX core/wet errors stay below 1.93e-16. The composed default
pre-remap chain matches number/gas exactly and volume within 2.19e-16 relative.
Capture/control outputs, runtime metadata and ordinary diagnostics are identical;
a fresh process repeats every boundary exactly. Downstream coagulation and
condensation files also match the existing fixtures byte for byte. The
[integration record](evidence/grid-hydration-2026-09-21.json) preserves provenance.

Reproduce isolated CPU timings with:

```sh
.venv/bin/python -m benchmarks.benchmark_hydration --repetitions 500 --trials 5
.venv/bin/python -m benchmarks.benchmark_hydration --single-case --repetitions 500 --trials 5
```

| Workload | JAX warm median | Fortran warm median | Fortran/JAX |
|---|---:|---:|---:|
| One curated `native-final-state` case | 29.14 µs | 1.346 µs | 0.046 |
| All 31 hydration cases | 53.75 µs | 38.72 µs | 0.721 |

These measurements show **no CPU speedup**. The single case comes from the
existing final-state fixture, distinct from the new initial-hydration fixture.
Raw [single-case](evidence/hydration-benchmark-single-2026-09-21.json) and
[batch](evidence/hydration-benchmark-batch-2026-09-21.json) records retain five
alternating trials, exact cases, source hashes, compiler flags and thread settings.
JAX compilation is separate (0.582/0.236 s for single/batch), and both physical
outputs are synchronized. Fortran uses O2 without fast-math; checked and optimized
outputs and iteration counts agree exactly before timing. Both paths use fixed
inputs; Fortran includes input/property setup and output copying, while JAX
includes dispatch/synchronization and excludes initial transfers. Iteration
counts are validated outside timing; the Fortran diagnostic switch remains off.
No full-model, trajectory or accelerator performance claim follows.
