# Native remapping and post-remapping correction

The C15–C17 kernels preserve the current Fortran redistribution and correction
arithmetic for the fixed 17-bin, five-material configuration. They are
independently callable and leave inputs unchanged. Production Fortran sources
are not changed by this port or its capture instrumentation.

## One redistribution call

```python
import jax
jax.config.update("jax_enable_x64", True)
from salsa_jax.grid import size_grid
from salsa_jax.processes.remapping import remap, RemappingParameters

grid = size_grid()
redistributed = jax.jit(remap)(
    number, volume, temperature, grid,
    transfer_soluble=False, parameters=RemappingParameters(),
)
```

Number has shape `(...,17)` in particles/m³; material volume has shape
`(...,17,5)` in material m³/air m³, ordered SO4/OC/BC/SS/DU. Temperature is K.
Leading state axes must match; environment, geometry and scalar parameters
broadcast over case axes. `SizeGrid` provides lower/upper volume limits in
m³/particle and midpoint diameter in m. Enable float64 before constructing
arrays or compiling kernels.

`RemappingState` returns `number`, `volume`, and per-cell `iterations`.
Iterations count descending scans, including the initial scan of an empty
cell. `RemappingParameters` supplies sulfate/water molar masses (g/mol),
densities (kg/m³), surface tension (J/m²), organic soluble fraction, saturation
limit, ion factor and number threshold (particles/m³). Physical constants and
defaults are centralized in `salsa_jax.constants`; native fixtures carry the
actual initialized properties.

Redistribution follows native order, using **one-based** bin numbers here:

- Scan bins 16 down to 1, skip bin 10, and process only number strictly above
  `nlim`. Bins 10 and 17 are terminal receivers; their own size violations are
  ignored by redistribution. The optional soluble transfer still visits bin 17.
- Move undersized particles down one bin, except bin 11 moves to bin 3 and
  undersized bin 1 stays put. Oversized transfers move up one bin.
- Preserve the source's branch order, pre-transfer mean volume and repeated-scan
  stopping condition. A receiving bin already visited in the descending scan
  is not automatically revisited. No extra convergence sweep is added.
- With `transfer_soluble=True`, move eligible 2b bins 11–17 to corresponding
  2a bins 4–10 when the native critical fraction is strictly below
  `(SO4 volume + epsoc * OC volume) / total volume`. The critical fraction
  depends on temperature, midpoint volume, water/sulfate properties, surface
  tension, ion factor and `log(saturation_limit)²`.

The native large-transfer algorithm can leave material in a bin whose number
becomes zero. Tests retain this **orphan-material behavior** and check material
and number totals independently; they do not silently redistribute or discard
the residual. With soluble transfer enabled, an active 2b bin having zero total
volume causes a checked native division failure. JAX marks that cell's number
and volume NaN. This signals an invalid reference case, not a scientific repair.

## Cleanup, hydration and geometry

```python
from salsa_jax.processes.hydration import HydrationParameters, solubility_mask
from salsa_jax.processes.remapping_correction import correction

corrected = jax.jit(correction)(
    redistributed.number, redistributed.volume, gas,
    relative_humidity, temperature, grid.dpmid,
    soluble_mask=solubility_mask(hydrate_insoluble=True),
    parameters=HydrationParameters(), organic_enabled=False,
)
```

Gas has shape `(...,3)`, ordered SO2/H2SO4/OC in molecules/m³; RH is a
dimensionless fraction. Native captures with no organic gas are adapted to a
zero-filled third slot while `organic_enabled=False` retains the absent-mapping
semantics. The number threshold must be positive.

Cleanup considers number **at or above** `nlim`, unlike redistribution and
hydration's strict greater-than checks. It removes particles with dry diameter
strictly below 1e-10 m. Diameter uses SO4+OC volume in bins 1–3 and all five
materials in bins 4–17. Removed sulfate and organic volumes return to their gas
slots using actual density, molar mass and Avogadro's constant. All particle
volumes are then cleared, so **tiny-particle inert-material loss** is preserved
and explicitly tested.

The existing **D007 absent-organic-gas access** remains unrepaired: native
cleanup accesses gas index zero when that mapping is absent, even if organic
volume is zero. Affected JAX cells return `valid=False`, NaN floating outputs
and zero hydration iteration counts; valid neighboring cells remain usable.

Correction calls the [hydration kernel](grid-and-hydration.md), then returns:

| Field | Units or meaning |
|---|---|
| `number`, `volume`, `gas` | Updated state in the units above |
| `core` | Dry volume, m³/particle |
| `wet_diameter` | m |
| `dry_radius_cm`, `wet_radius_cm` | cm, matching native HAM-facing geometry |
| `water_mass` | kg water/m³ air |
| `iterations` | Hydration iterations per bin |
| `valid` | Per-cell cleanup validity |

Water follows the original wet-minus-dry volume expression without clipping,
including small signed roundoff residuals. This validity flag detects the
absent-gas cleanup branch; it is not a general physical-validity certificate.
Hydration's documented nonfinite and zero-volume restrictions still apply.

## Two actual SALSA passes and captured boundaries

SALSA executes exactly **two successive** repetitions of
`distr_update → tiny cleanup → equilibration → radius/water conversion`.
Calling `remap` twice without intervening corrections is a useful isolated
oracle test, but does not reproduce that entire caller sequence.

`--capture-remapping-state` instruments only a copied `mo_ham_salsa.F90`.
Each pass produces entry, remap-exit, cleanup-exit and correction-exit records:
eight files in total. Records include prognostic number, material volume and
gas, meteorology, actual geometry, species mappings/properties, flags,
thresholds and constants. Correction exits also contain core volume, wet
diameter, radii and water mass. Full-precision values retain Fortran array order.

`tools.remapping_capture.load_capture` checks stage identity, shapes, finite
values, pure-input invariance, unchanged context and pass-to-pass continuity.
The exporter requires successful checked execution and matching recorded file
hashes; `load_fixture` verifies the retained inventory and hashes again.
`tools.remapping_fixture.load_native_case` adapts
`tests/fixtures/remapping-default/`, rejecting incompatible mappings and
constants before flattening native cell axes.

These boundaries support independent acceptance of both native passes and
composition with grid, initial hydration, coagulation and condensation.
**Density calculation, final HAM tracer conversion, the complete SALSA call
and box-model trajectories remain separate acceptance work.** Capturing a
finite first microphysics call does not establish full-model validity.

## Reproduce reference checks and capture

The remapping oracle extracts actual `distr_update` from
`mo_ham_salsa_update.F90`. The correction oracle extracts the actual cleanup
through water-conversion block from `mo_ham_salsa.F90` and links checked native
hydration. Neither reference is a Python rewrite of the tested arithmetic.
Checked failures must remain visible; tests do not skip a missing oracle.

```sh
.venv/bin/python -m pytest -q tests/test_remapping_oracle.py \
  tests/test_remapping.py tests/test_remapping_correction.py \
  tests/test_remapping_capture.py tests/test_remapping_native.py
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-checked.toml --scope first-microphysics \
  --capture-runtime --capture-salsa-state --capture-hydration-state \
  --capture-coagulation-state --capture-condensation-state \
  --capture-remapping-state --run --output build/remapping-capture-new
```

Use a new output directory. Compare an instrumented run with an identical
uninstrumented control and a fresh repeat, including all first-call HAM arrays,
earlier process captures and ordinary diagnostics. Retain source/overlay hashes,
compiler flags, runtime configuration and raw fixtures.

Standalone eager/JIT tests use rtol=3e-13 and atol=0 for number, material,
gas and geometry, with exact iteration-count comparisons. Water is the
difference of nearly equal wet/dry volumes, so relative error alone is unstable
near zero. Its error bound is `3e-13*abs(reference_water) +
32*float64_epsilon*(reference_core*water_density*reference_number)`; there is
no fixed absolute floor or clipping. Cases exercise size and number
thresholds, terminal bins, descending transfers, optional solubility transfer,
orphan material, tiny cleanup, molecular budgets and inherited failure paths.

All **541 tests pass** in the repository's `.venv` (59.49 s). The 47-case
remapping matrix has maximum relative errors 3.20e-14 for number and 4.26e-15
for material volume, in eager and JIT mode, for one and two successive calls.
Number/material budgets close to less than 4.14e-16 relative; all recorded native
descending-scan counts are one and match exactly. This matrix does not establish
convergence for every input. Checked and optimized Fortran outputs match exactly.

Tests at, below and above the soluble-fraction threshold preserve the native
strict branch in scalar and batched forms. XLA's reciprocal rewrite of division
by literal 1000 shifted this threshold by one ULP. A barrier on the array-shaped
denominator preserves the original divisions; no comparison tolerance or
threshold adjustment was introduced.

In the 54 correction cases, number/material/gas match exactly and geometry
errors stay below 3.53e-16 relative. Maximum water error is 2.72e-20 kg/m³, or
1.73e-15 normalized by dry mass. Native tiny negative water residuals are retained
as roundoff behavior; some eager results round to zero. This is not clipping.

For both actual default native passes, JAX remapping number/volume and corrected
number/volume/gas match exactly, including the interleaved two-pass chain starting
at the captured condensation output. Geometry errors stay below 2.81e-16 relative;
water error is below 2.07e-24 kg/m³ and 7.36e-16 of dry mass. The extracted remap
oracle differs from the full-driver capture by up to 3.20e-14 in number and
4.26e-15 in volume, within the same roundoff tolerance. The complete earlier
process chain is separately accepted; a single complete SALSA API and its
all-process composition gate remain open.

Three strict native runs give identical capture/control HAM arrays, runtime
metadata and ordinary diagnostics; all eight new boundaries repeat exactly.
Earlier hydration/coagulation/condensation fixtures remain byte-identical.
The [integration evidence](evidence/remapping-2026-09-21.json) retains original
source/input hashes, compiler flags, overlays and numerical measurements.
Inherited [D011 orphan material and D012 tiny inert-material loss](reference-repairs.md)
remain unrepaired; compatibility is not unrestricted scientific validation.

## Isolated CPU benchmark

```sh
.venv/bin/python -m benchmarks.benchmark_remapping --repetitions 500 --trials 5
.venv/bin/python -m benchmarks.benchmark_remapping --single-case --repetitions 500 --trials 5
```

The benchmark measures one `distr_update` call on identical fixed case cells.
It excludes cleanup, hydration/corrections and the second SALSA pass. Before
timing it verifies checked/optimized Fortran outputs and iteration counts
agree, then compares JAX physical outputs and untimed iteration diagnostics.

Fortran uses separately compiled O2 arithmetic without fast-math/LTO; timing
includes input/property setup and physical output copies, excluding process
startup, I/O, checksums and iteration capture. JAX includes dispatch and output
synchronization, excluding initial transfers and reporting compilation separately. Trials
alternate implementations; thread settings and case/source hashes are retained.
Repeated timings reuse fixed inputs rather than advancing a trajectory.

Recorded CPU medians over five trials of 500 fixed-input repetitions:

| Workload | JAX | Optimized Fortran | Fortran/JAX |
|---|---:|---:|---:|
| One `native-condensation-exit` case | 60.45 µs | 0.222 µs | 0.00367 |
| All 47 cases | 150.13 µs | 5.916 µs | 0.0394 |

There is **no measured CPU speedup**. Compilation is separate: 0.112 s for one
case and 0.130 s for the batch. Raw [single-case](evidence/remapping-benchmark-single-2026-09-21.json)
and [batch](evidence/remapping-benchmark-batch-2026-09-21.json) records include
all trials and workload/provenance details. Timing both complete correction
passes remains a separate task. No full-model, accelerator or gradient
performance claim follows from this isolated benchmark.
