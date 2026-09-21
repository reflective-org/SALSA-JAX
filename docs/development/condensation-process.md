# Standalone condensation with nucleation disabled

`src/salsa_jax/processes/condensation.py` ports the standalone SALSA
condensation routine for `nsnucl=0`, `nsoa=0`, without `HAMMOZ`. It includes
transport coefficients, transition corrections, sulfate uptake and optional
nonvolatile organic uptake. Supported nucleation is now available through the separate
[condensation coupling API](nucleation-process.md). VBS, semivolatile partitioning,
hydration and remapping remain separate work.

## API and state contract

```python
import jax
jax.config.update("jax_enable_x64", True)
from salsa_jax.processes.condensation import condensation_step, CondensationParameters

step = jax.jit(condensation_step)
result = step(number, volume, gas, wet_diameter, temperature, pressure, timestep,
              organic_enabled=False, parameters=CondensationParameters())
# New result.number, result.volume and result.gas; inputs are unchanged.
```

| Input | Shape | Meaning / units |
|---|---|---|
| `number` | `(...,17)` | Particle concentration, particles/m³ |
| `volume` | `(...,17,5)` | Material m³ per air m³; SO4, OC, BC, SS, DU |
| `gas` | `(...,3)` | Canonical SO2, H2SO4, OC, in molecules/m³ |
| `wet_diameter` | `(17,)` or `(...,17)` | Supplied hydrated diameter, m |
| `temperature`, `pressure` | Scalar or leading axes | K, Pa |
| `timestep` | Scalar or leading axes | s |
| `organic_enabled` | Scalar or leading axes | Whether the OC gas slot is registered |
| `enabled` | Scalar boolean | False preserves all state exactly |

State leading axes must match; environment and species-property inputs
broadcast over them. Use finite, nonnegative concentrations and timestep,
positive diameters/environment/species properties, and physical accommodation
coefficients in (0,1]. The supported grid has 17 bins with 1a/2a/2b boundaries
3/10/17. This API does not accept a nucleation switch: nucleation is excluded.

The initialized full-model default has only SO2/H2SO4 gas slots and no organic
gas. Its canonical adapter pads an inactive third slot with zero. A positive
value in an inactive OC slot is preserved, not consumed. Registered organic
cases are explicit kernel fixtures; this port does not enable organic gas
registration in the full Fortran driver. Other gas registries require their
own explicit mapping and validation.

Number, inert SO2, BC, salt and dust remain exactly unchanged. Only gas H2SO4/OC
and their matching aerosol volumes can change. Condensation accepts hydrated
diameters; it neither recomputes hydration nor updates bin geometry. Relative
humidity, boundary-layer index and row index are unused when nucleation is off
and are omitted from the numerical API. The Fortran diagnostic timestep global
is also irrelevant to this update; `timestep` is the actual integration step.

## Properties, thresholds and equations

`CondensationParameters` is a JAX-compatible named tuple. Sulfate/organic molar
masses default to **96.0631/180 g/mol**, with densities **1841/2000 kg/m³**.
These are the actual initialized legacy values; the older regression fixture's
98/150 g/mol values are not substituted. Each property can carry leading case
axes. Mass accommodation defaults to 1 and can be scalar, `(17,)` or
`(...,17)`. The original molecular diameter and physical constants are fixed
in `constants.py` and checked against compiled runtime metadata.

`condensation_rates` returns per-bin sinks in s⁻¹, including particle number.
The first two bins include molecule size and particle diffusion; subsequent
bins use the native vapor-only diffusion expression. Both vapors share this
sink with nucleation off. A vapor updates only when its concentration is
strictly greater than 1e-10 molecules/m³ and the total sink is strictly greater
than 1e-30 s⁻¹. The native semi-implicit gas update is `gas/(1 + dt*sink)`;
the difference is distributed by per-bin sink fraction and molecular volume.
No new clipping or conservation correction is applied.

## Reference and verification

The oracle extracts the current Fortran routine verbatim, including the
separately reviewed D001/D002 repairs. An aborting nucleation stub proves that
these tests cannot accidentally validate a coupled nucleation calculation.
Original physical-constant modules are compiled, and an untimed assignment
captures the actual per-bin rates. Checked and optimized oracle results agree
exactly in all 29 cases.

```sh
.venv/bin/python -m pytest -q
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-checked.toml --scope first-microphysics \
  --capture-salsa-state --capture-coagulation-state --capture-condensation-state \
  --run --output build/condensation-new
.venv/bin/python tools/condensation_capture.py \
  --run-directory build/condensation-new --output build/condensation-fixture-new
```

The reviewed fixture is `tests/fixtures/condensation-default/`. Its capture is
immediately around the real condensation call, after coagulation and any
optional second hydration, before remapping. All six returned HAM arrays and
the native/coagulation capture files match with condensation capture on/off.
A fresh process reproduces both condensation files exactly.

Tests cover the actual boundary, absent/zero/positive organic gas, empty
particles, zero timestep, exact/adjacent gas thresholds, tiny-sink thresholds,
small-bin corrections, heterogeneous accommodation/species properties and
seeded meteorology. Zero-initial-condensate cases make budget checks sensitive
to newly transferred material. Eager/JIT state parity uses relative tolerance
2e-13, rates 1e-13, each species budget 3e-14, with no absolute floor.
Twenty repeated calls use a 4e-13 tolerance and retain fixed supplied diameters.
A separate test chains the ported coagulation and condensation outputs against
their captured boundaries; it does not implement hydration or remapping.

## CPU benchmark

```sh
.venv/bin/python -m benchmarks.benchmark_condensation --repetitions 1000 --trials 5
.venv/bin/python -m benchmarks.benchmark_condensation --single-case --repetitions 1000 --trials 5
```

Compilation, startup and I/O are outside warmed elapsed timing. Each invocation
uses the same initial inputs; this benchmark does not advance a trajectory.
Fortran includes mutable-input reset and species setup, with runtime checks
disabled for timing, `-O2`, no fast-math and no LTO. JAX includes dispatch and
synchronization of all outputs. Trials alternate order. Thread settings and
raw timings are recorded; costs are disclosed rather than treated as identical.
The oracle's strict checks remain enabled in correctness tests. These timings
exclude hydration, nucleation, remapping and the complete SALSA model.

Recorded ARM64 CPU results (five trials, 1000 repetitions):

| Workload | JAX median warm time | Optimized Fortran median |
|---|---|---|
| Captured default, one case | 10.10 µs | 0.263 µs |
| Full 29-case batch | 35.31 µs | 6.40 µs |

There is no measured speedup. Number and gas outputs match exactly; maximum
volume relative error is 6.01e-16. The original species-budget relative errors
are below 3.34e-16. JAX species budgets also pass their explicit checks.

See [integration evidence](evidence/condensation-process-2026-09-21.json),
[batch measurements](evidence/condensation-benchmark-batch-2026-09-21.json) and
[single-case measurements](evidence/condensation-benchmark-single-2026-09-21.json).
The full suite passed 241 tests; the final 14-test condensation suite includes
one added two-process integration case, covering all **242 tests at that milestone**.
