# Nucleation sources and condensation coupling

The JAX port now covers supported sulfate nucleation schemes `nsnucl=1,2,3`
and all three growth options `nj3=1,2,3`, independently of condensation, then
restores the actual condensation call order and first-bin particle insertion.
It uses the D001/D002/D006-repaired Fortran reference, `nsoa=0`, without HAMMOZ.
No production Fortran source changes accompany this port.

## APIs and units

Enable JAX float64 before calling kernels. The [rate leaves](nucleation-rates.md)
return critical clusters; the process source API adds vapor limitation and
survival to the fixed 3 nm target:

```python
import jax
jax.config.update("jax_enable_x64", True)
from salsa_jax.processes.nucleation import nucleation_sources
from salsa_jax.processes.condensation_nucleation import condensation_nucleation_step

sources = nucleation_sources(number, gas, wet_diameter, temperature,
                             relative_humidity, pressure, timestep,
                             scheme=2, growth=1, boundary_layer=True)
step = jax.jit(condensation_nucleation_step, static_argnames=("scheme", "growth"))
state = step(number, volume, gas, wet_diameter, temperature, relative_humidity,
             pressure, timestep, scheme=2, growth=1, boundary_layer=True)
```

| Input/output | Shape | Units or convention |
|---|---|---|
| `number` | `(...,17)` | Particles/m³ |
| `volume` | `(...,17,5)` | Material m³/air m³; SO4/OC/BC/SS/DU |
| `gas` | `(...,3)` | Molecules/m³; SO2/H2SO4/OC |
| `wet_diameter` | `(17,)` or `(...,17)` | Supplied hydrated diameter, m |
| Temperature, RH, pressure, timestep | Scalar or leading axes | K, fraction, Pa, positive s |
| `boundary_layer` | Scalar or leading axes | Native `jj >= ppbl`, including boundary level |
| `organic_enabled` | Scalar or leading axes | Explicitly registered nonvolatile organic vapor |
| `sources.molecular_rate`, `.fractions` | `(...,2)` | SO4/OC molecules/(m³ s), composition fractions |
| Coupled result | Named tuple | Updated `.number`, `.volume`, `.gas` |

State leading axes must match. Use finite nonnegative concentrations, positive
physical properties/diameters and accommodation in (0,1]. Environmental values
broadcast. `CondensationParameters` supplies actual species masses, densities
and accommodation; defaults are documented in the [condensation guide](condensation-process.md).
The activation coefficient defaults to 1e-7 s⁻¹. Scheme and growth are static
JIT options; each combination compiles separately.

Scheme 0 returns zero sources/fractions and the coupled API dispatches to plain
condensation. Coupled `enabled=False` preserves all state exactly. Schemes 1–3
return sulfate/organic fractions 1/0 even where the source is inactive. Organic
vapor may influence cluster growth without being part of nucleated particles;
its separate condensation path remains optional. Unsupported scheme 4 needs
NH3; schemes above 4 are outside the initialized native configuration.

## Native arithmetic retained

The source routine does not consume gas or insert particles. It limits critical
formation by available sulfate, applies the selected growth/survival formula,
then returns surviving rate times **158.79 molecules per 3 nm particle**.
Growth 1 uses the empirical condensation sink; growth 2 uses critical/target
coagulation sinks; growth 3 additionally performs five self-coagulation updates
using the original 30-term Taylor population series. No convergence repair or
blanket clipping is introduced.

Several inherited behaviors need scientific review separately:

- Growth's local sulfate concentration subtracts a limited molecular **rate**
  without a timestep factor, as written in Fortran.
- Growth 2/3 retain the source's `radius**2 + gamma` expression in the transition
  correction, including its dimensional inconsistency.
- The native condensation gate requires a positive background sink before
  adding the nucleation sink. An empty background can therefore have a positive
  source while the coupled state remains unchanged.
- First-bin number insertion requires molecular source strictly above 1;
  smaller positive sources still enter the vapor uptake expression.

Coupling computes original collision sinks once, evaluates organic then sulfate
uptake, adds the nucleation sink in the first bin, and applies the native
semi-implicit gas update. The corresponding added volume supplies the particle
number increment. Diameters remain caller-supplied throughout this process call.

## Independent acceptance

```sh
.venv/bin/python -m pytest -q tests/test_nucleation_rates.py \
  tests/test_nucleation_sources.py tests/test_nucleation.py \
  tests/test_nucleation_process_oracle.py
```

The process oracle compiles the complete native nucleation module, retaining a
verbatim byte copy, and mechanically extracts actual condensation. A small
untimed overlay records critical clusters; timing disables this diagnostic.
There is no translated nucleation formula in the reference tool. Strict checked
and optimized outputs match exactly in the **110-case matrix**.

Cases cover every scheme/growth combination, above/within boundary layer,
active binary nucleation, registered/absent organic gas, zero/limited vapor,
empty backgrounds, cutoff neighbors, heterogeneous physical properties and
seeded growth-3 states. Separate tests verify unchanged isolated state, eager/JIT
sources, coupled state, species budgets, scalar/multiaxis inputs, switches,
small-source insertion gates and ten successive fixed-diameter calls.

Maximum isolated source relative error is 1.74e-12 eager and 5.68e-15 JIT.
Acceptance tolerances are 5e-12 eager and 3e-13 JIT, no absolute floor. Coupled
state uses 3e-13; species budgets 3e-14; ten-call comparisons 1e-12. These
are finite-case acceptance results, not validation throughout the Taylor
series' parameter domain or proof of physical correctness of inherited formulas.

## Scope and next integration gates

The default driver has nucleation disabled. The original 110-case matrix uses
explicit initialized kernel fixtures. A subsequent [active native capture](nucleation-native-capture.md)
now validates scheme 2/growth 1 through the checked driver's actual nested
call and parent condensation boundary. [Grid and hydration](grid-and-hydration.md)
are now separately ported and accepted through the default pre-remap chain.
Both [remap/correction passes](remapping.md) are now separately accepted;
the complete JAX SALSA timestep remains pending. Full radiation still
has the D005 blocker and coagulation retains the D009 budget defect.
The [status ledger](../plans/STATUS.md) distinguishes completed process tasks
from pending native capture and trajectory gates.

## CPU benchmark

```sh
.venv/bin/python -m benchmarks.benchmark_nucleation --repetitions 500 --trials 5
.venv/bin/python -m benchmarks.benchmark_nucleation --single-case --repetitions 500 --trials 5
```

Both commands measure isolated sources and the coupled update for all nine
static scheme/growth groups. Add `--scheme 2 --growth 3 --operation coupled`
to select a group. The native oracle always computes two vertical cells per
case; JAX duplicates inputs with matching above/within-boundary flags. Parity
is checked for **both** cells before timing. A “single case” therefore means
two physical cells, not one.

Five trials of 500 fixed-input repetitions on the recorded ARM64 CPU show
**no speedup** in any measured group. Representative median warmed times:

| Workload | Operation | JAX µs | Fortran µs |
|---|---|---|---|
| Scheme 2, growth 3, 2 cells | isolated | 25.20 | 5.54 |
| Scheme 2, growth 3, 2 cells | coupled | 47.41 | 5.97 |
| Scheme 2, growth 3, 24 cells | isolated | 59.28 | 49.63 |
| Scheme 2, growth 3, 24 cells | coupled | 73.47 | 53.33 |

The batch comprises each group's curated cases, not a uniform 110-case batch
per configuration. Fortran includes mutable-state resets, per-case species
setup and selected-output copies; JAX includes dispatch and synchronization
of returned arrays. Compilation, transfers, process startup, I/O and critical
capture are outside warmed timing. Trials alternate implementation order;
thread settings are recorded rather than normalized. Fortran uses O2 without
fast-math, LTO or runtime checks for timing; correctness also uses strict checks.
Compilation costs are reported separately for every shape/configuration.

See the [integration record](evidence/nucleation-process-2026-09-21.json),
[batch measurements](evidence/nucleation-benchmark-batch-2026-09-21.json) and
[single-case measurements](evidence/nucleation-benchmark-single-2026-09-21.json).
Both operations match Fortran below 5.68e-15 relative error across the paired
benchmark cells; coupled state error is below 4.36e-15. The full repository
suite passed **315 tests at that milestone**, followed by 384 with active native
acceptance. Grid/hydration reached 459; subsequent remapping/correction
acceptance raises the current count to **541** in the repository's `.venv`.
