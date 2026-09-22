# Complete standalone coagulation port

`src/salsa_jax/processes/coagulation_step.py` ports the complete
`mo_ham_salsa_dynamics.F90:coagulation` call for standalone SALSA: 17 bins,
five materials, `nsoa=0`, without `HAMMOZ`. It includes coefficient construction,
sequential number/volume updates in 1a/2a/2b, cross-regime transfer, salt
relabeling and the final native volume clamp. Condensation and nucleation are
separate processes and are not included.

## API and inputs

Enable double precision before creating scientific arrays:

```python
import jax
jax.config.update("jax_enable_x64", True)
from salsa_jax.processes.coagulation_step import coagulation_step

step = jax.jit(coagulation_step)
state = step(number, volume, wet_diameter, dry_diameter,
             temperature, pressure, timestep)
# state.number and state.volume are new arrays; inputs are unchanged.
```

| Input | Shape | Units / contract |
|---|---|---|
| `number` | `(...,17)` | Particles/m³, nonnegative |
| `volume` | `(...,17,5)` | Material m³ per air m³, nonnegative; SO4/OC/BC/SS/DU |
| `wet_diameter` | `(...,17)` | Positive m; diameter despite the stale Fortran radius comment |
| `dry_diameter` | `(17,)` or `(...,17)` | Positive reference-bin midpoint diameter, m |
| `temperature`, `pressure` | Scalar or broadcastable leading axes | Positive K and Pa |
| `timestep` | Scalar or broadcastable leading axes | Nonnegative s |
| `enabled` | Scalar boolean, keyword | False returns unchanged number/volume |

The state leading axes must match. They may represent columns, levels or
independent cases. The fixed bin partitions are 0:3 (1a), 3:10 (2a), and
10:17 (2b); the 2b geometry duplicates 2a. Arbitrary bin partitions, extra
materials and VBS/aqueous paths are not supported.

Hydration supplies wet diameters before this call. This port accepts those
arrays explicitly; it does not silently substitute dry diameters. Fortran's
`pcore` argument is unused here and is omitted from the Python API. The
approximate particle masses use reference dry diameters and the original
1500 kg/m³ density, independently of material composition.

`coefficient_matrix` and `coagulation_update` expose the coefficient and transfer
stages separately. The former builds the native upper triangle and mirrors it;
the latter preserves sequential bin updates, so later bins see the already
updated state. Reductions may differ by floating-point evaluation order.

The native routine converts salt in insoluble bins to sulfate **by volume**
and clamps negative volume after all transfers. These operations also execute
at `timestep=0`; disabling the process skips them. No extra clipping, timestep
splitting or conservation correction has been introduced.

## Independent Fortran checks

```sh
.venv/bin/python -m pytest -q
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-checked.toml --scope first-microphysics \
  --capture-salsa-state --capture-coagulation-state \
  --run --output build/coagulation-new
.venv/bin/python tools/coagulation_capture.py \
  --run-directory build/coagulation-new --output build/coagulation-fixture-new
```

The copy-only overlay captures immediately around the real coagulation call,
after initial hydration and before subsequent hydration/condensation. The
reviewed fixture is `tests/fixtures/coagulation-default/`, with source/input
hashes, compiler flags, array dimensions and units. Entry and exit use binary64
round-trip decimal precision. Native captures and all six returned HAM arrays
are identical with this overlay on/off; a fresh captured run repeats exactly.

The separate oracle extracts the original process and `coagc` verbatim. A
single diagnostic assignment copies the actual coefficient matrix. Minimal
dependency modules supply the explicit grid/composition contract; they contain
no replacement process formulas. Bounds/FPE-checked and optimized builds
produce identical outputs in the acceptance cases.

The 31-case matrix covers the actual captured boundary, empty state, every
single bin, zero timestep, mixed dense cases, unequal hydration, seeded mixtures
and a minimal cross-regime stress case. Tests compare eager/JIT coefficients,
transfers and final state, enforce exact disabled behavior, and compare 20
successive isolated-process calls with fixed supplied geometry. These sequences
are not full SALSA trajectories. Relative tolerances are 1e-13 for coefficients,
2e-13 for a process call and 4e-13 for the repeated-call test, with zero absolute
floor; exact reference zeros must stay zero.

## D009: inherited cross-regime conservation defect

Numerical agreement is established; unrestricted scientific acceptance is not.
The native sequential scheme can create material volume. A 2a bin gains from
still-unupdated 2b volume, while that 2b bin later loses material using the
already-reduced 2a number. These transfers do not cancel under stiff collisions.

A minimal pure-sulfate case populates corresponding bins 4 and 11 with 1e12
particles/m³ each, dry=wet diameter about 80.15 nm, temperature 280 K, pressure
101325 Pa and timestep 3600 s. Fortran creates **240.2863%** additional total
material volume. The mixed dense case creates **89.4139%**. This is independent
of the salt-to-sulfate relabeling approximation. The captured default call has
near-roundoff budget closure, which does not establish a safe domain for all
other concentrations/timesteps.

The port retains this reference behavior and tests the discrepancy explicitly.
A conservative update would be a separate scientific change with its own
oracle, budget matrix and trajectory review; this milestone does not make that
change or declare the stress cases physically valid.

## Benchmark scope

```sh
.venv/bin/python -m benchmarks.benchmark_coagulation_process \
  --repetitions 1000 --trials 5
.venv/bin/python -m benchmarks.benchmark_coagulation_process \
  --single-case --repetitions 1000 --trials 5
```

Both use warmed elapsed wall time, fixed identical inputs on every repetition,
separate compilation and output synchronization. JAX compilation/transfer and
Fortran startup/I/O are excluded. Fortran includes input reset copies; JAX
includes dispatch. Optimized Fortran disables runtime checks for timing while
retaining `-O2`, no fast-math and no LTO. The separate tests retain strict runtime
checks. Trial order alternates; thread settings are recorded, not normalized.
These are CPU coagulation timings, not GPU or full-model speedups.

On the recorded ARM64 CPU, five trials of 1000 repetitions gave:

| Workload | JAX median warm time | Fortran median warm time |
|---|---|---|
| Captured default, one case | 42.51 µs | 4.11 µs |
| Full 31-case batch | 132.09 µs | 123.61 µs |

There is no measured speedup. Maximum relative error in number/volume outputs
is 1.82e-15; the coefficient matrix error is 1.30e-14. The reference-zero
outputs remain exactly zero. These are local measurements, not performance
promises for other hardware or workloads.

See the [checked integration and coefficient evidence](evidence/coagulation-process-2026-09-21.json),
[batch benchmark](evidence/coagulation-benchmark-batch-2026-09-21.json) and
[single-case benchmark](evidence/coagulation-benchmark-single-2026-09-21.json).
The full suite passed 188 tests before two final process cases were added;
the final 15-test process suite also passed, covering the **190 tests at that
coagulation milestone**. Later totals are in the [status ledger](../plans/STATUS.md).

The separate [condensation port](condensation-process.md) is now available
with nucleation disabled; [supported nucleation](nucleation-process.md) has a
separate coupling API. [Grid and hydration](grid-and-hydration.md) are also ported
and validated in the default pre-remap chain. This standalone API still accepts
supplied geometry/wet diameters. Both [remap/correction passes](remapping.md)
are now separately accepted; complete native composition remains pending.
