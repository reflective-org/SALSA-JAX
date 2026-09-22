# P5/P6 — Benchmarking, optimization, documentation, and delivery

All timing work follows the [validation protocol](../development/validation-and-benchmarks.md).
Correctness is a gate, not a percentage traded for speed. Report regressions and
slowdowns as readily as improvements. One row is one intended commit.

## P5.1 — Reproducible measurements

| ID | Depends on | Deliverable / suggested commit | Verification |
|---|---|---|---|
| P01 | C01.J | Benchmark collision coefficient with equal case batches | Warm synchronized JAX and internally timed Fortran; checksums/parity pass |
| P02 | P01 | Record case/source hashes and machine/runtime metadata | Another contributor can reconstruct exact workload/toolchain |
| P03 | C11.J,S08 | Add complete process kernel timing adapter | Same reset/copy costs and result materialization on both sides |
| P04a | P03,C11.J | Benchmark complete coagulation separately | Bin/material updates included; raw samples and compilation separated |
| P04b | P03,C14.J | Benchmark condensation with nucleation disabled | Same gas/aerosol cases and reset costs; gas/material budgets pass |
| P04c | P03,N07.J,N08a.J | Benchmark isolated nucleation and condensation coupling separately; implemented | Matched two native cells per case; nine static groups, checked parity and raw compile/warm measurements |
| P04d1 | P03,C06.J | Benchmark water equilibration; implemented | One/31-case CPU evidence, original convergence criterion and exact untimed iteration counts; no speedup |
| P04d2 | P03,C17.J | Benchmark bin remapping; one-call evidence recorded | Preserve transfer order; distinguish one redistribution call from both complete correction passes |
| P04e | P04a,P04b,P04c,P04d1,P04d2,I01 | Benchmark original native process composition | Coupled parity passes; distinguish composed timing from sum of isolated times |
| P05 | B03.J | Add matched full-driver timing mode | Equal initialization, forcing, steps, diagnostics and output cadence |
| P06 | P05 | Compare full-driver repeated runs | Startup/I/O and compute totals separately reported with trajectories |
| P07 | P04e | Batch independent columns using `vmap` | Batch size 1 and heterogeneous ensembles match serial accepted cases |
| P08 | P07 | Benchmark 1/32/256/1024-column ensembles | Same Fortran workload; report throughput, latency, threads and memory |
| P09 | P06,P08 | Publish machine-scoped baseline report | Raw-data links, exact commands, limitations, no extrapolated speedups |

The current P04d2 adapter, `benchmarks/benchmark_remapping.py`, measures one
isolated `distr_update` call with equal physical case counts, checked/optimized
Fortran parity, untimed iteration checks, separate compilation and synchronized
warm timings. It excludes cleanup, hydration and the second SALSA pass.
The [remapping guide](../development/remapping.md) records one/47-case results
and raw measurements; no CPU speedup is observed. A one-call result does not close the planned
both-pass benchmark. P04e now separately measures accepted complete native
composition, including both corrections, interface conversions, density and
optional recalculation. Its equal-work adapter and raw evidence are documented
in the [composition guide](../development/salsa-composition.md). Native oracle
case/property/grid setup and JAX static-group dispatch are included; HAM driver,
initialization and evolving forcing are excluded. Full-driver timing remains
behind P05/P06.

## P5.2 — Optimize only measured bottlenecks

| ID | Depends on | Deliverable / suggested commit | Verification |
|---|---|---|---|
| P10 | P09 | Profile the dominant accepted kernel | Reproducible trace and identified cost; no speculative rewrite |
| P11 | P10 | Remove one measured allocation/transfer overhead | Same parity/budgets; repeated timing improves or change is rejected |
| P12 | P11 | Compile native process composition | No extra recompiles for changing runtime arrays; numerical gate passes |
| P13 | P12 | Introduce `lax.scan` for a fixed trajectory if beneficial | Every checkpoint matches; compile/runtime/memory tradeoff reported |
| P14 | P13 | Audit adaptive loops for JAX control-flow suitability | Original convergence/branch behavior preserved, explicit limits justified |
| P15 | P14 | Evaluate one available accelerator backend | Float64 support verified; include transfers, cold cost and CPU comparison |

Each additional optimization receives its own P11 child task. Keep a simple
accepted implementation available for diagnosing altered expression/loop order.
An accelerator task remains pending when the required hardware is unavailable.

## P6.1 — Maintainable documentation and CI

| ID | Depends on | Deliverable / suggested commit | Verification |
|---|---|---|---|
| Q01 | F04,F08,C01.J | Add quickstart and honest capability/status table | Fresh virtual environment builds/tests first leaf |
| Q02 | S07 | Add architecture/state/units guide | Named fields cross-linked to source and fixture contracts |
| Q03 | I04 | Add worked process-port tutorial | Contributor can add a leaf oracle without editing baseline physics |
| Q04 | F14 | Document case capture and fixture review | Regeneration command/provenance diff demonstrated |
| Q05 | P09 | Add benchmark reproduction and interpretation guide | Kernel/end-to-end scopes and cold/warm results unambiguous |
| Q06 | C01.J | Add mandatory compiler-enabled leaf parity CI | Missing compiler/zero oracle comparisons fail the job |
| Q07 | I03 | Add core process/trajectory CI | Every supported switch exercised; performance not gated on noisy hosts |
| Q08 | B03.J | Add full-box regression CI | Checked Fortran, process snapshots and JAX trajectory all run |
| Q09 | Q08 | Verify wheel/install and table packaging | Tests from outside source checkout; every required asset included |
| Q10 | Q09 | Audit attribution and supported-feature matrix | Preserve source notices; no missing branches advertised as supported |
| Q11 | Q05,Q10 | Prepare release/checkpoint report | Numerical evidence, known defects, benchmark data and setup documented |

## P6.2 — Optional differentiability research after parity

| ID | Depends on | One-commit deliverable | Verification |
|---|---|---|---|
| G01 | C01.J | Check coagc derivatives away from domain limits | Analytic/symmetry identities and finite-difference step-size sweep |
| G02 | I03 | Catalog discontinuities and nondifferentiable branches | Clipping, remap thresholds and adaptive termination explicitly listed |
| G03 | G02 | Validate one smooth process JVP | Directional finite differences on accepted cases |
| G04 | G03 | Define separate experimental smoothing/solver proposal | Scientific budgets and accuracy target specified before implementation |

Gradients are not a completion criterion for the compatibility port. No smoothing,
solver replacement or weakened parity tolerance enters the reference path merely
to make autodifferentiation convenient.
