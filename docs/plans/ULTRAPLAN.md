# SALSA reorganization and Python/JAX port: master plan

Prepared 2026-09-21 against Fortran revision
`d87dc072084336fd351d00a2ade863304c6f313d`.
Development now lives in `reflective-org/SALSA-JAX`, with the upstream Fortran
history retained and tagged `fortran-baseline-d87dc07`.
This is a staged research/software project, not a claim that the full model is
ported. See [execution status](STATUS.md) for completed work and measured results.

## 1. Objective and scope

Produce an organized, documented repository with a reproducible Fortran oracle,
independently callable Python/JAX aerosol processes, numerical parity with the
existing SALSA path, and honest CPU/ensemble performance measurements. Preserve
Fortran as an executable reference throughout the migration.

Initial assumptions: CPU float64 first; the existing SALSA branch is the primary
target; M7 remains a build/regression dependency rather than a second port.
Cloud activation, removal, radiation and the driver follow core microphysics.
VBS, aqueous chemistry, unsupported nucleation branches, accelerators and gradients
have separate gates. No numerical algorithm or physical constant is modernized
inside a compatibility port. Changes to suspicious existing behavior require a
separate reproducer, scientific rationale, and reviewed baseline change.

Current user priority: sulfuric-acid coagulation, condensation and nucleation.
The [focused experiment](../development/sulfuric-acid.md) now provides a runnable
core workflow and selected long trajectories. Further HAM/driver migration is
deferred from that immediate objective; the broader backlog below is retained.

## 2. Plans, subplans, and dependency gates

| Plan | Subplans | Entry → exit |
|---|---|---|
| [P0/P1: foundation and organization](01-foundation-and-organization.md) | provenance, build, diagnostics, source moves, configuration | current checkout → reproducible reference and clean layout |
| [P2/P3: state and core processes](02-state-and-core-processes.md) | contracts, grid, water, coagulation, condensation, remap, nucleation | leaf oracle → complete native SALSA call parity |
| [P4: coupled processes](03-coupled-processes.md) | HAM conversion, cloud, deposition, radiation, driver, optional chemistry | core parity → supported box-model trajectory parity |
| [P5/P6: performance and delivery](04-performance-and-delivery.md) | comparable timing, batching, optimization, CI, docs, optional gradients | accepted numerics → reproducible release evidence |

Critical path: P0 reference → P2 state/contracts → P3 water/coagulation/
condensation/remap → native composition → P4 HAM/driver → P5 full benchmark.
The independent `coagc` leaf can proceed while full-driver issues are diagnosed.
Source relocation waits for reproducibility and checked-build defect disposition;
it must never be mixed with a science change.

## 3. Commit-sized execution rules

Every task ID in the subplans is one intended commit. Its **deliverable** is the
change boundary, and its **verification** is the acceptance test. The row's action
is also a suggested commit subject. Dependencies are explicit; do not mark a
parent plan complete when only one child is done. Record actual commit hashes
after committing; uncommitted working-tree work is labeled as such.

For each process task: write down inputs/outputs, shapes, units, supported
branches and globals; capture original Fortran behavior; port the smallest
coherent kernel; compare; diagnose the earliest mismatch; document results.
The `.R` and `.J` tasks are separate commits for reference and JAX implementation.
Do not complete a `.J` task without an executed Fortran comparison. Benchmark
tasks depend on numerical acceptance, not merely a running implementation.

If an implementation row requires unrelated algorithms or more than one
independently reviewable formula family, split it into child IDs before coding.
Each child must compile/test independently; never leave a broken intermediate
commit. Keep fixture changes separate from tolerance changes. Tests target
scientific properties and reference behavior, not implementation details.

## 4. Planned layout (create directories when they have content)

```text
src/salsa_jax/             Python state, constants and pure process kernels
  processes/              coagulation, condensation, nucleation, water, remap…
fortran/                  original source after P1 relocation gates
  src/                    initially a flat source set to minimize move risk
  include/                namelist declarations
  drivers/                box driver after separate build verification
configs/                  checked run/build settings and named scientific cases
tools/                    build, provenance and reference capture commands
tests/                    unit, Fortran parity, process and trajectory checks
tests/fixtures/           small versioned cases with manifests
benchmarks/               reproducible equal-work timing entry points
docs/                     architecture, process contracts, plans and results
input/                    existing bulk scientific inputs, retained initially
data/                     existing sample data, retained until provenance review
build/                    ignored binaries and isolated reference runs
outputs/                  ignored bulk diagnostics and benchmark artifacts
```

No mass rename by filename prefix until dependency analysis and input-path tests
pass. Preserve bytes and license headers when moving source. Keep historical
sample data until its purpose is established; do not classify it as disposable
just because the extension is `.dat`.

## 5. Success criteria

1. Fresh documented commands reproduce reference runs and identify compiler,
   source/config/input hashes and effective process switches.
2. All supported process branches have reviewed cases, physical budgets,
   finite-state checks, and calibrated per-field tolerances.
3. Eager, JIT and batched JAX agree with the oracle at process boundaries, one
   complete call, and short/long trajectories. No skipped oracle passes the gate.
4. Kernel and end-to-end benchmarks use equal work, synchronized results and
   repeated samples; compilation/startup/I/O costs are separately visible.
5. Documentation supports a new contributor from environment setup to adding a
   process, reproducing a mismatch, and regenerating benchmark evidence.

## 6. Collaboration and iteration

Use an integration branch with one short-lived branch/worktree per independent
task. Put worktrees beside the checkout when permissions allow, or in an ignored
workspace location; do not edit another task's files. Commit the shared foundation
before branching dependent tasks. Parallel agents may handle reference capture,
an independent accepted-contract kernel, or documentation; a coordinator owns
interfaces, dependency order and integrated verification. Do not parallelize
sequential bin-transfer algorithms by changing their scientific execution order.

Each work cycle is: select ready task → state acceptance → implement → run oracle
and property checks → inspect diff → document result → commit → integrate and
rerun affected gates. A failed gate creates a minimal reproducer and a discrepancy
entry; it never causes automatic relaxation of tolerances or fixture replacement.

## 7. Evidence and risks

Read the [architecture audit](../development/architecture.md),
[local-port lessons](../development/local-port-lessons.md), and
[validation/benchmark protocol](../development/validation-and-benchmarks.md).
Key risks are mixed units, persistent module state, dependent bin updates,
nucleation nested in condensation, iterative convergence, provisional driver
inputs, unsupported branches, and low-precision legacy diagnostics.
Each has an explicit reference/contract/validation task below rather than an
assumed fix. Performance is measured; no speedup target is promised in advance.
