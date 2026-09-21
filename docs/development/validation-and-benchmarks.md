# Validation and benchmark protocol

**Status: planned acceptance requirements.** This document does not claim that
SALSA has implemented the harness, captured approved fixtures or passed JAX
parity/performance gates. Record actual runs and completion in the project status.

## Reference and precision

Freeze source revision plus dirty-file hashes, input/configuration hashes,
compiler and NetCDF versions, preprocessing definitions, flags and platform.
Keep build-time instrumentation patches separate and hash them too. Compare an
instrumented run against an uninstrumented run before trusting captured states.
Use independent run directories so diagnostics never overwrite reference inputs.

The first target is CPU float64. `mo_kind.F90` aliases `dp` to `JPRB`, declared
as `SELECTED_REAL_KIND(13,300)` in `parkind1.F90`; verify kind/storage size in the
built oracle rather than assuming compiler defaults. Preserve literal evaluation
and cast order. Set JAX x64 at startup and assert input/output dtypes; otherwise
requests for double precision can be truncated. See the official
[JAX precision and programming constraints](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).

Maintain a checked build with bounds/runtime checks and a release build for
timing. Verify the release build against the checked baseline. Record compiler
optimization and floating-point options; do not silently enable fast-math.

## Gates, in order

| Gate | Required evidence |
|---|---|
| Reference | Repeat fresh-process runs reproduce the baseline; effective configuration is recorded. |
| Data contract | Species/bin indices, shapes, units, initialization and process switches match exactly. |
| Leaf kernel | Identical captured inputs produce matching outputs and all mutated state. |
| Process | Complete process calls match, including substeps, coupled gas changes and diagnostics. |
| Composition | One SALSA call matches at every process boundary before checking final output. |
| Trajectory | Short and longer identical-forcing runs match with documented error growth and budgets. |
| Transformations | JIT and batched execution retain approved eager behavior. |
| Performance | Only numerically accepted cases enter speed comparisons. |

For each process, capture before/after arrays, timestep, meteorology, options,
active indices and relevant module state. Restore inputs between leaf calls.
Check process-off behavior and inactive bins. Preserve SALSA's actual sequence,
including repeated equilibration and distribution updates, and explicitly
capture nucleation within condensation rather than assuming independent switches.

Use representative initialized states plus constructed cases: empty and nearly
empty bins, a single populated bin/species, dry/wet conditions, gas depletion,
bin-boundary transfers, clipping thresholds and every supported nucleation mode.
Select temperature/humidity endpoints from supported source ranges. Record
which branches fixtures reach; a trajectory alone is not branch coverage.

## Errors, budgets and acceptance

Report each field's maximum absolute error, scaled relative error, RMS error,
worst index and nonfinite count. Gate elementwise with
`abs(jax - reference) <= atol[field] + rtol[field] * abs(reference)`.
Choose absolute tolerances in documented physical units; relative error alone
is meaningless near zero. Exact gates apply to discrete indices and decisions.

Calibrate candidate tolerances using repeated oracle runs, precision checks,
debug/release differences and measured roundoff before approving each process.
Commit the tolerance rationale separately from changed fixtures. Do not loosen
thresholds to hide an unexplained mismatch; retain the smallest reproducer and
identify the earliest divergent process, branch or arithmetic expression.

Check species/element mass with gas-aerosol exchange, sources and sinks included;
check number budgets according to each process. Coagulation consumes number,
nucleation creates it, and transfer may move it between bins. Account for water,
clipping and reference corrections explicitly. Reject unexpected NaN/Inf and
unphysical values; report reference defects instead of silently repairing them.

## Fixture provenance and mandatory oracle execution

The planned manifest records fixture/case IDs, schema version, capture command,
source/build/config hashes, compiler flags, platform, precision, process and call
position; each array has its name, units, dtype, shape and content checksum.
Capture in fresh subprocesses until state-reset equivalence is demonstrated.
Read effective switches back and ensure intended case changes actually occurred.

Store small reviewed fixtures with the tests; keep bulk diagnostics and timing
artifacts outside tracked scientific inputs. Detect changed, missing and unlisted
fixtures. Regeneration requires an explicit command and reviewable provenance
diff; normal tests must never rewrite expected results.

A lightweight suite may mark oracle-dependent tests separately, but the required
parity job must fail on missing compiler/library/wrapper, zero executed oracle
cases or unexpected skips. Report executed/passed/failed/skipped counts and the
oracle identity. A green pure-Python suite is not evidence of Fortran parity.

## Timing protocol

Measure equal precision, configuration, simulated duration, output cadence and
workload. Separate compilation/first call from warmed execution, place arrays
before kernel timing, and synchronize all result leaves before stopping the
clock. These address asynchronous dispatch and compilation effects described in
the official [JAX benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html).

Publish three labeled measurements: (1) in-process Fortran/JAX process kernels
with equivalent input/reset/copy costs, (2) complete simulations with matched
startup, initialization and I/O scope, and (3) ensemble throughput versus a
Fortran ensemble of the same size. State whether bindings are included; measure
binding overhead for very short kernels. Never label subprocess-versus-hot-JIT
timing as a pure kernel speedup.

Use a monotonic high-resolution clock; retain raw repeated samples, repetition
counts, median and spread. Warm each shape/configuration, distinguish persistent
cache state, and record hardware, thread settings, memory, JAX/jaxlib versions,
device/backend and compiler flags. Alternate measurement order to expose drift.
Report initialization, transfers, compile latency and peak memory separately.
Keep performance claims scoped to the measured machine and workload.

## Gradients and alternative solvers: later gates

After forward parity, compare selected derivatives with analytic identities or
finite differences over a step-size sweep away from discontinuities. Test JVPs
or directional derivatives for larger states. Document undefined or discontinuous
behavior at bin transfers, clipping and branch thresholds. Alternative solvers
and smoothed physics require separate accuracy/conservation studies and must
not replace the compatibility path merely to obtain gradients or speed.
