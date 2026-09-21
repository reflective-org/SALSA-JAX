# Execution status

Updated 2026-09-21. Baseline HEAD:
`d87dc072084336fd351d00a2ade863304c6f313d`.
Development repository: [reflective-org/SALSA-JAX](https://github.com/reflective-org/SALSA-JAX).
Reference repairs and initialization capture extend the initial implementation below.
The upstream Fortran revision is pinned by tag `fortran-baseline-d87dc07`.
The [master plan](ULTRAPLAN.md) defines the remaining commit-sized backlog.

## Completed initial work

| IDs | State | Evidence / scope |
|---|---|---|
| F01–F03 | Documented | Architecture audit, four local-port studies, linked phase/subplans and acceptance tasks |
| F04 | Implemented | `.gitignore`, `pyproject.toml`, tested dependency snapshot, fresh `.venv` and editable install |
| F05 | Implemented | Isolated source/input copies, build/run logs and manifests; setup-failure regression test |
| F06 | Verified | Added three missing VBS dependency rules in Makefile and `.depend`; clean serial/parallel builds pass |
| F07 | Smoke checks pass | Two fresh compatibility runs and parallel-build run have identical hashes for all nine diagnostic files |
| F08 | Reference gate still fails | D001–D004 repaired; fresh checked run now fails in standalone radiation (D005) |
| F09/F10 | Implemented | Historical condensation failure retained; absent-gas guard preserves all nine compatibility hashes |
| F10a | Implemented | Organic fraction initialized with nucleation off; positive-organic fixture and budgets pass |
| F09b/F10b | Implemented | Activation velocity rank/units reproducer and correction; eight tests |
| F09c/F10c | Implemented | Water-output extent reproducer and allocation for active class count; three tests |
| F11a | Initialization capture verified | Copy-only overlay; typed dimensions/switches/gas mapping; nine diagnostics identical with/without capture |
| S01 | Partial, first leaf only | Legacy constants and float64 policy for `coagc`; broader state/kind readback remains planned |
| C01.R / C01.J | Leaf parity passes | Verbatim original-Fortran oracle; four tests, 1734 cases; maximum relative error 8.95e-15 |
| P01/P02 | Preliminary leaf evidence | Wall-to-wall batch timing with source/case hashes; no full-process or model claim |
| Q01 | Implemented | Quickstart, capability limits and reproduction commands |

Total current suite: **48 passed** in this repository's `.venv`. Original inputs
remain unchanged; Fortran scientific/driver repairs are separate commits. Generated
builds/outputs remain ignored. Source relocation has not started. The pre-existing
`AGENTS.md` remains unchanged; this ledger and the repair record carry current
validation status.

## Next ready tasks, in order

1. **F09d:** reproduce missing standalone radiation initialization/table data
   (D005); inspect supplied optical inputs and initial diagnostic values. Define
   a supported full-driver case or an explicitly scoped microphysics reference
   before adding another repair. Never fabricate optical values.
2. **F09e/F10e:** isolate nucleation's absent-organic-gas read and repair supported
   sulfate-only modes with a narrow checked fixture before enabling nucleation.
3. **F11b/F12:** extend metadata to complete state/tracer contracts and capture
   full-precision SALSA entry/exit. Initialization capture is not timestep evidence.
4. **F13–F16:** capture separate coagulation, condensation and nucleation process
   states, validate budgets, repeat/reset behavior and configuration isolation.
5. **C02.R/C02.J:** port grid geometry after its state/bin contract is captured;
   then follow the independent process tasks in the core plan.
6. **O01–O04:** improve build portability and cleanup; mechanical source moves
   still depend on passing reference gates.

Coagulation, condensation and nucleation each have separate Fortran oracles,
Python kernels, tests and benchmarks. Condensation is first isolated with
`nsnucl=0`; nucleation rate/survival/insertion is validated independently before
reintroducing the original nested condensation call. Complete coagulation is
still pending: only its collision-coefficient leaf is ported.

## Initial implementation commits

| Commit | Scope |
|---|---|
| `b5954a9` | Python package, ignore rules and isolated environment dependencies |
| `ff9eb5c` | VBS module dependency edges |
| `725674a` | Isolated Fortran audits, configs and failure-manifest test |
| `e815e02` | Original Fortran coefficient oracle and case generator |
| `3953e7f` | JAX coefficient/constants and parity tests |
| `17848d8` | Scoped coefficient benchmark |

The repository-launch documentation commit adds this ledger, the roadmap,
reviewed historical evidence and SALSA-JAX quickstart. Future commits should
record their task IDs, checks and result here as each gate passes.

A fresh `.venv` was created in `SALSA-JAX`; the initial five tests passed there, with
imports resolving to this checkout. Its isolated full Fortran compatibility
run also completes and produces the same diagnostic hashes as the original
checkpoint. The original SALSA working directory was left unchanged during
repository creation and these subsequent repairs. Historical benchmark evidence retains its original
provenance and has not been relabeled as a new benchmark run.

## Known limits

The checked full-model failure prevents declaring a scientifically validated
trajectory baseline. Matching formatted diagnostics does not resolve undefined
memory access. The scalar coefficient port is independently tested and does not
call the defective condensation path. GPU, complete coagulation, other processes,
full-model JAX parity, gradients and OpenIFS integration are not implemented.

See [recorded results](../development/results-2026-09-21.md) for numerical and
timing evidence, and [quickstart](../development/quickstart.md) for commands.

## Reference repair commits

| Commit | Task / scope |
|---|---|
| `14744f2` | F09: preserve the original missing-organic-gas failure |
| `4b5adaa` | F10: guard optional organic-gas condensation |
| `1de4bd7` | F10a: initialize organic fraction with nucleation off |
| `e659216` | F09b/F10b: activation input shape and pressure-velocity units |
| `81f24b5` | F09c/F10c: allocate aerosol-water output for active classes |
| `64db941` | F11a: copy-only effective runtime metadata and validation |

The [repair record](../development/reference-repairs.md) explains scientific
scope and remaining defects. The [runtime capture guide](../development/runtime-metadata.md)
links initialization evidence and the instrumented/uninstrumented comparison.
The current ordinary outputs differ from the original in five diagnostic files;
this is recorded as a baseline change, not scientific acceptance. No new
performance claim is made by these diagnostic runs.
