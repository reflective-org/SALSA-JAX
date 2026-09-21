# Execution status

Updated 2026-09-21. Baseline HEAD:
`d87dc072084336fd351d00a2ade863304c6f313d`.
Development repository: [reflective-org/SALSA-JAX](https://github.com/reflective-org/SALSA-JAX).
The initial implementation is committed in focused changes listed below.
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
| F08 | Audit complete; reference gate fails | Checked build succeeds; run stops on D001, index 0 in organic-gas condensation |
| S01 | Partial, first leaf only | Legacy constants and float64 policy for `coagc`; broader state/kind readback remains planned |
| C01.R / C01.J | Leaf parity passes | Verbatim original-Fortran oracle; four tests, 1734 cases; maximum relative error 8.95e-15 |
| P01/P02 | Preliminary leaf evidence | Wall-to-wall batch timing with source/case hashes; no full-process or model claim |
| Q01 | Implemented | Quickstart, capability limits and reproduction commands |

Total current suite: **5 passed**, comprising four leaf tests and one reference
setup-failure test. Original `.F90` files and scientific input data are unchanged.
Generated builds/outputs remain ignored. Source relocation has not started.

## Next ready tasks, in order

1. **F09:** minimize D001 using the inactive organic-gas configuration. Trace
   `isubm_ocnv` registration and `locgas`/SOA control flow; retain a checked failing
   case. Full reproducer already exists in the checked audit command.
2. **F10:** repair the invalid access in one isolated commit. Do not use Fortran
   `.AND.` as a short-circuit guard; verify indices before referencing the array.
   Compare compatibility outputs and investigate any scientific difference.
3. Repeat checked-run triage for any newly exposed independent defects; create
   numbered F09/F10 child tasks instead of a broad cleanup patch.
4. **F11–F16:** capture effective configuration and full-precision process states,
   establish fixture integrity and numerical gates.
5. **O01–O04:** improve root build portability/regeneration/cleanup, then permit
   mechanical source moves only when their reference gates pass.
6. **C02.R/C02.J:** grid geometry is the next independent numerical port, after
   the required state/bin contract is captured. Follow the core plan thereafter.

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

A fresh `.venv` was created in `SALSA-JAX`; all five tests pass there, with
imports resolving to this checkout. Its isolated full Fortran compatibility
run also completes and produces the same diagnostic hashes as the original
checkpoint. The original SALSA working directory was left unchanged during
repository creation. Historical benchmark evidence retains its original
provenance and has not been relabeled as a new benchmark run.

## Known limits

The checked full-model failure prevents declaring a scientifically validated
trajectory baseline. Matching formatted diagnostics does not resolve undefined
memory access. The scalar coefficient port is independently tested and does not
call the defective condensation path. GPU, complete coagulation, other processes,
full-model JAX parity, gradients and OpenIFS integration are not implemented.

See [recorded results](../development/results-2026-09-21.md) for numerical and
timing evidence, and [quickstart](../development/quickstart.md) for commands.
