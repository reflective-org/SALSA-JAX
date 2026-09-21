# Execution status

Updated 2026-09-21. Baseline HEAD:
`d87dc072084336fd351d00a2ade863304c6f313d`.
Development repository: [reflective-org/SALSA-JAX](https://github.com/reflective-org/SALSA-JAX).
Reference repairs and native state capture extend the initial implementation below.
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
| F09d | Audit complete | Both optical tables rejected by NetCDF; standalone initialization/path/order issues documented |
| F10d | Explicit scope verified | Checked first microphysics call completes with six finite arrays; full driver remains unvalidated |
| F09e/F10e | Implemented | D006 absent-organic nucleation guard; 29 checked regressions, original/current equality and coupled sulfate budgets |
| F11a | Initialization capture verified | Copy-only overlay; typed dimensions/switches/gas mapping; nine diagnostics identical with/without capture |
| F09f/F10f | Implemented | D008: preserve registered component maps; original overwrite reproduced, all 69 identities and initial volumes checked |
| F11b | Partial | Native material/component/gas/number maps and species/tracer table captured; remaining HAM wrapper tables/flags documented |
| F12 | Default first call verified | Full-precision native entry/exit, geometry and units; six HAM arrays identical with capture on/off |
| F14a/F15a/F16a | Default first-call checks pass | Hash-verified fixture, native conversions/material budgets and byte-identical fresh-process repeat |
| S01 | Standalone process subset | Legacy constants and float64 policy for coagulation/condensation; compiled runtime values checked, broader model constants remain pending |
| C01.R / C01.J | Leaf parity passes | Verbatim original-Fortran oracle; four tests, 1734 cases; maximum relative error 8.95e-15 |
| P01/P02 | Preliminary leaf evidence | Wall-to-wall batch timing with source/case hashes; no full-process or model claim |
| F13b | Default process boundary verified | Coagulation inputs after hydration and outputs before condensation; capture/control and fresh repeat match exactly |
| S02/S04–S06 | Coagulation/condensation subset | Fixed 17-bin/five-material layout, immutable state return, explicit geometry/environment and enabled flag; full state API remains pending |
| C07–C11.R/J | Standalone numerical parity passes | Original process/coefficient extraction matches baseline tag; 31 cases, actual capture, eager/JIT and 20-call sequences |
| C11.D009.R | Legacy defect reproduced | Minimal 2a/2b pair creates volume; preserved in JAX and documented, scientific repair pending |
| P03/P04a | Initial scoped process measurement | One and 31-case CPU timings against optimized Fortran; no measured speedup |
| F13c | Condensation subset verified | Actual conditional condensation boundary at nsnucl=0; capture/control and fresh repeat exact, nested nucleation remains pending |
| C12–C14.R/J | Standalone numerical parity passes | Repaired Fortran oracle, 29 cases, eager/JIT, species budgets and 20 calls; organic fixtures explicit |
| P04b | Initial scoped process measurement | One and 29-case CPU timings against optimized Fortran; no measured speedup |
| Q01 | Implemented | Quickstart, capability limits and reproduction commands |

All **242 current tests pass** in this repository's `.venv`: the full suite
passed 241, followed by 14 condensation tests including the added two-process
integration case. Original inputs
remain unchanged; Fortran scientific/driver repairs are separate commits. Generated
builds/outputs remain ignored. Source relocation has not started. The pre-existing
`AGENTS.md` remains unchanged; this ledger and the repair record carry current
validation status.

## Next ready tasks, in order

1. **N01–N07 / remaining F13c:** port nucleation independently, starting with
   the supported rate schemes, then growth/survival and aerosol insertion.
   Reintroduce the nested condensation call only after separate acceptance.
2. **C11.D009.A/F/J:** review and repair the inherited cross-regime volume
   creation in a separate scientific change; do not silently alter this port.
3. **C02/C04–C06 / F13a:** port grid generation and hydration for coupled
   execution. Coagulation and condensation accept their already computed outputs.
4. **F13d / F14–F16:** capture remapping and complete process-switch/reset,
   configuration and trajectory acceptance. Finish F11b HAM wrapper metadata.
5. **D007:** reproduce and guard absent-organic access in tiny-particle cleanup
   before accepting empty/tiny-volume full-native cases.
6. **D005:** obtain valid optical tables and repair standalone radiation before
   full-driver acceptance. Source relocation still depends on reference gates.
7. **O01–O04:** improve build portability and artifact cleanup separately.

Processes are tracked independently. Coagulation and condensation have JAX
kernels, separate Fortran oracles, tests and benchmarks. Condensation currently
uses `nsnucl=0`; nucleation rate/survival/insertion will be validated before
restoring coupling. Hydration and nucleation JAX ports remain pending; this is
not yet a coupled SALSA simulation.

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
depend on the full-driver radiation path. The named first-microphysics scope
completes one finite interface call, not a full timestep or trajectory.
GPU, other complete processes, full-model JAX parity, gradients and OpenIFS
integration are not implemented. Coagulation preserves the native D009 volume
creation in stiff cross-regime cases; matching Fortran is not unrestricted
scientific acceptance.

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
| `0dd5281` | F09e: original absent-organic failure in sulfate nucleation |
| `4a04894` | F10e: zero absent-organic vapor without changing registered-gas behavior |
| `68b14e7` | Nucleation boundary documentation and checked numerical evidence |
| `786e854` | F10d: explicit checked first-microphysics scope and completion checks |
| `0f843dd` | F09f/F10f: preserve registered component maps during initialization (D008) |
| `50d190f` | Full-precision HAM arrays for capture on/off comparison |
| `f27fa5f` | F12/F14a: native state capture, checked fixture and strict map validation |

The [repair record](../development/reference-repairs.md) explains scientific
scope and remaining defects. The [runtime capture guide](../development/runtime-metadata.md)
links initialization evidence and the instrumented/uninstrumented comparison.
Before D008, ordinary outputs differed from the original in five diagnostic
files. D008 changes initialization again; no full-trajectory comparison for
that corrected baseline is claimed here. No new
performance claim is made by these diagnostic runs.

The D006 nucleation repair preserves all nine default ordinary-run diagnostics
exactly; the default has nucleation disabled. Its separate kernel tests cover
schemes 1/2/3 with growth options 1/2/3, registered-zero and positive organics,
zero sulfate, and coupled sulfate budgets. See the
[nucleation reference](../development/nucleation-reference.md) for scope limits.

## Native capture milestone

The [capture guide](../development/native-state-capture.md) and
[state contract](../development/salsa-state-contract.md) describe the reviewed
fixture and remaining wrapper metadata. Three checked runs establish native
capture on/off invariance and fresh-process repeatability. Native budgets retain
the original salt-to-sulfate volume relabeling approximation. See the
[source/configuration evidence](../development/evidence/native-state-2026-09-21.json).
F14a/F15a/F16a cover only this native fixture; the parent process/trajectory
gates remain open. This native-capture milestone preceded the complete coagulation port below.

## Complete coagulation milestone

The [process API and contract](../development/coagulation-process.md) cover the
full standalone routine, accepting supplied wet diameters and dry geometry.
Its separate original-Fortran oracle, actual captured boundary, 31-case matrix,
strict checked builds and repeated-call tests establish numerical parity.
The native D009 volume-budget discrepancy remains explicitly unrepaired.

The CPU benchmark shows no speedup: JAX/Fortran median warm times are
42.51/4.11 µs for one captured case and 132.09/123.61 µs for 31 cases.
Maximum number/volume relative error is below 2e-15. Full provenance and limits
are in the [integration record](../development/evidence/coagulation-process-2026-09-21.json)
and linked benchmark artifacts. The subsequent condensation milestone is below;
nucleation remains unported.


| Commit | Coagulation deliverable |
|---|---|
| `887c386` | F13b: checked process capture and fixture |
| `57dc829` | C07–C11.R: verbatim Fortran oracle, 31 cases and D009 reproducer |
| `104c82b` | C07–C11.J: complete standalone JAX coagulation and comparisons |
| `ddbe581` | P04a: scoped CPU benchmark with raw measurements |

## Standalone condensation milestone

The [condensation API and contract](../development/condensation-process.md)
cover sulfate and optional nonvolatile organic uptake with nucleation disabled.
Actual native capture, 29 oracle cases, species budgets, eager/JIT parity and
20 successive fixed-diameter calls pass. A separate check connects the two
ported processes against their captured boundaries. Full SALSA, hydration,
nucleation and remapping acceptance remain open.

JAX/Fortran median warmed CPU times are 10.10/0.263 µs for the captured case
and 35.31/6.40 µs for 29 cases. Number/gas outputs match exactly; maximum volume
relative error is about 6e-16. The
[integration evidence](../development/evidence/condensation-process-2026-09-21.json)
and linked raw benchmarks retain provenance and timing limitations.


| Commit | Condensation deliverable |
|---|---|
| `d2169ad` | F13c subset: checked condensation capture and fixture |
| `6b41e42` | C12–C14.R: repaired Fortran oracle and 29 cases |
| `d06f7a2` | C12–C14.J: standalone JAX condensation and comparisons |
| `0e6ea69` | P04b: independent CPU benchmark with raw measurements |
