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
| F13c | Default and active boundaries verified | Default condensation plus nested nucleation/parent condensation for explicit scheme 2/growth 1; capture/control and fresh repeat exact |
| C12–C14.R/J | Standalone numerical parity passes | Repaired Fortran oracle, 29 cases, eager/JIT, species budgets and 20 calls; organic fixtures explicit |
| P04b | Initial scoped process measurement | One and 29-case CPU timings against optimized Fortran; no measured speedup |
| N01–N03.R/J | Rate leaves pass | Actual Fortran leaf extraction; 216 cases, binary/activation/kinetic and boundary semantics |
| N04–N07.R/J | Supported isolated process passes | 110 cases, growth options 1–3, vapor limitation, Taylor population and source dispatcher |
| N08a.R/J | Process coupling passes | Actual condensation call order, particle insertion, species budgets, ten fixed-diameter calls; parent N08 remains open |
| F13c.N1–N3 / N08b.R/J | Active native acceptance passes | Copy-only profile, active nested/parent fixtures, exact capture invariance and actual Fortran/JAX parity |
| P04c | Initial scoped measurements | Matched two-cell workloads, isolated/coupled, nine groups; no measured CPU speedup |
| C02.R/J | Native grid passes | Four ordered limit sets; all five fields bit exact, preserving binary32 spacing fractions |
| C04–C06.R/J | Standalone hydration passes | Actual molality/equilibration oracle; 31 cases, exact stopping counts, eager/JIT float64 parity |
| F13a | Initial native hydration verified | Copy-only capture/control and fresh repeat exact; actual material registry replay and pre-remap JAX chain pass |
| P04d1 | Initial scoped hydration measurement | One and 31-case CPU timings; exact untimed iteration checks, no measured speedup |
| Q01 | Implemented | Quickstart, capability limits and reproduction commands |

All **459 current tests pass** in this repository's `.venv` (50.13 seconds).
The latest 75 checks cover native grid arithmetic, inverse molality, hydration,
checked iteration diagnostics, capture integrity and pre-remap composition.
Original Fortran sources and scientific inputs are unchanged by this milestone.
Generated builds/outputs remain ignored. Source relocation has not started.
The pre-existing `AGENTS.md` remains unchanged; this ledger and the repair record
carry current validation status.

## Next ready tasks, in order

1. **C15–C17 / F13d:** capture and port native remapping decisions, transfers
   and radius corrections, retaining sequential order and zero-bin policy.
2. **C18 / N08 / F14–F16:** compose the complete native process order, including
   optional diameter recalculation and both remap passes; accept process switches,
   active nucleation and repeated trajectories. Finish F11b HAM wrapper metadata.
3. **C03:** port the seven-mode `size_distribution` initialization independently
   before claiming a standalone JAX box model.
4. **C11.D009.A/F/J:** review and repair inherited cross-regime volume creation
   as a separate scientific change; preserve compatibility until that decision.
5. **D007/D010:** resolve tiny-particle cleanup's absent-organic access and
   invalid active zero-volume hydration before accepting those native inputs.
6. **D005:** obtain valid optical tables and repair standalone radiation before
   full-driver acceptance. Source relocation still depends on reference gates.
7. **O01–O04:** improve build portability and artifact cleanup separately.

Processes are tracked independently. Grid generation, hydration, coagulation,
condensation and supported nucleation have JAX kernels and separate Fortran
references. Hydration, coagulation and condensation now compose against the
actual default pre-remap boundaries, with `recalc=False`. Nucleation's active
native process acceptance is separate. Remapping and complete SALSA composition
remain pending; this is not yet a complete coupled SALSA simulation.

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
nucleation was subsequently ported in the milestone below.


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
ported processes against their captured boundaries. Full SALSA and remapping acceptance remain open. Hydration and nucleation
were subsequently ported below.

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

## Supported nucleation and condensation coupling milestone

The [nucleation guide](../development/nucleation-process.md) distinguishes
critical-rate leaves, isolated molecular sources and the coupled condensation
update. N01–N07 support schemes and growth options 1–3. N08a restores vapor
uptake and first-bin particle insertion with explicit initialized fixtures;
N08b native capture is verified in the next milestone; parent N08 complete-SALSA
acceptance remains open.

The complete native-module oracle covers 110 cases, including nonzero binary
sources for every growth option. Checked/optimized outputs match exactly.
Eager source error is below 1.74e-12; paired JIT source/coupled-state errors are
below 5.68e-15/4.36e-15. Species budgets and ten-call fixed-diameter sequences
pass. Native rate/growth quirks and the empty-background insertion gate are
preserved and documented; no production Fortran edits accompany this port.

All recorded small CPU groups are slower than Fortran. For example, scheme 2,
growth 3 over 24 physical cells takes 59.28/49.63 µs for JAX/Fortran isolated
sources and 73.47/53.33 µs for coupled uptake. Timings match both native levels
per case and disclose reset/dispatch costs. See the [integration record](../development/evidence/nucleation-process-2026-09-21.json)
and linked raw measurements.

| Commit | Deliverable |
|---|---|
| `df8342b` | N01–N08a.R: complete native oracle and 110-case matrix |
| `03f13b0` | N01–N03.J: binary/activation/kinetic rate leaves |
| `bb270f2` | N04–N07.J: growth/survival/Taylor and source dispatcher |
| `519a697` | N08a.J: condensation coupling and insertion |
| `5fa043f` | P04c: matched isolated/coupled CPU benchmark and raw measurements |

## Active native nucleation milestone

The [capture guide](../development/nucleation-native-capture.md) and
[evidence record](../development/evidence/nucleation-native-2026-09-21.json)
document three strict checked first-call runs. The profile sets scheme 2,
growth 1 and sulfate 1e14 molecules/m³ only in the copied driver, because the
standalone build ignores the SALSA namelist and default first-call sulfate is
below the nucleation cutoff. All other initialization and forcing are retained.

Capture/control HAM arrays, runtime metadata and ordinary diagnostics match
exactly; captured native/nucleation/condensation boundaries repeat exactly.
The molecular source is 1.6066412692623088e8 molecules/m³/s, and first-bin
number increases by 1.0077747032660246e6 particles/m³. The independent Fortran
oracle and eager/JIT kernels match these actual boundaries; species budgets
pass. No production Fortran or JAX science code changed. This validates one
active native process case, not the complete JAX SALSA timestep or trajectory.

| Commit | Deliverable |
|---|---|
| `a8809ea` | F13c.N1: controlled first-call profile and isolated configuration |
| `f571960` | F13c.N2: nested nucleation and parent condensation capture |
| `736c048` | F13c.N3: checked active fixture pair and capture failure checks |
| `d60258f` | N08b.R/J: actual-boundary replay, adapter and JAX acceptance |

## Grid and hydration milestone

The [grid/hydration contract](../development/grid-and-hydration.md) covers C02
and C04–C06. Native binary32 grid fractions are preserved; all five geometry
fields match exactly. Actual Fortran inverse molality and equilibration cover
31 hydration cases. Eager/JIT core and wet-diameter errors remain below 4e-16
relative, and all iteration counts match (maximum four in this matrix).

Three checked native runs establish capture invariance and exact fresh repeats.
The new JAX grid → hydration → coagulation → condensation test supplies computed
geometry and wet diameters; number and gas match exactly and material-volume
error is below 2.19e-16 relative. This default `nsnucl=0`, `recalc=False` chain
ends before remapping. C18/N08/I01 remain open. D010 records a native zero-volume
hydration failure; the JAX API signals NaN for that invalid domain.

Warm JAX/Fortran CPU medians are 29.14/1.346 µs for one curated native-final-state
case and 53.75/38.72 µs for 31 cases; no speedup is claimed. See the
[integration evidence](../development/evidence/grid-hydration-2026-09-21.json)
and linked benchmark artifacts for hashes, checks and limitations.

| Commit | Deliverable |
|---|---|
| `b92dca5` | C02.R: verbatim native grid oracle |
| `09ed944` | C02.J: grid geometry preserving binary32 spacing |
| `5e8446a` | C04–C06.R: molality/hydration oracle and iteration diagnostics |
| `5348ac0` | C04–C06.J: JAX molality and equilibrium hydration |
| `408ec37` | F13a: copy-only initial hydration capture |
| `245496b` | F13a: checked fixture and required capture checks |
| `4ef4eb7` | C06 native adapter and composed pre-remap acceptance |
| `ecf7b8a` | P04d1: matched hydration benchmark and raw evidence |
