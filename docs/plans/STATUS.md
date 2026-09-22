# Execution status

Updated 2026-09-22. Baseline HEAD:
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
| C15–C16.R/J | Ordered remapping passes | 47 cases; exact strict soluble-threshold decisions, material/number budgets and two-call comparisons |
| C17.R/J | Native correction passes | 54 cleanup/hydration/radii-water cases; D007 reproduced, no clipping or scientific repair |
| F13d | Both native passes verified | Eight capture stages; control/repeat exact; remapping and interleaved corrections match actual boundaries |
| P04d2 | Initial isolated remapping measurement | Matched one-call CPU workload; corrections and second pass excluded |
| C18a–c.R/J | Native interface passes | Initial cleanup, all69 mixed-unit mass components, runtime maps and per-region density; checked original blocks |
| C18d / N08 | Supported native composition passes | Complete call order, optional recalculation, active nucleation, both correction passes and final output |
| I01 | Selected complete-call matrix passes | 27 cases × two levels, all coagulation/condensation/recalc combinations, supported nucleation scheme/growth pairs |
| I02 | Short fixed-environment acceptance | Three native mass-interface calls in six configurations; depleted soluble residual has an explicit conditioning bound |
| F14–F16 | Expanded scoped capture acceptance | Active, recalc and active+recalc control/capture/repeat exact; all four default/configured call boundaries match JAX |
| P04e | Initial complete-call measurement | Matched native/JAX workloads, seven caller outputs, checked/optimized parity, separate compilation and warm execution |
| C03.R/J | Seven-mode number initializer accepted | 23 original-Fortran cases, eager/JIT/vmap, actual checked driver boundary and exact capture/control/repeat |
| H01.R/J1 | Independent gas-conversion leaf accepted | All three native codes, precision, runtime tracer mapping, duplicate writes, unchanged tracers and native padding |
| H01.F/J2 | Actual gas caller accepted | Default/active first gas2mmr boundaries, complete tracers, full-cell meteorology, preserved driver scaling |
| H02.R/J | Material/tracer initialization accepted | 18 original-Fortran cases, all 69 materials, native fractions/constants/number mutation, duplicate writes and budgets |
| H02 native composition | Initialization chain accepted | Six checked control/capture/repeat runs; linked number → material → gas JAX chain matches default/active boundaries |
| P04g | Initial material-initialization measurement | One/18 curated cases, three outputs synchronized, checked/optimized parity; no measured CPU speedup |
| Q01 | Implemented | Quickstart, capability limits and reproduction commands |

All **837 tests pass** in this repository's `.venv` (137.39 seconds), including
mandatory original-Fortran comparisons, native number/material initialization
and first gas conversion. Original Fortran sources,
scientific inputs and `AGENTS.md` are unchanged. Generated builds/outputs remain
ignored, and source relocation has not started.

## Next ready tasks, in order

1. **H03.R / H03.J / H03.F:** extract and port `mmr2conc` diagnostic
   conversion in separate commits, preserving native units, constants and
   mapping order. Verify native numerical round trips and captured tracer states.
2. **F11b / H04–H07 / I03:** capture remaining full-wrapper metadata and
   tendencies, then port boundary conversions, water writes, `xt_borrow` and
   wrapper composition separately. Expand core trajectories with checkpoints
   and budget reports before evolving driver forcing.
3. **C11.D009.A/F/J / D011/D012:** review scientific repairs for coagulation
   volume creation, orphan material after remapping and tiny nonvolatile loss;
   preserve current compatibility until each reference decision is made.
4. **D007/D010:** repair the reproduced absent-organic cleanup access and resolve
   invalid active zero-volume hydration before accepting those native inputs.
5. **D005:** obtain valid optical tables and repair standalone radiation before
   full-driver acceptance. Source relocation still depends on reference gates.
6. **O01–O04:** improve build portability and artifact cleanup separately.

Processes remain independently callable and tested. Complete supported native
SALSA composition now includes their shared call order and native interface.
Number/material/tracer initialization and first gas conversion are accepted.
Remaining HAM wrapper conversions, long trajectories and full-driver acceptance
remain pending.

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
ported processes against their captured boundaries. Full SALSA acceptance remains open. Hydration, nucleation and remapping
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
ends before remapping. C18/N08/I01 remained open at that milestone. D010 records a native zero-volume
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

## Remapping and correction milestone

C15–C17 and F13d are accepted within the [remapping contract](../development/remapping.md).
The 47-case redistribution matrix preserves native sequential transfers, strict
soluble thresholds, terminal-bin handling and orphan material. Maximum relative
number/volume errors are 3.20e-14/4.26e-15; all scan counts match. A targeted
compiler barrier prevents a one-ULP threshold change from reciprocal rewriting.

The 54 correction cases preserve tiny cleanup, hydration and radius/water
conversion. Number/material/gas are exact, geometry errors below 3.53e-16;
water cancellation uses a documented dry-mass scale. D007's actual bounds error,
D011 orphan material and D012 tiny nonvolatile loss are reproduced explicitly.
No production Fortran or original scientific input changed.

Three checked native runs establish capture invariance and fresh-repeat identity.
Eight captured boundaries cover both full remap/correction passes. The JAX chain
starting at native condensation output matches number/material/gas exactly;
geometry error is below 2.81e-16, and water absolute error below 2.07e-24 kg/m³.
That milestone alone did not close C18/N08 or the final density/HAM conversion and trajectory gates.

The isolated one-call CPU benchmark shows no speedup. Warm JAX/Fortran medians
are 60.45/0.222 µs for one case and 150.13/5.916 µs for 47 cases. Both complete
correction passes are outside these timings. See the [integration record](../development/evidence/remapping-2026-09-21.json)
and linked raw measurements.

| Commit | Deliverable |
|---|---|
| `db7342f` | C15–C16.R: original remapping oracle and 47 cases |
| `2872745` | C17.R: verbatim native correction block and hydration reference |
| `1266a2c` | F13d: copy-only capture of both remap/correction passes |
| `5b6589b` | F13d: reviewed native fixture and mandatory-capture checks |
| `7cae9e5` | C15–C16.J: ordered remapping and exact threshold branches |
| `7a38c0d` | C17.J: cleanup/hydration/geometry correction and failure signaling |
| `585e001` | C17: native adapter and independent/interleaved two-pass acceptance |
| `538117a` | P04d2: isolated one-call CPU benchmark and raw evidence |

## Complete native composition milestone

Final validation: `.venv/bin/python -m pytest -q` — **658 passed in 128.53 s**.

C18a–d and N08 now compose the supported native SALSA path. The complete
27-case/54-cell matrix passes eager and JIT comparisons against five unchanged
scientific Fortran modules. Maximum relative error is `2.01e-14` in mass and
`1.52e-13` in number; gas is exact. Water uses the documented dry-mass
cancellation bound, with maximum normalized error below `1.92e-15`.

Fresh checked active, recalc and active+recalc runs have identical capture/control
outputs and byte-identical process repeats. Eight tests compare every captured
process boundary and final output for these profiles plus default. Three-call
fixed-environment repeats feed back native mass/number/gas. The almost depleted
soluble-transfer bin has relative residual sensitivity near `2e-8`, reproduced
by one-ULP native input perturbations; its narrow absolute bound is documented.
No scientific defect was repaired or hidden by these acceptance checks.

Final isolated CPU medians are 100.84/17.028 µs (JAX/Fortran) for two default
cells and 2149.49/405.91 µs for the 54-cell matrix. No CPU speedup is measured.

See the [API, scope, measurements and evidence](../development/salsa-composition.md).
This closes supported microphysics composition, not HAM/driver or long-trajectory
gates. S08's general shared report and the broader state/driver contracts remain
separate tasks; the current matrix has its own fieldwise error/budget evidence.

| Commit | Deliverable |
|---|---|
| `1a59112` | C18a.R: initial cleanup oracle |
| `c55fa8d` | C18b–c.R: conversion and density oracle |
| `11a02a6` | C18b–c.J: native interface kernels and 20 tests |
| `0df0eb8` | C18a.J: shared cleanup and 6 tests |
| `5481daf` | C18d.R: complete native oracle and fixture adapter |
| `0c11ec9` | C18d.J/N08/I01/I02: composition, selected matrix, short repeats |
| `f955a96` | Copy-only checked recalculation profiles and provenance |
| `e11170c` | Actual configured boundaries and capture invariance |
| `43e9009` | P04e: matched complete-call benchmark adapter |
| `ab093be` | P04e: isolated CPU raw measurements and parity evidence |

## Number initialization and first HAM conversion milestone (2026-09-22)

Final validation: `.venv/bin/python -m pytest -q` — **740 passed in 129.33 s**.

C03 now has a JAX number initializer and unchanged-module Fortran oracle.
The 23-case matrix passes eager/JIT with maximum relative error 2.84e-14 and
zero nonfinite/negative results. Native midpoint-density quadrature and duplicate
2a/2b populations are preserved. The default 17-bin count is about 0.88536 times
the sum of modal inputs; no normalization was introduced. Driver binary32
literals are captured and retained rather than replaced with decimal64 values.

Actual driver initialization capture passes control/repeat equality and four
native acceptance tests. Its output reaches the already accepted SALSA entry
through the native unit conversions within tolerance. This validates number
initialization; `conc2mmr` material fractions and HAM tracers remain separate.

H01.R/J1 adds the original `gas2mmr` reference and pure gas-to-tracer conversion.
Fourteen focused tests cover all 3 conversion codes, untouched values, duplicate
mapping order and unsuffixed-literal precision. H01.F/J2 actual caller capture
and the complete HAM wrapper remain open. See the
[initialization guide](../development/initialization.md) and
[HAM conversion audit](../development/ham-conversion-audit.md).

| Commit | Deliverable |
|---|---|
| `9d60de3` | C03.R: unchanged Fortran initializer and case matrix |
| `5888609` | C03.J: JAX initializer and fieldwise evidence |
| `cafe944` | Actual initialized-driver capture overlay and tests |
| `fcdc98e` | Checked native fixture, adapter and capture invariance |
| `32e4dd3` | H01.R: original gas conversion reference |
| `49d77a1` | H01.J1: gas conversion leaf and precision/mapping tests |
| `fd331ae` | P04f: isolated initialization CPU benchmark and raw evidence |

Initializer CPU medians: 5.712/1.070 µs (JAX/Fortran) for default and
38.399/21.308 µs for 23 cases; no speedup is measured.

## Material initialization and native gas boundary milestone (2026-09-22)

Final validation: `.venv/bin/python -m pytest -q` — **837 passed in 137.39 s**.

H02 ports the native SALSA `conc2mmr` branch with all 69 material components,
fixed volume fractions, native mixed units and ordered tracer writes. Eighteen
reference cases have exact number output and mass/tracer relative errors below
`3.8e-16`; physical volume/fraction budgets are within `4.4e-16`. The distinct
registry molecular mass, native Avogadro constant and truncated sulfate tracer
factor are retained. Unknown material unit codes follow the native ELSE branch.

H01.F/J2 and H02 native acceptance use six checked runs: default and active
forcing, each control/capture/repeat. Scope outputs are identical with capture
on/off; all captured boundaries repeat exactly. Linked same-run number fixtures
allow eager/JIT number → material → gas tracer composition without substituting
captured intermediate outputs. Production Fortran, scientific inputs and
`AGENTS.md` are unchanged. See the [material API and evidence](../development/aerosol-initialization.md).

Warm CPU JAX/Fortran medians are 42.15/0.352 µs for one curated case and
68.49/5.630 µs for 18 cases; no speedup is observed. Compilation is separate;
all three outputs are synchronized. These timings exclude number generation,
gas conversion and the complete HAM wrapper.

| Commit | Deliverable |
|---|---|
| `577223f` | H02.R: verbatim conc2mmr oracle and 18-case matrix |
| `b819b69` | H02.J: material/tracer initializer, budgets and fieldwise evidence |
| `619a1d4` | H01.F/H02 capture: copy-only driver boundaries and strict loader |
| `9559aac` | H01.J2/H02 native acceptance: linked fixtures and composed chain |
| `7ea390a` | P04g: matched initialization benchmark and raw CPU measurements |

Next is H03 diagnostic conversion; F11b and H04–H07 remain open for full
wrapper metadata, tendencies, water pseudo-tracers and negative-tracer borrowing.
The checked full driver still fails in radiation before its later diagnostic
call, so no full-timestep or evolving-trajectory acceptance is claimed.
