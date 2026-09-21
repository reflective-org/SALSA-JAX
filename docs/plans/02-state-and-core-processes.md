# P2/P3 — Explicit state and process-by-process core port

## P2.1 — Shared contracts

| ID | Depends on | Commit-sized deliverable | Verification |
|---|---|---|---|
| S01 | F01,F04 | Define exact legacy physical constants and precision policy | Compare values/kinds to compiled Fortran; CPU float64 asserted |
| S02 | F11 | Define bin geometry and one-based/zero-based map | Exact 1a/2a/2b boundaries and 17 active bins |
| S03 | F11 | Define species and tracer index maps | Round-trip named maps; reject duplicate/missing required indices |
| S04 | S02,S03,F12 | Define immutable number/volume/gas state with units | Shapes/dtypes match captured state; pytree round trip |
| S05 | S04 | Define meteorology and timestep inputs | Pressure/temperature/RH and density conventions validated |
| S06 | S05 | Separate static process options from runtime arrays | Supported combinations explicit; JIT does not capture mutable globals |
| S07 | S06,F14 | Add one captured-state loader and explicit unit adapters | Fortran → Python → Fortran reproduces every field |
| S08 | S07 | Add common per-process error/budget report | Introduced unit/index error fails; no blanket clipping hides mismatch |

## P2.2 — Leaf and process contract pattern

Every unit below has **two separately numbered commits**:

- `<ID>.R`: add a narrow original-Fortran wrapper/capture, case inputs, shape/unit
  contract and provenance. Verify repeated fresh-process results and nontrivial
  branch coverage. Compile original code or mechanically extract a verbatim leaf;
  never use a translated formula as the oracle.
- `<ID>.J`: port only that unit, add eager/JIT comparisons and the listed physical
  or boundary checks, and document measured errors. Depends on its own `.R` and
  the prerequisite `.J` units shown below. No missing-oracle skips at acceptance.

For C01, the independent scalar oracle can precede full-model F11–F16. Later
processes use scoped fixtures, provenance and comparisons; passing those does
not close the complete parent trajectory gates. Coagulation C07–C11 now accepts
captured dry geometry and hydrated diameters explicitly, allowing its standalone
port before C02/C06. Condensation C12–C14 similarly accepts captured hydrated
diameters, with nucleation disabled. Those dependencies remain required for
coupled execution.
Documentation of the unit's equations, units, branch limits and tolerances belongs
in its `.J` commit. Split a row into additional named child kernels if its audited
implementation cannot be reviewed as one coherent change.

## P3.1 — Grid, water and coagulation

| Unit (two commits) | Prerequisite JAX units | Exact scope | Additional acceptance |
|---|---|---|---|
| C01.R / C01.J | S01 | `mo_ham_salsa_init:coagc`, six-scalar coefficient | Symmetry, positive/finite domain, heterogeneous diameter/mass/T/p cases |
| C02.R / C02.J | S02 | `set_sizebins` geometry | Limits/midpoints and region transitions match exactly or calibrated roundoff |
| C03.R / C03.J | C02 | `size_distribution` initialization | Seven lognormal modes, empty/single-mode cases, binwise match |
| C04.R / C04.J | S03,S04 | `zbinmol_inv` composition helper | Each species separately and mixed bins |
| C05.R / C05.J | C04,S05 | `equilibration` soluble regimes | Dry/wet and near-convergence inputs; original iteration criterion retained |
| C06.R / C06.J | C05 | `equilibration` 2b hydration switch | Both `lhydr2b` branches and inactive bins |
| C07.R / C07.J | C01,C06 | Coagulation coefficient matrix construction | Every sector's coefficients match; symmetry only where physics expects it |
| C08.R / C08.J | C07 | Coagulation transfer/update for regime 1a | Match intermediate sequential updates, number loss and material budget |
| C09.R / C09.J | C08 | Coagulation soluble 2a transfers | Empty/dense/unequal-bin cases; preserve update order |
| C10.R / C10.J | C09 | Coagulation insoluble/cross-regime transfers | Every supported sector and species transfer budget |
| C11.R / C11.J | C10 | Complete `coagulation` call | Process off/on; no timestep or clipping discrepancy |

## P3.2 — Condensation and bin remapping

| Unit (two commits) | Prerequisite JAX units | Exact scope | Additional acceptance |
|---|---|---|---|
| C12.R / C12.J | C06,S05 | Condensation transport/uptake coefficients | Matched sink coefficients over wet radius and meteorology cases |
| C13.R / C13.J | C12 | Sulfuric-acid condensation with `nsnucl=0` | Gas depletion and sulfate material budget including limiting branches |
| C14.R / C14.J | C13 | Existing organic condensation path | Explicit registered-organic kernel fixtures; `locgas` alone does not establish gas registration |
| C15.R / C15.J | C02,S04 | `distr_update` transfer decisions | Just below/on/above boundaries and nearly empty bins |
| C16.R / C16.J | C15 | `distr_update` material/number redistribution | Sequential transfers and loop termination; conserve reference budgets |
| C17.R / C17.J | C16,C06 | Post-remap radius/volume correction | Exact zero-bin policy and both remap passes |
| C18.R / C18.J | C11,C14,C17 | Native `salsa` composition without nucleation | Every process boundary plus final state; `recalc` off/on |

## P3.3 — Nucleation, retaining condensation coupling

| Unit (two commits) | Prerequisite JAX units | Exact scope | Additional acceptance |
|---|---|---|---|
| N01.R / N01.J | S05 | `binnucl` parameterization | Source validity limits and threshold cases |
| N02.R / N02.J | S05 | `actnucl` parameterization | Rate scaling and zero/low precursor behavior |
| N03.R / N03.J | S05 | `kinnucl` parameterization | Precursor scaling and limiting concentrations |
| N04.R / N04.J | N01,N02,N03,C12 | `nj3=1` growth/survival branch | Match rate, vapor sink and inserted number/volume |
| N05.R / N05.J | N04 | `nj3=2` growth/survival branch | Boundary cases and branch-specific intermediate rates |
| N06.R / N06.J | N04 | `nj3=3` growth/survival branch | Branch helper including `zNnuc_tayl` where reached; split helper if needed |
| N07.R / N07.J | N04,N05,N06 | Supported `nucleation` dispatcher | `nsnucl=0,1,2,3` effective options; preserve unsupported errors |
| N08a.R / N08a.J | N07,C14 | Isolated condensation+nucleation process coupling with supplied wet diameters | Actual native call order, vapor limitation, insertion and species budgets; implemented |
| N08b.R / N08b.J | N08a,F13c.N3 | Accept explicitly activated native process capture | Isolated source and coupled output match actual captured boundary |
| N08.R / N08.J | N08b,C18 | Complete native SALSA nucleation compatibility composition | Same call order and final state with hydration and remapping |

`nsnucl=4` currently aborts for missing NH3; values above 4 are rejected by setup.
`ternucl`, `orgnucl`, `sumnucl`, `hetnucl`, `SAnucl`, and `SAORGnucl` are **not**
assumed supported just because routines exist. Enabling each requires a separate
Fortran case and a new `.R`/`.J` pair after scientific configuration is established.

## P3.4 — Core integration gate

| ID | Depends on | Commit-sized deliverable | Verification |
|---|---|---|---|
| I01 | N08.J,S08 | Add one-step native SALSA acceptance matrix | Every enabled/off branch captured and compared |
| I02 | I01 | Add short repeated-call trajectories | Gas/number/material trajectories and accumulated errors reported |
| I03 | I02 | Add longer core trajectories and budget report | Drift measured at checkpoints; baseline defects remain explicit |
| I04 | I03 | Document supported core API and minimal example | Example runs from installed package with no sibling repository imports |

## Coagulation milestone and discovered follow-up

C07–C11 are implemented together as a standalone 17-bin/five-material process,
with separate commits for capture, original-Fortran oracle and JAX translation.
The small transfer branches share one sequential loop; splitting them into
separate runtime abstractions would obscure the required update order. Supplied
geometry/wet diameters satisfy this isolated API; grid/hydration JAX ports and
all coupled acceptance gates remain open.

| ID | Depends on | Commit-sized deliverable | Verification |
|---|---|---|---|
| C11.D009.R | C11.R | Preserve minimal cross-regime volume-creation reproducer | Original two-bin result and analytical transfer mismatch agree; implemented |
| C11.D009.A | C11.D009.R | Compare conservative update choices and document scientific decision | Positive/empty/stiff cases, number/material budgets and effect on dilute baseline |
| C11.D009.F | C11.D009.A | Implement the selected Fortran reference repair separately | Reproducer conserves; affected process/trajectory outputs reviewed |
| C11.D009.J | C11.D009.F | Update JAX to the repaired reference with explicit compatibility decision | Eager/JIT parity, budgets and repeat-call acceptance |

## Condensation milestone

C12–C14 are implemented as `condensation_rates` and `condensation_step`, with
`nsnucl=0`, `nsoa=0` and supplied hydrated diameters. Separate commits retain
actual boundary capture, the repaired Fortran oracle, the JAX port and the
benchmark. The 29-case matrix covers registered/absent organic gas and actual
runtime species constants. F13c is complete only for the condensation boundary;
nested active nucleation capture is subsequently verified for scheme 2/growth 1. N01–N07 and N08a are now
implemented as described below. A two-process check connects
coagulation to condensation with captured diameter inputs; it does not close
the hydration, remapping or complete-native integration gates.

## Nucleation milestone and active native capture

N01–N03 rate leaves and N04–N07 growth/source dispatcher are implemented.
N08a couples the accepted sources to the shared native condensation update,
without duplicating uptake physics. Its 110-case oracle, species budgets,
static-option JIT groups and ten successive calls pass. P04c records separate
isolated/coupled timings; no speedup is claimed. These process APIs accept
supplied wet diameters and do not close C18, N08, I01 or full-driver gates.

The following active-native tasks are now complete for scheme 2/growth 1,
with explicit first-call sulfate forcing. See the [capture guide](../development/nucleation-native-capture.md)
for native invariance and actual-boundary parity evidence:

| ID | Depends on | Commit-sized deliverable | Verification |
|---|---|---|---|
| F13c.N1 | N07.R,F13c | Add copy-only configured activation of one supported native scheme and growth option | Original source/namelists untouched; initialized switch captured; strict run reaches scoped completion |
| F13c.N2 | F13c.N1 | Capture nested nucleation input/output plus enclosing condensation boundary | Full-precision units/maps; no translated reference formulas; capture off/on arrays identical |
| F13c.N3 | F13c.N2 | Export one reviewed active fixture with source/configuration provenance | Fresh-process exact repeat, nonzero source/insertion and gas/material budgets |
| N08b.R | F13c.N3,N08a.R | Replay that fixture through isolated and coupled native oracles | Every relevant boundary matches captured source and state |
| N08b.J | N08b.R,N08a.J | Accept JAX source/coupled outputs against active native fixture | Eager/JIT parity, physical budgets and configuration documented |

Grid generation C02 and hydration C04–C06 remain prerequisites for complete
composition; remapping C15–C17 follows. Inherited nucleation arithmetic quirks
are recorded in the process guide and require separate scientific decisions
before any repair. They are not silently corrected in the compatibility port.

| Commit | Completed active-native task |
|---|---|
| `a8809ea` | F13c.N1: explicit profile and copied-driver configuration |
| `f571960` | F13c.N2: nested/parent capture and harness integration |
| `736c048` | F13c.N3: reviewed fixture pair and mandatory-capture checks |
| `d60258f` | N08b.R/J: native replay, adapter and eager/JIT acceptance |

Next implement C02.R/J (native grid generation and JAX geometry), then
C04.R/J (composition helper), C05.R/J (soluble equilibration) and C06.R/J
(insoluble hydration switch), each with the existing separate reference and
port acceptance. F13a should capture the actual hydration boundary before
claiming its native integration. Parent N08/C18/I01 remain open.
