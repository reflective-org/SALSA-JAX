# P4 — HAM adapters and coupled box-model processes

The `.R`/`.J` convention from [P2/P3](02-state-and-core-processes.md) applies:
each named pair is two commits, first a real Fortran contract/capture and then
the matching JAX unit with tests and documentation. Prerequisites name accepted
JAX units. Reference tasks require F14–F16. Preserve the driver's execution order.

## P4.1 — HAM state boundary

| Unit (two commits) | Depends on | Scope | Acceptance |
|---|---|---|---|
| H01.R / H01.J | S03,S07 | `gas2mmr` gas conversion | Molecules/m³ vs molecules/cm³ and kg/kg air explicitly tested |
| H02.R / H02.J | H01 | `conc2mmr` initialization/conversion | Species composition initialization matches; not treated as a simple inverse |
| H03.R / H03.J | H02 | `mmr2conc` diagnostic conversion | Every tracer/species/bin, water and density convention |
| H04.R / H04.J | I01,H03 | SALSA legacy-unit boundary adapter | Sulfate molecules/cm³, other mass µg/m³, radii cm and density g/cm³ |
| H05.R / H05.J | H04 | HAM water pseudo-tracer update | Dry/wet and empty-bin tendency budget |
| H06.R / H06.J | H05 | `xt_borrow` negative-tracer correction | Negative/zero/positive cases; match redistribution and resulting budget |
| H07.R / H07.J | H06 | `ham_subm_interface` composition | Full state/tendencies match, including conditional process switches |

## P4.2 — Cloud activation

| Unit (two commits) | Depends on | Scope | Acceptance |
|---|---|---|---|
| A01.R / A01.J | S05 | `lnS_crit_insol_over_A` | Insoluble-particle thresholds and source domain |
| A02.R / A02.J | H04 | `ham_activ_koehler_ab` SALSA branch | Composition and temperature inputs; soluble/insoluble distinctions |
| A03.R / A03.J | A01,A02 | `cloud_activation` critical-size calculation | Capture thresholds and active-bin decisions |
| A04.R / A04.J | A03 | `cloud_activation` activated fractions/counts | Fractions in reference range; counts bounded by available particles |
| A05.R / A05.J | S05 | `activ_updraft`/PDF preparation | Zero/positive updraft, PDF normalization and reference conventions |
| A06.R / A06.J | A04,A05 | `salsa_abdul_razzak_ghan` PDF integration | Match each PDF point then aggregate diagnostics |
| A07.R / A07.J | A06,H07 | Driver activation adapter | Allocatable diagnostics reset correctly; two fresh calls agree |

## P4.3 — Removal processes

| Unit (two commits) | Depends on | Scope | Acceptance |
|---|---|---|---|
| D01.R / D01.J | H04 | `ham_prep_sedi` settling preparation | Size/density dependence; compare per-bin velocities |
| D02.R / D02.J | D01 | `ham_sedimentation` tendency | Nonnegative reference state; removed material accounted for |
| D03.R / D03.J | D02,H07 | `sedi_interface` adapter | Layer/column conventions and disabled-process behavior |
| D04.R / D04.J | H04 | Dry-deposition aerodynamic/resistance helpers | Supported surface cases; provisional inputs documented |
| D05.R / D05.J | D04 | Dry-deposition particle velocity calculation | Diffusion/impaction/settling contributions separately compared |
| D06.R / D06.J | D05,H07 | `drydep_interface` flux/tendency composition | Units and surface sink budgets; driver call policy preserved |
| W01.R / W01.J | A07,H07 | Wet-deposition nucleation scavenging | In-cloud activation/removal and cloud-phase branches |
| W02.R / W02.J | W01 | Wet-deposition impaction scavenging | Size-dependent collection and empty/cloud-free cases |
| W03.R / W03.J | W02 | Below-cloud rain scavenging | Zero/raining and evaporation cases; tracer sink budget |
| W04.R / W04.J | W02 | Below-cloud snow scavenging | Ice/snow thresholds and sublimation cases |
| W05.R / W05.J | W03,W04 | Stratiform wet-deposition composition | Per-stage state/tendency and accumulated removal |
| W06.R / W06.J | W05 | Convective wet-deposition composition | Convective flags and ordering after stratiform call |

Audit large legacy removal routines into exact helper boundaries before their
`.R` task. If a row spans independent formula families, assign child IDs (e.g.,
D04a resistance, D04b aerodynamic terms) with separate captures and commits.
No entire 2,000-line legacy module should be translated in one commit.

## P4.4 — Radiation, driver, and optional chemistry

| Unit (two commits) | Depends on | Scope | Acceptance |
|---|---|---|---|
| R01.R / R01.J | H04 | Optical table loading and axis contract | Table hashes, axes, dimensions and endpoint conventions |
| R02.R / R02.J | R01 | Refractive-index mixing for SALSA | Component mixtures and dry/wet cases |
| R03.R / R03.J | R02 | Shortwave lookup/interpolation | Interior/edge cases and extrapolation policy |
| R04.R / R04.J | R02 | Longwave lookup/interpolation | Same axes/units checks, no silent table substitution |
| R05.R / R05.J | R03,R04 | `ham_rad` optical diagnostics | Optical depth, single-scattering albedo and asymmetry |
| B01.R / B01.J | H07 | Box forcing schedule | Original iteration-based Gaussian and unused ELVOC behavior documented |
| B02.R / B02.J | B01,A07,D03,D06,W06,R05 | One driver step orchestration | Every boundary in original order; apply/reset tendencies exactly once |
| B03.R / B03.J | B02 | Multi-step driver/checkpoint output | Short and 1000-step trajectories, fieldwise errors and budgets |

Optional extensions are separate projects after B03, not implied active physics:

| ID | Depends on | One-commit deliverable | Verification |
|---|---|---|---|
| E01 | B03.J | Inventory actually reachable VBS pathways | Demonstrate effective switches and baseline capture |
| E02 | E01 | Define VBS volatility/species mapping | Exact ordering, units and mass totals |
| E03.R / E03.J | E02 | One VBS partitioning helper | Reference equilibrium/kinetic cases and gas+particle budget |
| E04 | E03.J | Audit remaining VBS helpers into named commit pairs | Every helper has an oracle case before implementation |
| E05 | Every helper `.J` task enumerated by E04 | Compose accepted VBS helpers with SALSA | Explicitly enabled trajectory parity |
| E06 | B03.J | Inventory aqueous chemistry dependencies and missing inputs | Supported build/case or explicit missing-dependency report |
| E07 | E06 | Define aqueous helper-specific `.R`/`.J` backlog | No unsupported implementation promise without runnable oracle |

OpenIFS coupling is documented at the interface level initially. A host-model
integration test requires the actual host checkout, configuration and inputs;
box-model parity alone must never be described as OpenIFS integration validation.
