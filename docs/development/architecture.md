# Fortran architecture and process audit

Inspected baseline: `d87dc072084336fd351d00a2ade863304c6f313d`, 2026-09-21.
Paths describe the current checkout; relocation is planned, not yet performed.

## Actual driver and core order

`driver.F90` initializes submodels, tracers, HAM, activation, size distribution
and concentrations. It runs 1000 iterations at a 1 s timestep. Each iteration:

1. Updates sulfate forcing and converts gas concentration with `gas2mmr`.
2. Calls `ham_subm_interface` (`mo_ham_subm.F90`).
3. Prepares updraft/PDF, calls radiation, then the selected activation path.
4. Applies stratiform then convective wet deposition when enabled.
5. Calls dry deposition, then sedimentation when enabled.
6. Applies and resets tracer tendencies; converts diagnostics with `mmr2conc`.

Inside native `salsa` (`mo_ham_salsa.F90`): hydration/equilibration → optional
coagulation → optional repeated equilibration → optional condensation, containing
nucleation → **two** remap/geometry-correction/equilibration passes → output
reconstruction. The HAM wrapper additionally updates water pseudo-tracers and
applies `xt_borrow`. Native kernel parity does not cover these wrapper effects.

## Process map

| Process | Source and routines | Isolation concern |
|---|---|---|
| Grid / initial spectrum | `mo_ham_salsa_init.F90:set_sizebins`; `mo_ham_salsa_sizedist.F90:size_distribution` | Mutable module geometry; private initialization helper |
| Brownian coefficient | `mo_ham_salsa_init.F90:coagc` | Six scalar inputs; constants only; first implemented leaf |
| Coagulation update | `mo_ham_salsa_dynamics.F90:coagulation` | Sequentially mutated bin/material arrays |
| Condensation | Same module, `condensation` | Gas depletion, embedded nucleation and optional organic pathways |
| Nucleation | `mo_ham_salsa_nucleation.F90` | Rate branches plus growth/survival; configuration aborts matter |
| Water / composition | `mo_ham_salsa_properties.F90:zbinmol_inv,equilibration` | Data-dependent diameter convergence, hydration switch |
| Bin transfer | `mo_ham_salsa_update.F90:distr_update` | Data-dependent loop, region boundaries, order-sensitive updates |
| Activation | `mo_ham_salsa_cloud.F90`, `mo_ham_activ.F90`, `mo_activ.F90` | PDF integration, diagnostics allocations, soluble/insoluble bins |
| Removal | `mo_ham_*dep.F90`, `mo_ham_sedimentation.F90`, HAMMOZ interface modules | Tendencies, meteorology, surface/cloud inputs |
| Optics | `mo_ham_rad.F90`, `mo_ham_rad_data.F90` | Tables, mixing and interpolation distinct from microphysics |
| Chemistry extensions | `mo_ham_vbs*.F90`, `mo_ham_soa.F90` | Compiling a module does not prove its pathway is enabled |

## State, units and indices

Native bins are 1a=1–3, 2a=4–10, 2b=11–17 in Fortran, with sulfate, OC, BC,
sea salt and dust as base species. Python maps must explicitly document zero-based
indices and water/additional species; do not rely on array position by analogy
with a different model.

| Boundary | Unit/shape contract |
|---|---|
| Native number | `pnaero(horizontal,level,17)`, particles/m³ |
| Native material | `pvols(horizontal,level,17,species)`; material m³/air m³ **inferred from equations**, to verify by capture |
| SALSA gas | molecules/m³ |
| SALSA aerosol boundary | sulfate molecules/cm³, other components µg/m³ |
| SALSA output | radii cm, particle density g/cm³, water kg/m³ air |
| HAM tracer boundary | mass kg/kg air, number particles/kg air; wrapper radii m and density kg/m³ |

Globals in `mo_ham`, `mo_ham_salsactl`, `mo_species` and `mo_time_control` carry
species/tracer maps, geometry, switches and timestep. Python will make these
explicit immutable configuration/state inputs; kernels return changed state and
diagnostics. Host file I/O stays outside transformed JAX computations.

## Observed limitations and discrepancies

- **D001–D004 repaired:** absent organic-gas condensation, undefined organic
  fraction, activation velocity shape/units, and aerosol-water storage have
  separate regressions. A fresh checked run now fails inside standalone
  radiation (`ham_rad_refrac_volume`, D005). The ordinary trajectory is still
  unvalidated; see the [repair record](reference-repairs.md).
- Runtime capture confirms 64-bit `dp`, 109 tracers, 14 species, 17 aerosol
  classes, `nsol=17`, two gases, and an absent-organic-gas index of zero. These
  are initialized values for the default case, not inferred parameter defaults.
- Default `nsnucl=0`; condensation/coagulation and 2b hydration are enabled.
  Setup rejects `nsnucl>4`, and mode 4 aborts pending NH3 input. Unreachable
  parameterizations need new supported Fortran cases before being ported.
  Nucleation has its own unguarded organic-gas read, still to be reproduced and
  repaired before enabling those modes.
- `set_coagc` computes a local particle mass array but does not populate a global
  coefficient table; the routine name is not evidence of its behavior.
- The driver computes `new_pelvoc` without consuming it; sulfate forcing peaks at
  iteration 2000 while the default run ends at 1000. Preserve these facts before
  considering separate forcing corrections.
- Some dry-deposition driver inputs are explicitly provisional. The source also
  emits legacy interface/rank and character-length warnings during compilation.
  These require targeted reproductions, not blanket warning suppression.
- Existing formatted output has limited precision. Repeat-identical text is a
  useful smoke check, not enough for tight process parity or budget acceptance.

The [master plan](../plans/ULTRAPLAN.md) turns these issues into dependencies and
small verification tasks. No source relocation or full-model parity claim is
accepted until the checked-reference findings are resolved or explicitly scoped.
