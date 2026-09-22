# HAM concentration conversion audit

The H01 gas-conversion leaf now has a checked native reference and a pure JAX
implementation. The surrounding H02–H07 contracts below remain audit/proposed
work; no complete HAM adapter or native HAM capture acceptance is claimed. The
existing complete SALSA-call port starts inside these tracer conversions.

## Actual driver and wrapper order

1. The driver initializes HAM registries, computes moist-air density, generates
   the initial number distribution and midpoint core volumes, then calls
   `conc2mmr`. This initializes aerosol composition and tracer values.
   See [driver initialization](../../driver.F90#L622).
2. On each step, the driver overwrites sulfate gas with its prescribed forcing,
   multiplies its entire local gas array by 1e-6, then calls `gas2mmr`.
   The call therefore receives molecules/cm³ in the supported molecular-gas
   configuration. See [forcing and conversion](../../driver.F90#L749).
3. `ham_subm_interface` first calls `xt_borrow`, recomputes ambient density/RH,
   then converts **state plus incoming tendency × timestep** to concentrations.
   Aerosol mass/number are clamped nonnegative. SALSA receives gas and number
   after multiplication by 1e6; aerosol mass retains its mixed native units.
   See [negative-tracer correction](../../mo_ham_subm.F90#L230),
   [ambient state](../../mo_ham_subm.F90#L282),
   [input conversion](../../mo_ham_subm.F90#L346) and
   [SALSA call](../../mo_ham_subm.F90#L503).
4. After SALSA, the wrapper converts gas/number back to cm⁻³ conventions,
   overwrites selected tendencies with `(converted_result - pxtm1)/dt`,
   converts radii cm→m and density g/cm³→kg/m³, and directly overwrites aerosol
   water pseudo-tracers in `pxtm1`. See
   [reconversion](../../mo_ham_subm.F90#L521) and
   [water update](../../mo_ham_subm.F90#L652).
5. The driver subsequently runs other coupled processes, advances all tracers
   with accumulated tendencies, clears tendencies, and calls `mmr2conc` for
   diagnostics. This later boundary is outside first-microphysics acceptance.
   See [state advance](../../driver.F90#L1053).

## H01: implemented leaf contract

The reference extracts actual [gas2mmr](../../mo_convert_concentrations.F90#L214),
preserving literal kinds and multiplication/division order, with data-only
registry stubs and unit-code declarations read from native source.

The pure API in `salsa_jax.gas_conversion` is:
`gas_to_tracers(tracers, gas, air_density, pressure, temperature,
tracer_indices, molar_mass, unit_codes)`. It returns a new tracer array. Native dimensions are `(kbdim,klev,ktrac)` and
`(kbdim,klev,subm_ngasspec)`; the JAX contract uses matching leading case axes
`(...,ntrac)` and `(...,ngas)`. Density, pressure and temperature broadcast
over case axes. Registry arrays have final length `ngas`; tracer indices are
explicit zero-based indices. Preserve every unselected tracer. Test native
inactive padding separately, rather than treating it as another active case.

| Conversion code | Input gas concentration | Output tracer convention | Native divisor |
|---|---|---|---|
| 1, `immr2ug` | µg/m³ | kg/kg air | `1e9_dp * rho` |
| 2, `immr2molec` | molecules/cm³ | kg/kg air | `1e-3 * rho * avo / molar_mass` |
| 3, `ivmr2molec` | molecules/cm³ | volume mixing ratio | `(1e-6_dp * avo / argas) * pressure / temperature` |

Molar mass is g/mol, density kg/m³, pressure Pa and temperature K. The default
registry uses code 2 for SO2 and H2SO4; its organic gas is absent.
[Code definitions](../../mo_ham.F90#L143) and
[registered conversions](../../mo_ham_subm_species.F90#L82) are the authority.
H01 leaf tests can exercise all three defined codes, while full-wrapper
acceptance initially remains restricted to the captured code-2 configuration.

Caller preconditions are finite inputs, positive density/pressure/temperature
and molecular masses, and in-range integer tracer indices. The reference runner
checks these before execution; JIT kernels require host validation by callers.
Duplicate tracer indices preserve native gas order: the last recognized-code
write wins. Unknown conversion codes leave the destination unchanged because
SELECT CASE has no default. The JAX leaf preserves that no-op, including after a
previous gas wrote the same tracer. Negative finite gas values are not clipped.

## Arithmetic and state quirks to retain

- The code-2 factor `1e-3` is an **unsuffixed default-real literal**, then promoted
  in double-precision arithmetic. It is not identical to `1e-3_dp`. The HAM
  reverse tendency expression uses `1e3`, which is exactly representable.
  Preserve and measure this asymmetry; algebraic round-trip identity is not
  the acceptance criterion. See [gas divisor](../../mo_convert_concentrations.F90#L275)
  and [reverse factor](../../mo_ham_subm.F90#L552).
- Aerosol sulfate conversion uses `6.022e20_dp / mw_so4`, whereas molecular
  volume and gas conversion use the more precise native `avo`. Do not unify
  these constants or replace them with a modern value.
  See [aerosol factor](../../mo_convert_concentrations.F90#L173).
- `conc2mmr` overwrites aerosol mass from number/core with fixed material
  **volume fractions**: 1a SO4/OC = .2/.8; 2a SO4/OC/BC/SS/DU =
  .5/.05/.05/.1/.3; 2b SO4/OC/BC/DU = .01/.01/.01/.97.
  It then converts to tracers and mutates its number argument from m⁻³ to cm⁻³.
  Thus repeated calls on its own output are not idempotent. See
  [SALSA initialization](../../mo_convert_concentrations.F90#L112) and
  [number mutation](../../mo_convert_concentrations.F90#L207).
- `mmr2conc` returns number in m⁻³ and mixed-unit aerosol mass; it is not an
  inverse that restores all `conc2mmr` arguments. Unregistered tracer values
  are untouched. See [diagnostic conversion](../../mo_convert_concentrations.F90#L293).
- The wrapper clamps aerosol mass/number but does not generally clamp all gas
  concentrations. Its reverse gas SELECT CASE lacks code 3, despite accepting
  that code on entry. Keep code-3 leaf testing separate from a supported
  full-wrapper configuration.
- Driver gas forcing is separate from wrapper-local gas. Only sulfate is
  refreshed each iteration before the driver scales the entire gas array.
  Nonzero additional driver gases therefore need a dedicated multi-step audit.
- Density currently uses matching driver `rd1=287.04` and wrapper `rd=287.04`
  with `rho=P/[T*rd*(1+vtmpc1*q)]`; RH is clipped `q/qs`, not the separately
  calculated cloud-adjusted ambient humidity. Preserve both call boundaries.

## Missing native evidence

Existing SALSA fixtures already contain component/number/gas tracer mappings,
species molar masses/densities, gas conversion codes, full SALSA pressure/T/RH,
and native state before/after SALSA. Runtime metadata records only first-cell
air density. These do not prove the surrounding conversions.

Add copy-only captures of full `pxtm1` before/after `gas2mmr`, its exact gas
input, full-cell density/P/T, active extents, initialized gas mapping/code/mass,
and effective `avo`/`argas` plus default-real precision. Associate source,
profile, input and raw-file hashes with each fixture. For H02/H03, additionally
capture `subm_aero_idx`, `subm_aerounitconv`, `mw_so4`, initial core/number,
all overwritten aerosol masses and mutated number outputs. For H05–H07 add
water-tracer indices, incoming/outgoing tendencies, full q/qs and effective
`rd`/`vtmpc1`. Existing first-call scope arrays only record wrapper outputs.

## Commit-sized next sequence

1. **H01.R (complete):** actual gas2mmr extraction, checked reference executable and cases
   for all codes, reordered tracers, inactive padding and literal precision.
   Verify untouched sentinels and preserve invalid-code behavior explicitly.
2. **H01.F:** copy-only before/after native gas conversion capture and fixture.
   Verify capture/control/repeat equality and all-cell metadata association.
3. **H01.J1 (complete) / H01.J2:** the pure float64 leaf passes eager/JIT
   reference checks. Next accept actual native caller fixtures after H01.F;
   document the SI→native gas scaling boundary.
4. **H02.R:** capture/extract actual SALSA initialization composition and
   conversion. Test volume-fraction budgets and the number-array mutation.
5. **H02.J:** initialization adapter returning explicit mass, number and tracer
   results; connect the accepted size-distribution output without hidden units.
6. **H03.R/J:** separate diagnostic conversion reference, then JAX implementation.
   Verify round trips numerically against Fortran rather than idealized algebra.
7. **H04–H07:** wrapper boundary conversions, water writes, negative-tracer
   borrowing and composition, each retaining its own reference/JAX commits.

These are proposed units for the existing [HAM roadmap](../plans/03-coupled-processes.md).
The implemented H01 leaf is covered by 14 focused tests (eager/JIT, all three
codes, unknown-code no-op, duplicate mappings, untouched tracers/padding,
nonunity meteorology, scalar/multiple case axes and binary32 literal promotion):

```sh
.venv/bin/python -m pytest -q tests/test_gas_conversion.py
```

Physical comparisons use rtol=3e-15 and atol=0; unchanged tracer/padding checks
are exact. This is leaf acceptance only. Native gas2mmr capture, full-wrapper
acceptance, benchmarks and trajectories remain pending.
