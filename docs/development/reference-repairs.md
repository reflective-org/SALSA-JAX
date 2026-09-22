# Fortran reference repairs

Keep each defect, reproducer and scientific impact separate. The tag
`fortran-baseline-d87dc07` preserves the unchanged upstream version.

## D001 — condensation reads an absent organic gas

`map_ham_subm_species` no longer registers the optional nonvolatile organic gas;
its registration call is commented out. Nevertheless, the non-VBS condensation
branch reads `pc_gas(:,:,isubm_ocnv)` without checking whether an index exists.
The default gas list contains two entries, and the observed organic index is zero.
`locgas` is an emissions option, not proof that this gas has been registered.

Task F09 adds a minimal failing reference case in
`tests/test_condensation_regression.py`. The test extracts the actual condensation
routine from the baseline tag, compiles it with original precision/physical/math
constants, and checks that the missing-gas case stops on an index-zero error.
The fixture modules supply explicit data; they do not replace condensation
formulas. Their nucleation stub aborts if called, keeping this regression strictly
about condensation. The test needs GNU Fortran but no NetCDF data or full model.

The full reproducer remains:

```sh
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-checked.toml --output build/checked-before-repair --run
```

A positive registered-organic case is a deliberately constructed kernel fixture;
ordinary model initialization does not currently register that gas. It must not
be described as validation of a supported full-driver organic configuration.

### F10 correction and impact

The optional index now has an explicit zero sentinel. Non-VBS organic
condensation executes only inside `ELSE IF (isubm_ocnv > 0)`, so no array
reference depends on Fortran short-circuit behavior. Species registration and
tracer dimensions are unchanged.

Four isolated tests pass: the historical failing case, missing/zero organic
gas with sulfate material-budget checks, and an empty-particle case. All nine
default compatibility diagnostic hashes match the pre-fix run exactly. The
checked full run proceeds beyond D001 but exposes a further segmentation fault;
reference acceptance remains pending. See [recorded evidence](evidence/d001-repair.json).

## D002 — undefined organic fraction when nucleation is disabled

With an explicitly registered, positive organic gas and `nsnucl=0`, condensation
read `zxocnv` without assigning it. The sulfuric-acid fraction was initialized,
but its organic counterpart was only returned by the skipped nucleation call.
The positive-organic regression failed under signaling-NaN initialization and
floating-point traps before correction; assigning `zxocnv=0._dp` alongside
`zxsa` makes the no-nucleation contract explicit.

Seven condensation tests now pass, including positive-organic material-budget
closure, unchanged particle number, the concentration threshold, and zero sink.
The organic kernel fixture uses no sulfate vapor, so gas/species indexing errors
cannot be hidden by a simultaneous sulfate change. This does not enable organic
gas registration in the full driver or validate VBS.

## D003 — activation input shape and units

The driver initialized the last plane of a rank-three velocity array, whereas
the activation routine consumed a rank-two array. It also supplied m/s where
Pa/s was required. The corrected rank-two input converts the existing NetCDF
`W` sample with `-grav*zrhoa*W`. Eight isolated tests cover the original failure,
input metadata, signed/zero velocities, heterogeneous densities and PDF values.
See [updraft evidence and scientific limitations](updraft-input-repair.md).

## D004 — aerosol-water output storage is too small

`driver.F90` allocated `zww` with `nmod=7`, the size of the initial modal
distribution. `ham_subm_interface` writes `nclass=17` aerosol classes for SALSA.
The explicit-shape argument concealed the mismatch from runtime bounds checks;
ten excess writes could corrupt neighboring stack data and array descriptors.
An isolated debug run stopped crashing in activation when this storage was
corrected, then exposed the independent radiation failure below.

The driver now allocates `zww(kbdim,klev,nclass)` after model initialization.
The initial-distribution `nmod` remains seven. Three storage-contract tests
compile the actual driver declaration/allocation: the baseline fails for 17
classes, and the corrected allocation accommodates both seven and 17. These
tests validate storage, not complete M7 scientific behavior.

## D005 — standalone radiation reference remains unresolved

After the storage correction, a fresh isolated checked run reaches `ham_rad` and
traps in `ham_rad_refrac_volume`. In the current standalone SALSA build,
`ham_initialize` calls `ham_rad_initialize` only under `HAMMOZ`, while the
driver calls `ham_rad` unconditionally. The supplied optical-property files
are only 64 and 67 bytes; `ncdump` rejects both. The
[radiation audit](radiation-reference-audit.md) also identifies path construction
and initialization-order defects.
Do not invent optical values, disable traps, or describe this as an accepted
full-model reference. The fresh `build/reference-repairs-checked` run reproduced
the trap; symbolized frames identify `ham_rad_refrac_volume` → `ham_rad_refrac`
→ `ham_rad` → `driver.F90:808`. Audit initial diagnostic values separately from
microphysics. See [current comparison evidence](evidence/reference-repairs-2026-09-21.json).

The separate nucleation gas-index defect is repaired below. Full-driver
`nsnucl>0` trajectories still require their own reference acceptance.

## D006 — nucleation reads an absent organic gas

The independent nucleation routine now initializes optional organic vapor to
zero and accesses its gas entry only for a positive registered index. The
original index-zero failure is retained for all three sulfate nucleation schemes.
Twenty-nine checks establish original/current compatibility, zero-input behavior,
and sulfate budgets when nested in condensation; see the
[nucleation reference contract and evidence](nucleation-reference.md).
This does not enable unsupported nucleation modes or validate full trajectories.

## D007 — tiny-particle cleanup still reads absent organic gas

Six evaporation/cleanup branches in `salsa` access organic gas index zero
without a guard. The checked default fixture avoids these branches. This is
an outstanding repair task; D001 and D006 do not cover it. The extracted C17
correction oracle now reproduces the original checked bounds failure. JAX
signals invalid affected cells without inventing an organic sink; see the
[remapping/correction contract](remapping.md).

## D008 — initialization overwrites registered component maps

The first native capture exposed inconsistent bin/species identities:
`conc2mmr` overwrote species-major component maps with region-major indices
after HAM registered its 69 components and tracers. For example, OC in bin 1
was assigned component 4, already registered as sulfate in bin 4. Subsequent
unit conversion used the original metadata, so this affected scientific state.

Commit `0f843dd` removes only the redundant map assignments and counter,
retaining all composition recipes. Two checked Fortran regressions reproduce
the original overwrite and verify current maps, all 69 component identities
and the intended per-bin material volumes. The repaired native capture passes
strict mapping checks and material budgets; its six returned HAM arrays match
exactly with capture disabled. Fresh-process native files repeat byte for byte.

This changes the initialized scientific baseline. Previous full-driver
diagnostic comparisons do not establish acceptance of the repaired trajectory.
See the [state contract](salsa-state-contract.md) and
[capture evidence](evidence/native-state-2026-09-21.json).

## D009 — cross-regime coagulation creates volume in stiff cases

The complete-process oracle exposes a native numerical defect: regime 2a gains
use the old 2b volumes, then regime 2b losses use already-updated 2a numbers.
The mismatched transfers fail to conserve material volume. A minimal pair of
corresponding 2a/2b bins containing only sulfate gains 240.2863% total volume at
1e12 particles/m³ per bin and a 3600 s timestep. This is not caused by sea-salt
relabeling; the captured dilute default coagulation call closes to roundoff.

The JAX port reproduces the original Fortran behavior; no reference repair is
included in this milestone. The regression explicitly demonstrates the budget
failure, and parity tests must not be described as unrestricted scientific
acceptance. See the [coagulation contract](coagulation-process.md) for the
reproducer, measured errors, benchmarks and supported scope. A conservative
scheme requires a separate reviewed scientific change.

## D010 — active soluble zero-volume hydration fails

The original `equilibration` enters its soluble-bin iteration whenever number
is strictly above `nlim`. With zero volume in every material, both dry core and
computed wet diameter are zero; the subsequent Kelvin expression divides by
that diameter. `tests/test_hydration_oracle.py` reproduces the floating-point
failure under the checked original-Fortran oracle. Positive-volume pure BC/dust
are valid and remain dry; absence of sulfate/salt alone is not this failure.

The compatibility port does not modify Fortran or invent particle volume.
JAX reports NaN wet diameter for this invalid domain, avoiding an endless loop
that literal unchecked arithmetic could produce. `tests/test_hydration.py`
checks that signaling while preserving other bins. A scientific repair or
stronger caller invariant needs a separate decision before accepting such
inputs. See the [grid/hydration contract](grid-and-hydration.md).

## D011 — remapping leaves orphan material and oversized receivers

The original `distr_update` caps the transferred number fraction at one but
computes material transfer from its extrapolated receiving volume. A bin whose
mean volume is ten times its upper bound transfers all particle number while
retaining about 7.1394% of its material at zero number. Material/number totals
remain conserved across bins; the resulting bin state is inconsistent.

The descending scan checks only the donor after a transfer. It can leave an
already-visited receiver above its upper bound without requesting another
scan. SALSA calls redistribution twice with cleanup/hydration between calls;
this is not a general guarantee of bin-bound convergence. Terminal bins 10/17
also do not undergo their own size correction. These behaviors are reproduced
by the checked original oracle and preserved in JAX. No extra sweeps, clipping
or orphan removal are introduced. Enabling soluble transfer for an active
zero-volume terminal bin additionally divides by zero in the native fraction
calculation; JAX signals NaN for that invalid cell.

See [remapping acceptance](remapping.md) and
`tests/test_remapping_oracle.py`. A conservative, internally consistent remap
requires a separate scientific decision and reference change.

## D012 — tiny-particle cleanup discards nonvolatile material

Each native post-remap cleanup removes all five material volumes when its
particle diameter is below 1e-10 m, but returns only sulfate and organic
material to gas. A checked case with registered organic gas and tiny BC
particles therefore loses BC without a compensating reservoir. This is
independent of the absent-organic indexing failure D007: registering the gas
avoids the bounds error but does not conserve nonvolatile material.

The cleanup diameter in bins 1–3 counts only sulfate/OC, while shrinking bin 11
can move all five materials into bin 3. That pathway can expose the same loss.
The C17 port retains this behavior explicitly; matching the original does not
establish unrestricted material-budget validity. `tests/test_remapping_correction.py`
checks both the loss and D007's original bounds failure. Invalid D007 JAX cells
return `valid=False` and NaN physical outputs; this is failure signaling, not
an absent-gas repair. See the [correction contract](remapping.md).
