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
