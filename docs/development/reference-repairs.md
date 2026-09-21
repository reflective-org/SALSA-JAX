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
