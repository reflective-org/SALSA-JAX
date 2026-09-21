# Isolated nucleation reference

The D006 repair gives an absent nonvolatile organic gas concentration of zero.
Previously `nucleation` indexed `pc_gas(:,:,0)` before selecting even the
sulfate-only schemes. A positive registered index still reads the original gas
entry. No nucleation formula, constant, switch or default is changed.

## Actual Fortran boundary

`mo_ham_salsa_nucleation.F90:nucleation` receives gas concentrations in
molecules/m³, background particle numbers in particles/m³, wet diameters in m,
temperature in K, relative humidity as a fraction, pressure in Pa and timestep
in seconds. Arrays use horizontal/vertical/bin order and one-based indices.

The routine returns two molecular source rates in `pj3n3`, plus sulfate/organic
fractions. Despite legacy comments describing concentration changes, the output
is `zj3*n3*fraction`: molecules/(m³ s). It does not itself update gas or background
particle number. The enclosing condensation routine uses these rates to deplete
gas, add material, and insert particles in the smallest bin. Test and benchmark
these boundaries separately, then verify their composition.

## Regression and compatibility evidence

```sh
.venv/bin/python -m pytest tests/test_nucleation_regression.py -q
```

The 29 checks compile the complete original/current nucleation module with real
precision and physical constants. Fixture modules supply explicit initialized
state; no nucleation or condensation formulas are replaced.

- Original sulfate schemes 1, 2 and 3 each reproduce the index-zero bounds error.
- With the correction, each scheme and growth option `nj3=1,2,3` exactly matches
  the original routine supplied an explicitly registered zero organic gas.
- Registered positive-organic cases remain exactly unchanged. Organic vapor can
  affect cluster growth even when the nucleation composition is sulfate-only;
  the repair preserves that original behavior.
- A two-column/two-level case covers different temperatures and boundary-layer
  indices. Zero sulfate produces no source; an empty background preserves the
  activation source where scavenging vanishes.
- Coupled tests for `nj3=1` close the sulfate gas/material budget and bound new
  particle number by the consumed molecular volume. Mode 4 still reports missing
  ammonia rather than proceeding with fabricated input.

[Recorded evidence](evidence/nucleation-repair-2026-09-21.json) includes source
hashes, compiler flags, fixture inputs and rates. All nine ordinary default-run
diagnostics remain identical after this repair; that case has `nsnucl=0`.

## Process oracle and scope limits

These are checked kernel fixtures, not full-driver nucleation trajectories.
Original/current agreement for `nj3=2,3` does not establish validity throughout
the parameter domain. Ordinary initialization rejects modes above four; no
organic or ternary nucleation capability is enabled here.

`tools/nucleation_process_oracle.py` now supplies the independent port oracle.
It compiles the full module after preserving a byte-identical original copy,
records critical clusters with a small diagnostic overlay, and calls the actual
extracted condensation routine for coupled checks. Its 110 cases cover active
binary rates and all scheme/growth combinations, absent/registered organics,
cutoffs, vapor limitation, empty backgrounds and heterogeneous properties.
Two duplicated vertical cells exercise both sides of the native PBL boundary;
results select the requested cell and timing includes both cells.

```sh
.venv/bin/python -m pytest -q tests/test_nucleation_process_oracle.py
```

Thirteen checks establish compiled constants, untouched source provenance,
strict checked/optimized bit equality, nontrivial binary activity, isolated
state preservation and coupled gas/material budgets. This is an initialized
kernel oracle, not an activated full-driver capture. The [core roadmap](../plans/02-state-and-core-processes.md)
tracks the remaining native capture and coupled SALSA acceptance gates.
