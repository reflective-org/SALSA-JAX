# P0/P1 — Reference foundation and repository organization

Each row is a commit-sized task. Dependencies refer to task IDs; status is tracked
in [STATUS.md](STATUS.md). Keep mechanical changes separate from scientific fixes.

## P0.1 — Inventory and provenance

| ID | Depends on | Deliverable / suggested commit | Verification |
|---|---|---|---|
| F01 | — | Document source/process map and supported default configuration | Trace actual call sites, switches and units; flag inferred units |
| F02 | — | Document local JAX precedents with source revisions | Identify dirty/untracked examples; no sibling runtime dependency |
| F03 | F01 | Add phased plan and commit ledger | Every leaf has dependencies, scope and executable/review gate |
| F04 | — | Ignore generated outputs and establish Python package/environment | Existing tracked inputs remain visible; fresh venv imports package |
| F05 | F01 | Add isolated Fortran build/run and provenance manifest | No original inputs modified; failed builds cannot reuse old binary |
| F06 | F05 | Repair missing VBS module dependency edges | Clean serial and parallel builds agree; no `.F90` byte changes |
| F07 | F05,F06 | Capture repeat compatibility runs and diagnostic summary | Two fresh processes agree; finite fields and expected 1001×17 records |
| F08 | F07 | Add checked build profile and discrepancy report | Bounds/uninitialized checks executed; every failure recorded |

## P0.2 — Make the reference scientifically inspectable

| ID | Depends on | Deliverable / suggested commit | Verification |
|---|---|---|---|
| F09 | F08 | Reproduce first checked-build defect with a narrow case | Test fails at same source site before any repair |
| F10 | F09 | Repair that defect or document justified compatibility isolation | Reproducer passes; compatibility impact reported, never assumed |
| F11 | F07,F10 | Emit effective switches, timestep and kind/storage size | Values read from running Fortran, not inferred from config text |
| F12 | F11 | Add full-precision native SALSA entry/exit capture overlay | Instrumented/uninstrumented diagnostics unchanged |
| F13a | F12 | Capture equilibration boundaries | Number/material/water/radius fields and hydration options recorded |
| F13b | F13a | Capture coagulation boundaries | Before/after bin state and coefficients, including mutated globals |
| F13c | F13b | Capture condensation and nested nucleation boundaries | Gas depletion, rates and aerosol insertion distinguished |
| F13d | F13c | Capture remap/geometry boundaries | Both passes numbered, transfer state and radii recorded |
| F14 | F13d | Add fixture manifest and explicit regeneration command | Detect changed/missing/unlisted arrays and changed provenance |
| F15 | F14 | Add numerical comparison report with physical-unit tolerances | Deliberate perturbation fails; near-zero/NaN/wrong-shape cases fail |
| F16 | F15 | Add repeat/reset and process-switch isolation cases | Fresh-process results match; distinct configurations cannot collapse silently |

Repeat F09/F10 as numbered child tasks for additional independent defects; do not
turn a single repair task into a repository-wide warning cleanup.

### Discovered child tasks

| ID | Depends on | Commit-sized scope | Verification |
|---|---|---|---|
| F10a | F10 | Initialize organic fraction with nucleation disabled (D002) | Positive registered-organic fixture passes checked budgets |
| F09b/F10b | F10 | Reproduce and repair activation velocity shape/units (D003) | Original poisoned-plane failure; signed/zero PDF cases agree |
| F09c/F10c | F10b | Reproduce and repair aerosol-water storage extent (D004) | Historical 17-class failure; runtime 7/17-class storage checks |
| F09d | F10c | Isolate standalone radiation initialization/data failure (D005) | Fresh checked run and optical-input completeness inventory |
| F10d | F09d | Implement justified radiation repair or explicit reference scope | Supported configuration passes; missing tables cannot silently pass |
| F09e | F10 | Reproduce absent-organic read in nucleation | Sulfate-only configuration fails at the actual gas access |
| F10e | F09e | Guard supported nucleation inputs | Sulfate-only and registered-organic cases retain physical budgets |
| F11a | F10 | Capture initialized scalar configuration and gas mapping | Copy-only overlay leaves all nine diagnostics unchanged |
| F11b | F11a | Capture full tracer/species table and array contracts | Mapping bounds, units and actual compiled dimensions verified |

F09b/F10b and F09c/F10c each form one small regression-and-repair commit;
F09d/F10d and F09e/F10e remain separate investigation and correction tasks.

## P1.1 — Build portability before source relocation

| ID | Depends on | Deliverable / suggested commit | Verification |
|---|---|---|---|
| O01 | F06,F07 | Discover NetCDF flags in normal Make build | Clean build without local paths; explicit user overrides still work |
| O02 | O01 | Make `configure` fail before replacing files when tools are missing | Missing `makedepf90` preserves current Makefile/dependencies |
| O03 | O02 | Regenerate complete module dependencies reproducibly | Clean `make -j` and serial produce matching outputs |
| O04 | O03 | Fix clean target and separate build artifacts from source | Clean removes only generated build files; subsequent rebuild works |

## P1.2 — Mechanical moves and configuration

| ID | Depends on | Deliverable / suggested commit | Verification |
|---|---|---|---|
| O05 | O04,F16 | Move root `.F90` set into `fortran/src` and update build/oracle paths | Old/new source hash map identical; captures numerically identical |
| O06 | O05 | Move include files into `fortran/include` | Includes resolve; clean build and baseline unchanged |
| O07 | O06 | Separate box drivers into `fortran/drivers` | One program entry point; same link objects and trajectory |
| O08 | O07 | Move sample namelist into named case directory with compatibility entry point | Same effective switches and relative input resolution |
| O09 | O08 | Add configurable output directory for the box driver | Defaults preserved; two runs cannot overwrite each other |
| O10 | O09 | Externalize run duration, timestep and forcing parameters | Default case matches original; altered case actually changes forcing |
| O11 | O10 | Catalog scientific tables and historical sample data | Hashes, units, source/purpose recorded; no input deletion |
| O12 | O11 | Update README/contributor commands and directory links | Fresh checkout follows docs without undocumented working directory |

Use one subdirectory move per commit, retaining original source bytes. More
granular Fortran process directories are optional after this layout works; Python
process isolation does not require rearranging every legacy module.
