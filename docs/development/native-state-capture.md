# Native SALSA entry/exit capture

Use the checked first-microphysics scope to capture exactly one native `salsa`
call, before radiation, activation and removal:

```sh
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-checked.toml --scope first-microphysics \
  --capture-salsa-state --run --output build/native-state
```

This option also enables initialized runtime metadata. It instruments only the
copied `mo_ham_salsa.F90`; the production source and its CRLF line endings remain
unchanged. Repeated calls, unsupported SOA composition, duplicate/missing source
anchors, absent records or malformed captures fail explicitly.

## Files and precision

- `salsa-entry.txt`: initialized inputs, geometry, constants and numeric maps.
- `salsa-exit.txt`: changed state, output properties and final internal volumes.
- `salsa-mapping.txt`: actual species names and full tracer names/units/indices.
- `scope-state.txt`: the six HAM interface arrays after the call, written with
  native capture either enabled or disabled for an independent comparison.
- `manifest.json`: compiler/configuration, source/input/output hashes, overlays,
  parsed values, dimensions, units and completion status.

Real values use `ES26.17E3`, enough significant digits to round-trip binary64.
Each array carries its scalar type, rank and dimensions; values follow Fortran
column-major order. In Python, reconstruct with `reshape(shape, order="F")`.
Only active horizontal columns are captured at the native boundary. Indices in
mapping records remain one-based; zero denotes an absent material/gas mapping.

## Boundary interpretation

Native particle number is particles/m³ and gas is molecules/m³. `paerml` has a
flat component axis: sulfate uses molecules/cm³, while the other components use
µg/m³. `material_map` and component/bin/species/tracer tables define the mapping;
do not reshape it into a species cube. Internal `zvols` instead uses material
m³ per air m³, ordered sulfate, OC, BC, sea salt, dust.

Native exit radii are cm and density is g/cm³; the HAM wrapper subsequently
converts these to SI. Radius, density and water arrays are outputs for this
caller despite their `INOUT` declarations, so they are excluded from entry
capture. Reading uninitialized output storage would invalidate the experiment.
See the [audited state contract](salsa-state-contract.md) for details and limits.

## Checked fixture

Export only a completed checked capture into a new directory:

```sh
.venv/bin/python tools/export_state_fixture.py \
  --run-directory build/native-state --output build/native-fixture
```

The reviewed fixture is `tests/fixtures/native-default/`. Its manifest pins
compiler flags, source/input hashes and all three raw capture files. Loading
verifies hashes and mappings; the exporter refuses overwrites, altered files
and unchecked runs. Regeneration requires review before replacing a fixture.

Capture exposed D008, an initializer overwrite of registered component maps.
The fixture includes the separately committed correction. Tests check native
volume conversions and budgets, accounting for SALSA's existing sea-salt to
sulfate volume transfer into insoluble bins.

## Verification and limits

Compare fresh checked scoped runs with and without `--capture-salsa-state`.
Their `scope-state.txt` hashes must match; this compares full-precision returned
state/tendencies/properties rather than formatted legacy diagnostics alone.
Repeat the native capture in another process and compare all three native files.

The first fixture is a single default call with nucleation disabled. Capturing
it is not complete process parity or trajectory acceptance. Full model radiation
remains unsupported, and optional/edge configurations need their own checked
cases. Next steps are per-process boundaries and a validated Python state API.

The [integration evidence](evidence/native-state-2026-09-21.json) records the
repaired capture, a control without native instrumentation and a fresh repeat.
All six full-precision HAM arrays match the control, and all three native files
repeat exactly. The complete virtual-environment suite passes **142 tests**.
