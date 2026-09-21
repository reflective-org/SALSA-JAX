# Effective Fortran runtime metadata

The optional F11 capture emits actual initialized Fortran values before the time
loop. It instruments only the copied driver in a new isolated run directory;
the checkout's scientific sources and namelist remain unchanged.

```sh
.venv/bin/python tools/fortran_reference.py --run --capture-runtime \
  --output build/reference-with-runtime
```

Add `--config configs/fortran-checked.toml` for the checked build. Omit `--run` to
build the instrumentation without claiming that runtime values were captured.

## Captured evidence

`runtime-metadata.txt` contains versioned typed integer, logical and real records,
plus an ordered gas-species mapping. Values are written after initialization and
before the first process call, using full-precision real formatting. They include:

- Actual real kind, storage size, precision/range, and default-real storage size.
- Grid, tracer, species, class and component counts; SALSA bin boundaries.
- Microphysics, nucleation, hydration, removal and activation switches.
- Sulfate/organic-gas and cloud-tracer indices, gas species indices and conversion
  codes. `isubm_ocnv=0` explicitly records the absent-organic-gas sentinel.
- Timestep in seconds and pressure, temperature, humidity, density and cloud
  fraction at grid point `(1,1)`, with units/indices in field names.

Values come from compiled/runtime objects, not namelist parsing. Module switch
values describe configuration; they do not prove a process actually executed.
The gas-map index is one-based and the species index addresses the runtime
species table. Conversion codes preserve the Fortran enum values.

The manifest records original/instrumented driver hashes, the retained
`runtime-metadata.patch` hash, helper hash, runtime-file hash and parsed metadata.
Missing, duplicate, mistyped, nonfinite or dimensionally inconsistent records
fail capture. A successful executable exit with failed capture makes the command
fail. A later scientific crash can still retain valid initialization metadata,
while the overall run remains failed.

## Scope and verification

This first slice does not capture full tracer tables, complete meteorological
arrays, evolving state, branch execution or completed step counts. Those belong
to subsequent capture tasks. Metadata completeness is not scientific validation;
outstanding checked-build and uninitialized-input defects remain separate.

Parser/overlay checks:

```sh
.venv/bin/python -m pytest -q tests/test_runtime_metadata.py tests/test_fortran_reference.py
```

Before accepting a new overlay, run the same case with and without
`--capture-runtime` and compare scientific diagnostics, recording any undefined
initial rows separately. Verify the original checkout's driver hash is unchanged.

On 2026-09-21, paired ordinary runs after the D003/D004 driver changes produced
identical hashes for all nine captured diagnostic files; the checkout's driver
was unchanged by instrumentation. Runtime values included 64-bit `dp`, 109
tracers, 17 classes, `nsol=17`, two gas species, `isubm_ocnv=0`, 20 updraft bins
and a one-second timestep. A separate checked run retained identical valid
initialization metadata but subsequently failed; full-model checked acceptance
remains open. The focused parser/overlay/harness suite passed 26 tests.
Exact provenance and comparisons are retained in
[runtime capture evidence](evidence/runtime-capture-2026-09-21.json) and
[combined reference repair evidence](evidence/reference-repairs-2026-09-21.json).
