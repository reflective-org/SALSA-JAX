# Aerosol material and HAM tracer initialization

`salsa_jax.aerosol_initialization.initialize_aerosol` implements the standalone
SALSA branch of `conc2mmr`. It follows the accepted seven-mode number initializer
and constructs all 69 material components and the corresponding HAM tracers.
It is an initialization recipe, not a conversion of an existing material state.

## Inputs and returned state

Input `number (...,17)` is particles/m³, `core` is m³/particle, and air density
is kg/m³. The existing tracer vector shares the number array's leading axes.
Species masses/densities come from `HydrationParameters`; `mw_so4` is a separate
HAM parameter in g/mol. Core, parameters and integer maps broadcast over cells.

`component_map (...,17,5)` uses zero-based component IDs and `-1` for absent
SO4/OC/BC/SS/DU slots. Validate it outside JIT using `validate_component_map`.
Component tracer indices and unit codes end in 69 entries; number tracer indices
end in 17. All tracer indices are zero-based and must be in bounds. Inputs must
be finite, number/core nonnegative and densities/molecular weights positive.

The immutable `AerosolInitialization` result contains:

- `number`: particles/**cm³**, exposing the native mutation explicitly.
- `mass`: 69 components; sulfate molecules/cm³ and other materials µg/m³.
- `tracers`: updated HAM mass/number mixing ratios, preserving unselected entries.

Enable JAX float64. Input arrays remain unchanged. Feeding returned number back
as particles/m³ repeats the native unit conversion incorrectly; this API is not
an idempotent general-purpose initializer.

## Native material recipe and arithmetic

| Region | SO4 | OC | BC | SS | DU |
|---|---:|---:|---:|---:|---:|
| 1a | .2 | .8 | 0 | 0 | 0 |
| 2a | .5 | .05 | .05 | .1 | .3 |
| 2b | .01 | .01 | .01 | 0 | .97 |

These are volume fractions and sum to one per bin. Each native expression starts
with `fraction * number * core`. Sulfate mass uses molecular volume calculated
from the actual registry sulfate molar mass, native Avogadro constant and density.
Other materials multiply by their density and divide by `1e-9`.

Tracer conversion uses the independent, truncated factor `6.022e20/mw_so4` when
the component's effective unit code equals 2. Every other code uses `1e9`,
including unknown codes. This differs from gas conversion's unknown-code no-op.
Do not replace either molecular constant or simplify these expressions into a
single unit transform. The rounded number intermediate `N*1e-6` is divided by
`1e-6*air_density` to form its tracer value.

Component-tracer writes occur in component order, followed by number-tracer
writes in bin order. Duplicate destinations retain the last write; a number
tracer wins a collision with an aerosol mass tracer. Physical budgets apply to
returned material mass, not an arbitrarily colliding tracer registry.

## Validation scope

`tools/aerosol_initialization_oracle.py` compiles the verbatim original
`conc2mmr` routine with data-only support registries. The JAX tests compare all
three outputs in eager and JIT execution, then independently verify material
volume/fraction budgets, runtime component permutations, tracer-write order,
untouched tracers and the separate sulfate parameters.

```sh
.venv/bin/python -m pytest -q tests/test_aerosol_initialization.py \
  tests/test_aerosol_initialization_oracle.py tests/test_ham_conversion_native.py
```

Actual driver captures surround initialization and the first `gas2mmr` call.
The latter receives the gas values **after** the driver's m³→cm³ scaling and
before `ham_subm_interface`. Full-cell density/pressure/temperature, complete
tracer vectors, initialized maps, conversion codes and species properties are
recorded rather than inferred from the SALSA interior.

This work does not port `mmr2conc`, `xt_borrow`, tendencies, water pseudo-tracer
updates or the complete HAM wrapper. Those remain distinct tasks in the
[HAM roadmap](../plans/03-coupled-processes.md).

## Recorded acceptance

The 18-case original-Fortran matrix has exact number output, maximum relative
mass error `2.20e-16`, and tracer error `3.79e-16`. Material-volume and fraction
budget errors are below `4.4e-16`. Untouched tracer values and input arrays are
exactly preserved. See [fieldwise evidence](evidence/aerosol-initialization-2026-09-22.json).

Six checked native runs cover default and active forcing, each with control,
capture and fresh repeat. Scope outputs are identical with capture enabled;
HAM, linked number-initialization and SALSA captures repeat exactly. Both
profiles pass independent reference replay and eager/JIT number → material → gas
tracer composition. Active sulfate forcing is `1e14 molecules/m³`, becoming
`1e8 molecules/cm³` at the captured gas boundary. See
[native provenance and hashes](evidence/ham-conversion-native-2026-09-22.json)
and [regeneration commands](quickstart.md#seven-mode-initialization-and-ham-gas-conversion).

## Isolated CPU measurement

| Curated workload | JAX warm median | Fortran warm median | JAX compilation |
|---|---:|---:|---:|
| One case | 42.15 µs | 0.352 µs | 58.09 ms |
| 18 cases | 68.49 µs | 5.630 µs | 75.53 ms |

No CPU speedup is measured. These are five trials of 500 identical-state calls,
with all number/mass/tracer outputs retained and synchronized. Checked and
optimized Fortran outputs match exactly before timing. The first curated case
uses native number/core/maps/properties but synthetic density `1.2 kg/m³` and
zero initial tracers; it is not the actual driver caller state. The full matrix
also includes mapping, unit-code, property and duplicate-destination stresses.

Native timing includes registry/property setup and mutable input copies; JAX
includes dispatch and ordered tracer updates with immutable inputs. Compilation,
initial transfers, process startup and I/O are excluded. Number generation,
gas conversion and the complete HAM wrapper are outside this measurement.

```sh
.venv/bin/python -m benchmarks.benchmark_aerosol_initialization --single-case --repetitions 500 --trials 5
.venv/bin/python -m benchmarks.benchmark_aerosol_initialization --repetitions 500 --trials 5
```

Raw [one-case](evidence/aerosol-initialization-benchmark-single-2026-09-22.json)
and [18-case](evidence/aerosol-initialization-benchmark-batch-2026-09-22.json)
records include compiler/flags, source/case hashes, runtime/thread settings,
parity checks and every timing sample.
