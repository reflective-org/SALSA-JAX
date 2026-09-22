# Complete native SALSA composition

`src/salsa_jax/model.py:salsa_step` composes the supported standalone native
`mo_ham_salsa:salsa` call. Coagulation, condensation and nucleation retain
separate process APIs and original-Fortran comparisons. This entry point adds
the native interface conversions and executes their actual shared call order:

1. Convert the 69 native mass components into 17 bins × five material volumes.
2. Remove native tiny particles; equilibrate water and wet diameters.
3. Run optional coagulation, then optional diameter recalculation (`recalculate`).
4. Run optional condensation with its nested nucleation source and insertion.
5. Perform **two** ordered remapping → cleanup → hydration → radii/water passes.
6. Compute per-region density and convert represented material volumes to mass.

This covers a native microphysics call. The [seven-mode number initializer](initialization.md) is now independently
accepted. Material/tracer initialization, the HAM wrapper, evolving forcing,
radiation and the complete box driver remain separate roadmap tasks. The preserved Fortran files and scientific inputs are unchanged.

## Inputs, outputs and configuration

Enable JAX float64. Arrays share arbitrary leading cell axes. `mass` ends in
69 components: sulfate is **molecules/cm³**, other materials **µg/m³**. `number`
ends in 17 bins (particles/m³); `gas` ends in three canonical slots
SO2/H2SO4/OC (molecules/m³). An absent organic gas is padded with zero and
`organic_enabled=False`. `component_map` ends in `(17, 5)` and maps
SO4/OC/BC/SS/DU to zero-based mass components; `-1` denotes an absent component.
Validate maps on the host with `interface.validate_component_map`.

Environment inputs are kelvin, relative-humidity fraction, pascals and seconds.
Species properties come from `HydrationParameters`; material properties,
geometry and accommodation coefficients may vary between cells. Nucleation
supports schemes 0–3 and growth options 1–3, native fixed coefficients and a
3 nm target diameter. Active nucleation requires positive timesteps. Low-level
callers must supply finite, nonnegative state and positive material properties,
temperature, pressure, ordered regime limits and number threshold; `valid` does
not replace this input contract.

`SalsaState` returns mass, number, gas, material volume, dry core volume, wet
diameter, dry/wet radii, density, water mass and per-cell validity. Volume is
m³ material/m³ air, core m³/particle, wet diameter m, radii **cm**, density
**g/cm³**, and water kg/m³ air. Repeated calls must feed back **mass, number and
gas**, preserving native encoding and omitted material slots. Inputs are immutable.

Process switches, nucleation scheme/growth and `return_trace` are static JIT
arguments; group batches by these options. `boundary_layer` is a runtime mask
for native `level >= INT(ppbl)`. `return_trace=True` also returns intermediate
process states for diagnosing the earliest mismatch.

```python
import jax
jax.config.update("jax_enable_x64", True)
from salsa_jax.model import STATIC_OPTIONS, salsa_step
from tools.salsa_fixture import load_native_case
from tools.salsa_inputs import arguments

cases, expected, provenance = load_native_case()
args, kwargs = arguments(cases)
advance = jax.jit(salsa_step, static_argnames=STATIC_OPTIONS)
state = advance(*args, **kwargs)
state = advance(state.mass, state.number, state.gas, *args[3:], **kwargs)
```

The tools adapter is a reference workflow; production callers can supply the
same documented arrays directly. Configurations and fixture provenance live
outside the scientific kernels.

## Reference and acceptance

`tools/salsa_oracle.py` compiles all five original scientific modules byte for
byte, using data-only support stubs and native grid/coagulation initialization.
It runs two vertical cells per case, covering above/within-boundary-layer
behavior. JAX comparisons and timing use those same two cells. Native fixtures
add acceptance against the fully initialized checked first-call driver.

The selected matrix covers process switches, diameter recalculation, insoluble hydration,
soluble transfer, registered organic gas, nucleation schemes/growth options,
component permutations and initial tiny cleanup; it is not the Cartesian product
of every switch. Physical comparisons use
`rtol=3e-13, atol=0`. Water uses an additional `32*eps*dry_core*water_density*N`
cancellation allowance; signed native water is retained. `valid` flags cleanup
and nonfinite failures, **not** conservation or scientific validity.

```sh
.venv/bin/python -m pytest -q tests/test_salsa.py tests/test_salsa_native.py
.venv/bin/python -m pytest -q tests/test_interface.py tests/test_cleanup.py
```

Checked D007 cleanup failure returns NaN physical arrays and `valid=False` in
JAX, isolated per cell. Existing coagulation volume creation (D009), invalid
zero-volume hydration (D010), remapping orphan material (D011), and tiny-particle
nonvolatile loss (D012) remain inherited behavior. Read the
[repair record](reference-repairs.md) before interpreting physical budgets.

## Short repeated calls and numerical conditioning

Three-call checks feed each returned mass/number/gas state back through the
native interface with fixed environment. Default, active nucleation,
recalculation, disabled processes and insoluble-hydration configurations retain
the strict comparison. These are core trajectories, not evolving box forcing.

The soluble-transfer case almost empties native bin 13. By call three its number
is about 0.634 particles/m³, compared with an initial 5.53e7; the JAX/native
absolute difference is 1.22e-8 particles/m³ (relative 1.92e-8). Perturbing the
native initial mass by one representable float changes the native final number
by a similar amount. This is cancellation sensitivity in the original scheme.
The separate regression allows `32*eps*initial_same_component` only for that
bin's number and its four represented masses at calls two/three. All other
entries keep their strict bounds. This records a relative-accuracy limitation;
it does not justify relaxing unrelated fields or longer trajectories.

## Captured configurations and provenance

`tests/fixtures/native-default`, `native-active`, `native-recalc`, and
`native-active-recalc` preserve the actual complete native boundaries. Active
profiles select scheme 2/growth 1 with explicit sulfate forcing; recalculation
is configured before native initialization on copied sources. For each new
profile, checked control/capture outputs and fresh-process repeats are exact.
Corresponding `salsa-boundaries-*` fixtures retain initial hydration,
coagulation, condensation and both remap/correction boundaries from the same run.

The [matrix and short-call report](evidence/salsa-composition-2026-09-21.json)
contains compiler/source/case hashes, fieldwise errors and inherited budget
changes. The [native capture report](evidence/salsa-native-2026-09-21.json)
records profiles, input hashes and control/repeat invariance. Fixture manifests
record the complete configuration and regeneration instructions; the
[quickstart](quickstart.md#complete-native-salsa-call) provides commands.

## Complete-call CPU benchmark

The final full suite passes **658 tests in 128.53 s** in the repository `.venv`.
After validation jobs stopped, five trials of 500 fixed-input repetitions gave:

| Complete native workload | JAX warm median | Fortran warm median | Fortran/JAX |
|---|---:|---:|---:|
| Default case, 2 cells | 100.84 µs | 17.028 µs | 0.1689 |
| 27 cases, 54 cells in 21 static groups | 2149.49 µs | 405.91 µs | 0.1888 |

There is **no measured CPU speedup**: JAX is about 5.9× and 5.3× slower in these
workloads. Compilation is separate: 0.449 s for default and 9.601 s across the
21 groups. The heterogeneous matrix includes static-group dispatch overhead;
it is not a large homogeneous ensemble or accelerator benchmark.

Both sides produce the same seven native caller outputs for two levels per
case. Fortran timing includes native case/property setup, grid initialization,
input reset and output copies. JAX includes grid computation, complete process
composition, dispatch and synchronization of every physical output. Native
startup/I/O, initial JAX transfers, compilation, checksums and comparisons are
excluded from warm timing. Repetitions reset inputs; they do not advance a
trajectory. Checked and optimized Fortran outputs agree exactly before timing.

```sh
.venv/bin/python -m benchmarks.benchmark_salsa --single-case --repetitions 500 --trials 5
.venv/bin/python -m benchmarks.benchmark_salsa --repetitions 500 --trials 5
```

Raw [default](evidence/salsa-benchmark-single-2026-09-21.json) and
[matrix](evidence/salsa-benchmark-batch-2026-09-21.json) evidence records all
samples, thread environment, versions, source/case hashes and acceptance errors.
These measurements cover composed native microphysics with reference setup;
full HAM box-model performance remains unmeasured.
