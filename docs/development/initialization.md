# Seven-mode number initialization

`initialization.size_distribution` ports the complete native
`mo_ham_salsa_sizedist:size_distribution` routine. It takes seven modal number
concentrations (particles/m³), geometric mean diameters (m), geometric standard
deviations (dimensionless), and a `SizeGrid`. It returns 17 bin populations in
particles/m³, with broadcasting over leading cell axes. Enable JAX float64.

```python
import jax
jax.config.update("jax_enable_x64", True)
from salsa_jax.initialization import size_distribution
from tools.initialization_fixture import load_native_case

args, expected, capture = load_native_case()
number = jax.jit(size_distribution)(*args)
```

Supply finite nonnegative modal number, positive mean diameters, standard
deviations greater than one, positive grid diameters and increasing positive
volume bounds. Zero-number modes still need valid diameter/width parameters,
as in Fortran. Inputs remain unchanged. The kernel returns active cells; padded
inactive Fortran storage is a caller concern, tested separately in the oracle.

## Preserved numerical and physical choices

The native routine evaluates each lognormal density at the bin midpoint and
multiplies by the diameter-bin width. It does not integrate the lognormal CDF,
renormalize total number, or assign individual modes to soluble/insoluble
regimes. All seven modes contribute to every bin, so the seven overlapping
2a/2b bins receive identical populations. Consequently, summing all 17 bins
need not recover the sum of the modal inputs. This is inherited initialization
behavior, not a new conservation guarantee.

Driver literals such as `1.59` and `0.001` are evaluated as default-real
binary32 before promotion to double precision. The reference reads the actual
compiled driver assignments and captures their values; Python must not replace
them with ideal decimal64 defaults. Mode settings are runtime inputs to the
kernel, not a second hardcoded configuration.

This initializes **number only**. Native `conc2mmr` next assigns material
composition, changes number units and constructs HAM tracers. That separate
routine remains required before initialization can feed the full HAM wrapper.
The [HAM audit](ham-conversion-audit.md) describes these boundaries.

## Verification and regeneration

The independent oracle compiles the original size-distribution module byte for
byte, with data-only geometry support. Its 23-case matrix covers every isolated
mode, mixed/empty populations, scaling, shifted/narrow/broad modes and varied
grids. Tests compare eager, JIT and `vmap`, preserve duplicate-bin behavior and
check inactive native cells. Checked/optimized/repeated native outputs are exact.

```sh
.venv/bin/python -m pytest -q tests/test_initialization.py \
  tests/test_initialization_oracle.py tests/test_initialization_capture.py \
  tests/test_initialization_native.py
.venv/bin/python tools/fortran_reference.py \
  --config configs/fortran-checked.toml --scope first-microphysics \
  --capture-initialization-state --capture-salsa-state \
  --run --output build/initialization-new
```

Use a new run directory. The copy-only overlay surrounds the actual initialized
driver call before `conc2mmr`; it records full-precision modes, geometry, core
and number output. `tools.initialization_capture.export_fixture(run, target)`
exports a checked, hash-verified capture to a new fixture directory. The loader
rejects changed files, incompatible shapes/options and modified pure inputs.


The complete suite passes **740 tests** in the repository `.venv`. The 23-case
initializer comparison has maximum relative error 2.84e-14; no negative or
nonfinite populations occur. Checked control/capture outputs and fresh repeats
are exact. Independent reference output also matches the actual initialized
boundary and its subsequent native number-unit round trip into SALSA.

The [kernel evidence](evidence/initialization-2026-09-22.json) records source,
compiler and case hashes, per-case errors, modal/bin totals and literal rounding.
The [capture evidence](evidence/initialization-native-2026-09-22.json) records
three checked native runs and their output hashes. The complete box model still
requires material/tracer initialization, HAM coupling and the remaining driver
processes; this milestone does not validate those steps.

## Isolated CPU measurement

Five trials of 500 fixed-input repetitions were run after validation finished:

| Workload | JAX warm median | Optimized Fortran | Fortran/JAX |
|---|---:|---:|---:|
| One native default case | 5.712 µs | 1.070 µs | 0.1873 |
| All 23 cases | 38.399 µs | 21.308 µs | 0.5549 |

No CPU speedup is measured. JAX compilation is separate: 0.0422 s and 0.0447 s.
The benchmark times size-distribution evaluation on supplied geometry; grid
generation, material/tracer initialization and SALSA evolution are excluded.
Native geometry assignment and output zeroing are included, while modal arrays
are passed as existing slices. JAX dispatch and output synchronization are
included; initial transfers, compilation, native process startup and I/O are not.
Checked/optimized Fortran results must match exactly, and JAX must pass the
same 3e-13 relative bound before timing. Trials alternate execution order.

```sh
.venv/bin/python -m benchmarks.benchmark_initialization --single-case --repetitions 500 --trials 5
.venv/bin/python -m benchmarks.benchmark_initialization --repetitions 500 --trials 5
```

Raw [single-case](evidence/initialization-benchmark-single-2026-09-22.json) and
[23-case](evidence/initialization-benchmark-batch-2026-09-22.json) records include
all samples, thread settings, versions and source/case hashes.
