# Nucleation rate parameterizations

`src/salsa_jax/processes/nucleation_rates.py` independently ports the actual
`binnucl`, `actnucl` and `kinnucl` Fortran leaves. Enable JAX float64 first.

```python
from salsa_jax.processes.nucleation_rates import nucleation_rates
cluster = nucleation_rates(sulfate, temperature, relative_humidity,
                           scheme=2, boundary_layer=True)
```

Inputs are H2SO4 molecules/m³, K and RH fraction; arrays broadcast. The result
is `CriticalCluster(rate, sulfate_molecules, diameter)`, with rate in
particles/(m³ s), molecules per critical cluster and diameter in m. `scheme`
is static under JIT. The lower-level `binary_nucleation` uses sulfate in
**molecules/cm³**, matching the original leaf; the dispatcher performs this
conversion explicitly. Neither function updates gas or aerosol state.

- Scheme 1: binary sulfate/water parameterization throughout the column.
- Scheme 2: binary above the boundary layer, activation within it; the default
  activation coefficient is 1e-7 s⁻¹ and is an explicit runtime parameter.
- Scheme 3: binary above the boundary layer, kinetic within it.

`boundary_layer=True` corresponds to native `jj >= ppbl`. The shared boundary
level therefore takes the activation/kinetic override. Schemes 2/3 set the
critical diameter to 7.9375e-10 m even in cells retaining the binary rate.
This surprising native behavior is preserved deliberately.

Binary inputs clamp T to 190.15–300.15 K, RH to 1e-4–1 and sulfate to
1e4–1e9 molecules/cm³. The complete polynomial, low-rate threshold, collision
fallback and maximum-rate branch retain original constants and arithmetic.
Default-real literals in the fallback are promoted from their original
binary32 values; writing mathematically similar float64 literals changes
reference behavior. Other schemes are rejected, not silently approximated.

## Verification

```sh
.venv/bin/python -m pytest -q tests/test_nucleation_rates.py
```

The independent oracle mechanically extracts and compiles the actual Fortran
leaves. Eleven tests include a 216-case temperature/RH/sulfate grid, clamp and
threshold behavior, rate scaling, boundary overrides, scalar/batched inputs
and eager/JIT execution. On the recorded ARM64 CPU, maximum relative rate error
is 2.33e-12 eager and 2.22e-16 JIT; critical molecule/diameter errors are smaller.
The test tolerance is 1e-11 without an absolute floor. The grid exercises 109
zero-rate and 85 collision-fallback cases; it does not establish coverage of
the upper rate cap. Growth, survival and uptake require separate acceptance in
the [process guide](nucleation-process.md).
