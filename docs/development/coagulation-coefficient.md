# First port: Brownian coagulation coefficient

## Contract and source

`src/salsa_jax/processes/coagulation.py:coagulation_coefficient` translates
`mo_ham_salsa_init.F90:coagc` (baseline lines 292–382). It computes a collision
coefficient, **not** particle transfer, number loss or the full coagulation step.

| Input | Unit / domain |
|---|---|
| `diameter1`, `diameter2` | m, positive |
| `mass1`, `mass2` | kg, positive |
| `temperature` | K, positive |
| `pressure` | Pa, positive |
| Return | m³/s; broadcast input shape, scalar for scalar inputs |

The calculation follows the original air viscosity and mean-free-path fits,
Cunningham slip correction, particle diffusion/thermal velocities, and matching
between continuum and free-molecular collision fluxes. Expression order is kept
close to Fortran. Legacy `ak`, `p0sl_bg` and `pi` values are centralized in
`constants.py`; updating them to newer recommended values would change the model.
The empirical fit coefficients remain with the equation, not as user controls.

Callers enable JAX float64 before creating arrays. No invalid-domain clipping or
alternative physical formula has been added. The positive domain is an API
precondition; the tested range below is narrower than all positive real numbers.

## Oracle and cases

`tools/coagulation_oracle.py` extracts the actual function text without translating
it, compiles the original precision/math/physical constant modules and a narrow
driver, and records source/function/driver hashes and compiler flags. Each test
session builds a fresh executable; no imported sibling project is required.

Cases use 17 geometrically spaced diameters from 3 nm to 10 µm, all diameter
pairs, 230/273.15/310 K and 30000/101325 Pa. Particle densities are 1500 and
2100 kg/m³ to expose mistaken mass/diameter indexing: 1734 cases in total.
Tests compare scalar, broadcast, eager and JIT behavior and check symmetry,
finiteness and positivity. This unit has no material-transfer conservation test
because it does not change state; those belong to the complete process port.

Measured maximum relative error is below 9e-15 on the first tested toolchain.
Acceptance uses `rtol=1e-13`, `atol=0`, leaving roundoff headroom without masking
small coefficients with an absolute floor. It is not a universal tolerance for
all future processes or stress domains. Add reproducing cases before extending
the domain or altering expression order.

## Benchmark boundary

The benchmark times a batch of the same cases in a Fortran loop and JAX array
kernel. Fortran uses warmed internal `SYSTEM_CLOCK` wall time, separate compilation
without LTO, and checksum calculation outside the timer. JAX times repeated calls
to a compiled function with `block_until_ready`; input transfer and compilation
are separate. JAX dispatch remains included. Trial order alternates, and metadata
records thread settings without claiming equal thread allocation.

The first result shows JAX slower for this workload; see the
[results](results-2026-09-21.md). A complete-process or ensemble result may differ
and must be measured independently.
