# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Seven-mode native SALSA number initialization on the sectional size grid."""
import jax.numpy as jnp

from salsa_jax.constants import PI


def size_distribution(mode_number, geometric_diameter, geometric_stddev, grid):
    """Return native bin number (...,17) [particles/m3] from seven modes.

    Mode inputs end in seven entries: total number [particles/m3], geometric
    mean diameter [m], and dimensionless geometric standard deviation. Leading
    axes broadcast with the SizeGrid fields (...,17). Enable JAX float64.

    Caller preconditions: finite nonnegative mode numbers, positive diameters,
    standard deviations >1 (including zero-number modes), and finite positive
    grid diameters with upper volumes greater than lower volumes. This retains
    native midpoint-density times bin-width quadrature, not an exact lognormal
    integral or a normalized distribution. Every mode contributes to every bin;
    identical 2a/2b geometry produces identical populations in both regimes.
    Material composition and HAM tracer initialization belong to conc2mmr.
    Inputs are immutable; no process or module state is retained.
    """
    modes = tuple(jnp.asarray(value, dtype=jnp.float64) for value in
                  (mode_number, geometric_diameter, geometric_stddev))
    if any(value.shape[-1:] != (7,) for value in modes):
        raise ValueError('Expected seven mode values (...,7)')
    lower, upper, diameter = (jnp.asarray(value, dtype=jnp.float64) for value in
                              (grid.vlolim, grid.vhilim, grid.dpmid))
    if any(value.shape[-1:] != (17,) for value in (lower, upper, diameter)):
        raise ValueError('Expected native grid fields (...,17)')
    number, mean, sigma = (value[..., None, :] for value in modes)
    width = (upper**(1.0/3.0)-lower**(1.0/3.0))/(PI/6.0)**(1.0/3.0)
    diameter = diameter[..., None]
    log_sigma = jnp.log(sigma)
    contributions = (number*width[..., None]/(diameter*jnp.sqrt(2.0*PI)*log_sigma)*
                     jnp.exp(-jnp.log(diameter/mean)**2/(2.0*log_sigma**2)))
    total = contributions[..., 0]
    for mode in range(1, 7):
        total = total + contributions[..., mode]
    return total
