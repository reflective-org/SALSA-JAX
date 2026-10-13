# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Native five-material SALSA hydration with its original convergence criterion."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salsa_jax.constants import (
    AVOGADRO, BOLTZMANN, PI, HYDRATION_SPECIES_MOLAR_MASS,
    HYDRATION_SPECIES_DENSITY, WATER_MOLAR_MASS, WATER_DENSITY,
    WATER_SURFACE_TENSION, ORGANIC_SOLUBLE_FRACTION, HYDRATION_NUMBER_THRESHOLD,
)


class HydrationParameters(NamedTuple):
    """Initialized species/configuration data; scalar fields allow case axes.

    Species order is SO4/OC/BC/SS/DU. Molar masses are g/mol, densities kg/m3,
    surface tension J/m2, soluble fraction dimensionless and threshold #/m3.
    Species properties have a final axis of length five.
    """
    species_molar_mass: tuple = HYDRATION_SPECIES_MOLAR_MASS
    species_density: tuple = HYDRATION_SPECIES_DENSITY
    water_molar_mass: float = WATER_MOLAR_MASS
    water_density: float = WATER_DENSITY
    organic_soluble_fraction: float = ORGANIC_SOLUBLE_FRACTION
    water_surface_tension: float = WATER_SURFACE_TENSION
    number_threshold: float = HYDRATION_NUMBER_THRESHOLD


class HydrationState(NamedTuple):
    """Dry core volume [m3/particle], wet diameter [m], per-bin loop counts."""
    core: jax.Array
    wet_diameter: jax.Array
    iterations: jax.Array


def solubility_mask(hydrate_insoluble=True):
    """Native initialized flags: 1a/2a soluble; 2b follows lhydr2b."""
    return (jnp.arange(17) < 10) | jnp.asarray(hydrate_insoluble, dtype=bool)[..., None]


def inverse_molality(activity, *, organic_soluble_fraction=ORGANIC_SOLUBLE_FRACTION,
                     water_molar_mass=WATER_MOLAR_MASS):
    """Return native binary inverse molalities (...,5) [kg water/mol solute].

    Water activity must lie strictly between zero and one. Scalar/array inputs
    broadcast normally. The soluble OC fraction is included; BC/DU return zero.
    The source's direct polynomial order is retained rather than refactored.
    """
    aw, fraction, water_mass = jnp.broadcast_arrays(*[
        jnp.asarray(v, dtype=jnp.float64) for v in (activity, organic_soluble_fraction, water_molar_mass)])
    sulfate = 1.0 / (1.1065495e2 - 3.6759197e2 * aw + 5.0462934e2 * aw**2 -
                     3.1543839e2 * aw**3 + 6.770824e1 * aw**4)
    organic = fraction / (1.0 / (aw * (water_mass * 1e-3)) - 1.0 / (water_mass * 1e-3))
    salt = 1.0 / (5.875248e1 - 1.8781997e2 * aw + 2.7211377e2 * aw**2 -
                  1.8458287e2 * aw**3 + 4.153689e1 * aw**4)
    zero = jnp.zeros_like(aw)
    return jnp.stack((sulfate, organic, zero, salt, zero), axis=-1)


def equilibrate(number, volume, relative_humidity, temperature, dry_diameter, *,
                soluble_mask=True, parameters=HydrationParameters()):
    """Return per-particle dry volume and equilibrium wet diameter for 17 bins.

    Number (...,17) [#/m3] and material volume (...,17,5) [m3/m3] have matching
    leading axes; supplied dry midpoint diameter (...,17) is m. RH is a fraction
    and T is K; environment/properties broadcast over leading case axes. Enable
    JAX float64. Inputs remain unchanged. Active bins (number > threshold) must
    have positive total dry volume. Other concentrations are nonnegative and
    physical properties positive; soluble fractions lie in [0,1].

    Native RH clamps to [.05,.95]. Actual sizeclass solubility flags govern
    iteration; use solubility_mask(lhydr2b) for native initialized flags. The
    default matches lhydr2b=True. Empty bins use midpoint-derived volume/diameter.
    Each active soluble bin retains the native one-percent diameter criterion,
    initial Kelvin factor 1.001 and uncapped loop; no new convergence repair.
    A soluble active zero-volume bin or nonfinite Kelvin factor yields a NaN
    wet diameter, exposing the native floating-point failure without a JIT hang.
    """
    number, volume, dry = (jnp.asarray(v, dtype=jnp.float64) for v in (number, volume, dry_diameter))
    if (number.shape[-1:] != (17,) or volume.shape != number.shape + (5,) or dry.shape[-1:] != (17,)):
        raise ValueError('Expected number (...,17), volume (...,17,5), dry diameter (...,17)')
    leading = number.shape[:-1]
    def environment(value):
        return jnp.broadcast_to(jnp.asarray(value, dtype=jnp.float64), leading)[..., None]
    dry = jnp.broadcast_to(dry, number.shape)
    rh = jnp.clip(environment(relative_humidity), .05, .95)
    temperature = environment(temperature)
    threshold = environment(parameters.number_threshold)
    density = jnp.broadcast_to(jnp.asarray(parameters.species_density, dtype=jnp.float64), leading + (5,))
    mass = jnp.broadcast_to(jnp.asarray(parameters.species_molar_mass, dtype=jnp.float64), leading + (5,))
    water_density = environment(parameters.water_density)
    water_mass = environment(parameters.water_molar_mass)
    fraction = environment(parameters.organic_soluble_fraction)
    surface_tension = environment(parameters.water_surface_tension)
    active = number > threshold
    soluble = active & jnp.broadcast_to(jnp.asarray(soluble_mask, dtype=bool), number.shape)
    divisor = jnp.where(active, number, 1.0)
    particle_volume = volume / divisor[..., None]
    # Scalar Fortran SUM over the five species; preserve accumulation order.
    core = particle_volume[..., 0]
    for index in range(1, 5):
        core = core + particle_volume[..., index]
    core = jnp.where(active, core, (PI / 6.0) * dry**3)
    dry_actual = (core / (PI / 6.0))**(1.0 / 3.0)
    invalid = soluble & (core <= 0)
    wet = jnp.where(invalid, jnp.nan, jnp.where(soluble, 0.0, jnp.where(active, dry_actual, dry)))
    previous = jnp.ones_like(number)
    kelvin = jnp.full_like(number, 1.001)
    iterations = jnp.zeros_like(number, dtype=jnp.int32)
    sulfate_molecular_volume = (mass[..., 0, None] / 1000.0) / AVOGADRO / density[..., 0, None]

    def condition(state):
        wet, previous, _, _ = state
        return jnp.any(soluble & (jnp.abs(wet - previous) > 1e-2 * jnp.abs(previous)))

    def update(state):
        wet, previous, kelvin, iterations = state
        running = soluble & (jnp.abs(wet - previous) > 1e-2 * jnp.abs(previous))
        old = jnp.maximum(wet, 1e-20)
        molality = inverse_molality(rh / kelvin, organic_soluble_fraction=fraction, water_molar_mass=water_mass)
        water_content = jnp.zeros_like(number)
        for species in range(5):
            water_content = water_content + (volume[..., species] * density[..., species, None] /
                                             mass[..., species, None] * 1e3 * molality[..., species])
        updated = (water_content / divisor / water_density / (PI / 6.0) + core / (PI / 6.0))**(1.0 / 3.0)
        # Bins not iterating have no Kelvin update in Fortran. Safe placeholders
        # avoid division by zero on their traced, unselected arithmetic path.
        next_kelvin = jnp.exp(2.0 * surface_tension * sulfate_molecular_volume /
                             (BOLTZMANN * temperature * jnp.where(running, updated, 1.0)))
        updated = jnp.where(jnp.isfinite(next_kelvin), updated, jnp.nan)
        return (jnp.where(running, updated, wet), jnp.where(running, old, previous),
                jnp.where(running, next_kelvin, kelvin), iterations + running.astype(jnp.int32))

    wet, _, _, iterations = jax.lax.while_loop(condition, update, (wet, previous, kelvin, iterations))
    return HydrationState(core, wet, iterations)
