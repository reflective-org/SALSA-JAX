"""Standalone SALSA condensation, with nucleation disabled (nsnucl=0).

Ports the D001/D002-repaired mo_ham_salsa_dynamics.F90:condensation without
HAMMOZ/VBS. Number is unchanged; hydrated diameters are supplied by the caller.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salsa_jax.constants import (
    AVOGADRO, BOLTZMANN, GAS_CONSTANT, PI, REFERENCE_PRESSURE,
    SULFATE_MOLAR_MASS, ORGANIC_MOLAR_MASS, SULFATE_DENSITY, ORGANIC_DENSITY,
    SULFURIC_ACID_DIAMETER, CONDENSATION_GAS_THRESHOLD, CONDENSATION_SINK_THRESHOLD,
    NUCLEATION_CLUSTER_MOLECULES,
)


class CondensationParameters(NamedTuple):
    """Species/configuration data; scalar properties may carry leading case axes.

    Molar mass is g/mol and density kg/m3. Mass accommodation
    is dimensionless: a positive scalar, (17,), or (...,17). Defaults are the
    actual initialized standalone species, not older synthetic regression data.
    """

    sulfate_molar_mass: float = SULFATE_MOLAR_MASS
    sulfate_density: float = SULFATE_DENSITY
    organic_molar_mass: float = ORGANIC_MOLAR_MASS
    organic_density: float = ORGANIC_DENSITY
    mass_accommodation: float = 1.0


class CondensationState(NamedTuple):
    """Number [#/m3], material volumes [m3/m3], canonical gas [molecules/m3]."""

    number: jax.Array
    volume: jax.Array
    gas: jax.Array


def condensation_rates(number, wet_diameter, temperature, pressure, *, parameters=CondensationParameters()):
    """Return per-bin vapor collision sinks (...,17) [s-1], including number.

    Number is particles/m3, wet diameter m, temperature K, pressure Pa. All
    inputs must be finite, number nonnegative, others positive. The first two
    bins include molecular size and particle diffusion; later bins do not.
    Leading environmental/property axes broadcast against the state axes.
    """
    number = jnp.asarray(number, dtype=jnp.float64)
    diameter = jnp.asarray(wet_diameter, dtype=jnp.float64)
    if number.shape[-1:] != (17,) or diameter.shape[-1:] != (17,):
        raise ValueError("Condensation requires the 17-bin SALSA grid")
    temperature = jnp.asarray(temperature, dtype=jnp.float64)[..., None]
    pressure = jnp.asarray(pressure, dtype=jnp.float64)[..., None]
    molar_mass = jnp.asarray(parameters.sulfate_molar_mass, dtype=jnp.float64)[..., None]
    accommodation = jnp.asarray(parameters.mass_accommodation, dtype=jnp.float64)
    viscosity = 7.44523e-3 * temperature**1.5 / (5093.0 * (temperature + 110.4))
    vapor_diffusion = 5.1111e-10 * temperature**1.75 * REFERENCE_PRESSURE / pressure
    mean_free_path = 3.0 * vapor_diffusion * jnp.sqrt(PI * (molar_mass / 1000.0) / (8.0 * GAS_CONSTANT * temperature))
    small_bins = jnp.arange(17) < 2
    collision_diameter = diameter + jnp.where(small_bins, SULFURIC_ACID_DIAMETER, 0.0)
    knudsen = 2.0 * mean_free_path / collision_diameter
    beta = (knudsen + 1.0) / (0.377 * knudsen + 1.0 + 4.0 / (3.0 * accommodation) * (knudsen + knudsen**2))
    particle_diffusion = BOLTZMANN * temperature * beta / (3.0 * PI * viscosity * diameter)
    return (2.0 * PI * collision_diameter *
            (vapor_diffusion + jnp.where(small_bins, particle_diffusion, 0.0)) * beta * number)


def condensation_step(number, volume, gas, wet_diameter, temperature, pressure, timestep, *,
                      organic_enabled=False, enabled=True, parameters=CondensationParameters()):
    """Condense sulfate and optionally nonvolatile organic vapor; no nucleation.

    State shapes: number (...,17), volume (...,17,5), gas (...,3), with matching
    leading axes. Materials are SO4/OC/BC/SS/DU; canonical gas slots are inert
    SO2/H2SO4/OC. An absent organic mapping uses organic_enabled=False and its
    slot is preserved, never indexed through a Fortran zero sentinel.

    Number/volume/gas and timestep must be nonnegative; physical properties,
    hydrated diameters and environment positive. organic_enabled broadcasts
    across cases; enabled is a scalar JAX-compatible switch. Enable float64.
    Inputs are immutable. Unsupported nucleation/VBS modes have no API here.
    """
    number, volume, gas = (jnp.asarray(value, dtype=jnp.float64) for value in (number, volume, gas))
    if (number.shape[-1:] != (17,) or volume.shape != number.shape + (5,) or
            gas.shape != number.shape[:-1] + (3,)):
        raise ValueError("Expected matching number (...,17), volume (...,17,5), gas (...,3)")
    timestep = jnp.asarray(timestep, dtype=jnp.float64)

    def condense():
        return _condense(number, volume, gas, wet_diameter, temperature, pressure, timestep,
                         organic_enabled, parameters)

    return jax.lax.cond(enabled, condense, lambda: CondensationState(number, volume, gas))


def _condense(number, volume, gas, wet_diameter, temperature, pressure, timestep,
              organic_enabled, parameters, sources=None):
    """Shared native vapor update; optional sources restore nucleation coupling."""
    rates = condensation_rates(number, wet_diameter, temperature, pressure, parameters=parameters)
    sink = jnp.sum(rates, axis=-1)
    new_number, new_gas, new_volume = number, gas, volume
    # The original collision sink is computed once, before any insertion.
    # Native organic uptake precedes sulfate uptake.
    for gas_slot, material_slot, molar_mass, density, active in (
        (2, 1, parameters.organic_molar_mass, parameters.organic_density, organic_enabled),
        (1, 0, parameters.sulfate_molar_mass, parameters.sulfate_density, True),
    ):
        vapor = gas[..., gas_slot]
        # Preserve the native background-sink gate even with a nucleation source.
        update = (jnp.asarray(active) & (vapor > CONDENSATION_GAS_THRESHOLD) &
                  (sink > CONDENSATION_SINK_THRESHOLD))
        vapor_rates, vapor_sink = rates, sink
        if sources is not None:
            source = sources.molecular_rate[..., material_slot]
            source_sink = source / jnp.where(update, vapor, 1.0)
            vapor_rates = rates.at[..., 0].add(source_sink)
            vapor_sink = sink + source_sink
        denominator = jnp.where(vapor_sink > CONDENSATION_SINK_THRESHOLD, vapor_sink, 1.0)
        fraction = vapor_rates / denominator[..., None]
        remaining = vapor / (1.0 + timestep * vapor_sink)
        remaining = jnp.where(update, remaining, vapor)
        molecular_volume = (jnp.asarray(molar_mass, dtype=jnp.float64) / 1000.0) / AVOGADRO / jnp.asarray(density, dtype=jnp.float64)
        addition = fraction * molecular_volume[..., None] * (vapor - remaining)[..., None]
        new_volume = new_volume.at[..., material_slot].add(addition)
        new_gas = new_gas.at[..., gas_slot].set(remaining)
        if sources is not None:
            composition = sources.fractions[..., material_slot]
            ratio = jnp.where(source > 1.0, source / jnp.where(source > 1.0, source + vapor * rates[..., 0], 1.0), 0.0)
            inserted = ratio * addition[..., 0] / molecular_volume / (NUCLEATION_CLUSTER_MOLECULES * jnp.where(composition > 0, composition, 1.0))
            new_number = new_number.at[..., 0].add(jnp.where(composition > 0, inserted, 0.0))
    return CondensationState(new_number, new_volume, new_gas)
