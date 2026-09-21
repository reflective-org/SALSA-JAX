"""Supported SALSA sulfate nucleation sources and legacy growth/survival.

Schemes 1/2/3 and growth options 1/2/3 retain the repaired native arithmetic.
This routine returns molecular source rates; condensation performs vapor
uptake and particle insertion. It does not mutate particle number or gas.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salsa_jax.constants import (
    AVOGADRO, BOLTZMANN, GAS_CONSTANT, PI, REFERENCE_PRESSURE,
    SULFURIC_ACID_DIAMETER, NUCLEATION_ACTIVATION_COEFFICIENT,
    NUCLEATION_CLUSTER_MOLECULES, NUCLEATION_TARGET_DIAMETER,
    NUCLEATION_CRITICAL_DIAMETER,
)
from salsa_jax.processes.condensation import CondensationParameters
from salsa_jax.processes.nucleation_rates import nucleation_rates


class NucleationSources(NamedTuple):
    """SO4/OC molecular rates [molecules/m3/s] and composition fractions."""

    molecular_rate: jax.Array
    fractions: jax.Array


def _cluster_sink(diameter, wet_diameter, number, temperature, density, mean_free_path):
    """Native coagulation sink for a critical or target cluster, including its quirks."""
    cluster_mass = 4.0 / 3.0 * PI * (diameter / 2.0)**3 * density
    particle_mass = 4.0 / 3.0 * PI * (wet_diameter / 2.0)**3 * density[..., None]
    cluster_velocity = jnp.sqrt(8.0 * BOLTZMANN * temperature / (PI * cluster_mass))
    particle_velocity = jnp.sqrt(8.0 * BOLTZMANN * temperature[..., None] / (PI * particle_mass))
    combined_velocity = jnp.sqrt(cluster_velocity[..., None]**2 + particle_velocity**2)
    viscosity = 1.81e-5 * (temperature / 293.0)**0.74
    knudsen_cluster = 2.0 * mean_free_path / diameter
    knudsen_particle = 2.0 * mean_free_path[..., None] / wet_diameter
    correction_cluster = 1.0 + knudsen_cluster * (1.142 + 0.558 * jnp.exp(-0.999 / knudsen_cluster))
    correction_particle = 1.0 + knudsen_particle * (1.142 + 0.558 * jnp.exp(-0.999 / knudsen_particle))
    diffusion_cluster = BOLTZMANN * temperature * correction_cluster / (3.0 * PI * viscosity * diameter)
    diffusion_particle = BOLTZMANN * temperature[..., None] * correction_particle / (3.0 * PI * viscosity[..., None] * wet_diameter)
    diffusion = diffusion_cluster[..., None] + diffusion_particle
    gamma_cluster = 8.0 * diffusion_cluster / PI / cluster_velocity
    gamma_particle = 8.0 * diffusion_particle / PI / particle_velocity
    radius = diameter[..., None] / 2.0 + wet_diameter / 2.0
    # Native expressions contain radius**2 + gamma (not gamma**2). Preserve
    # the literal source; changing dimensional consistency needs separate review.
    omega_cluster = ((radius + gamma_cluster[..., None])**3 -
                     (radius**2 + gamma_cluster[..., None])**1.5) / (3.0 * radius * gamma_cluster[..., None]) - radius
    omega_particle = ((radius + gamma_particle)**3 - (radius**2 + gamma_particle)**1.5) / (3.0 * radius * gamma_particle) - radius
    sigma = jnp.sqrt(omega_cluster**2 + omega_particle**2)
    coefficient = 4.0 * PI * radius * diffusion / (radius / (radius + sigma) + 4.0 * diffusion / (combined_velocity * radius))
    return jnp.maximum(1e-20, jnp.sum(coefficient * number, axis=-1)), cluster_velocity


def cluster_population_taylor(diameter, target, exponent, rate, eta, growth_rate):
    """Native 30-term zNnuc_tayl series; no convergence or stability repair."""
    total = jnp.zeros_like(rate)
    factorial = 1.0
    for i in range(30):
        if i > 1:
            factorial *= float(i)
        power = float(i) * (exponent + 1.0) + 1.0
        term = eta**i / (power * factorial)
        total = total + term * (target**power - diameter**power)
    return total * rate * jnp.exp(-eta * diameter**(exponent + 1.0)) / growth_rate


def nucleation_sources(number, gas, wet_diameter, temperature, relative_humidity, pressure, timestep, *,
                       scheme=2, growth=1, boundary_layer=True, organic_enabled=False,
                       activation_coefficient=NUCLEATION_ACTIVATION_COEFFICIENT,
                       parameters=CondensationParameters()):
    """Return supported sulfate nucleation source rates, retaining native growth.

    Number (...,17) [#/m3], canonical gas (...,3) SO2/H2SO4/OC [molecules/m3],
    wet diameter (...,17) [m], temperature [K], RH fraction, pressure [Pa] and
    positive timestep [s]. Scalar/leading environment axes broadcast. Number
    and gas must be nonnegative; physical properties positive. ``boundary_layer``
    marks native jj >= ppbl (including the shared boundary level).

    scheme/growth are static JIT arguments. Scheme0 returns zero sources and
    fractions. Schemes1/2/3 use binary/activation/kinetic rates; scheme4 lacks
    required NH3 and 5+ are outside the supported initialized configuration.
    Fractions are SO4=1,OC=0 even in inactive cells for schemes1–3. Registered
    organic vapor influences cluster growth but not the supported nucleation
    chemistry. Source arrays start at zero as in the actual condensation caller.
    """
    if scheme not in (0, 1, 2, 3) or growth not in (1, 2, 3):
        raise ValueError("Supported nucleation schemes are 0/1/2/3 and growth options 1/2/3")
    number, gas, wet = (jnp.asarray(v, dtype=jnp.float64) for v in (number, gas, wet_diameter))
    if number.shape[-1:] != (17,) or gas.shape != number.shape[:-1] + (3,) or wet.shape[-1:] != (17,):
        raise ValueError("Expected number (...,17), gas (...,3), wet diameter (...,17)")
    shape = number.shape[:-1]
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    if scheme == 0:
        return NucleationSources(jnp.stack((zeros, zeros), -1), jnp.stack((zeros, zeros), -1))
    temperature, rh, pressure, timestep = (jnp.broadcast_to(jnp.asarray(v, dtype=jnp.float64), shape)
                                          for v in (temperature, relative_humidity, pressure, timestep))
    wet = jnp.broadcast_to(wet, number.shape)
    sulfate = gas[..., 1]
    organic = jnp.where(organic_enabled, gas[..., 2], 0.0)
    critical = nucleation_rates(sulfate, temperature, rh, scheme=scheme, boundary_layer=boundary_layer,
                               activation_coefficient=activation_coefficient)
    active = (critical.rate >= 1e3) & (sulfate + organic >= 1.0)
    # Inactive cells skip the native growth code. Safe placeholders keep those
    # paths finite under vectorized selection without modifying active cells.
    diameter = jnp.where(active, critical.diameter, NUCLEATION_CRITICAL_DIAMETER)
    molecules = jnp.where(active, critical.sulfate_molecules, 2.0)
    limited_vapor_rate = jnp.minimum(critical.rate * molecules, sulfate / jnp.where(active, timestep, 1.0))
    rate = jnp.where(active, limited_vapor_rate / molecules, 0.0)
    # Native local growth concentration subtracts a rate without a dt factor;
    # retain this expression rather than introducing a scientific change.
    local_sulfate = jnp.maximum(1.0, sulfate - limited_vapor_rate)
    local_organic = jnp.maximum(1.0, organic)
    cluster_growth = 2.3623e-15 * jnp.sqrt(temperature) * (local_sulfate + local_organic)  # nm/h
    cluster_growth = jnp.where(active, cluster_growth, 1.0)
    molar_mass = jnp.broadcast_to(jnp.asarray(parameters.sulfate_molar_mass, dtype=jnp.float64), shape)
    density = jnp.broadcast_to(jnp.asarray(parameters.sulfate_density, dtype=jnp.float64), shape)
    diffusion = 5.1111e-10 * temperature**1.75 * REFERENCE_PRESSURE / pressure
    mean_free_path = 3.0 * diffusion * jnp.sqrt(PI * (molar_mass / 1000.0) / (8.0 * GAS_CONSTANT * temperature))
    if growth == 1:
        knudsen = 2.0 * mean_free_path[..., None] / (wet + SULFURIC_ACID_DIAMETER)
        accommodation = jnp.asarray(parameters.mass_accommodation, dtype=jnp.float64)
        beta = (knudsen + 1.0) / (0.377 * knudsen + 1.0 + 4.0 / (3.0 * accommodation) * (knudsen + knudsen**2))
        sink = jnp.sum(wet * beta * number, axis=-1)
        count = jnp.sum(number, axis=-1)
        mean_diameter = jnp.sum(number * wet, axis=-1) / jnp.where(count > 0, count, 1.0) * 1e9
        gamma = (0.23 * (diameter * 1e9)**0.2 * (mean_diameter / 150.0)**0.048 *
                 (temperature / 293.0)**(-0.75) * (density / 1000.0)**(-0.33))
        eta = jnp.where(sink < 1e-30, 0.0, jnp.minimum(gamma * sink / cluster_growth, diameter * 1e11))
        survival_exponent = eta / 3.0 - eta / (diameter * 1e9)
    else:
        sink, velocity = _cluster_sink(diameter, wet, number, temperature, density, mean_free_path)
        target = jnp.full(shape, NUCLEATION_TARGET_DIAMETER)
        target_sink, _ = _cluster_sink(target, wet, number, temperature, density, mean_free_path)
        exponent = jnp.log(target_sink / sink) / jnp.log(target / diameter)
        gamma = ((target / diameter)**(exponent + 1.0) - 1.0) / (exponent + 1.0)
        total_growth = cluster_growth * 1e-9 / (60.0**2)
        total_sink = sink
        if growth == 3:
            population = rate / sink
            for _ in range(5):
                total_sink = sink + 5e-16 * population * 1e-6
                total_growth = (cluster_growth * 1e-9 / 3600.0 +
                                1.5708e-6 * 6.0 * diameter**3 * (population * 1e-6) * velocity * AVOGADRO * 1e-9 / 3600.0)
                eta = -total_sink / ((exponent + 1.0) * total_growth * diameter**exponent)
                population = cluster_population_taylor(diameter, target, exponent, rate, eta, total_growth)
            total_sink = sink + 5e-16 * population * 1e-6
            total_growth = (cluster_growth * 1e-9 / 3600.0 +
                            1.5708e-6 * 6.0 * diameter**3 * (population * 1e-6) * velocity * AVOGADRO * 1e-9 / 3600.0)
        survival_exponent = -gamma * diameter * total_sink / total_growth
    formation = rate * jnp.exp(jnp.minimum(0.0, survival_exponent))
    molecular_rate = jnp.where(active, formation * NUCLEATION_CLUSTER_MOLECULES, 0.0)
    return NucleationSources(jnp.stack((molecular_rate, zeros), -1), jnp.stack((jnp.ones_like(zeros), zeros), -1))
