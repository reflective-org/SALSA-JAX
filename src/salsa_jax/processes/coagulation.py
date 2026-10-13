# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Brownian collision coefficient from mo_ham_salsa_init.F90:coagc."""

import jax.numpy as jnp

from salsa_jax.constants import BOLTZMANN, PI, REFERENCE_PRESSURE


def coagulation_coefficient(diameter1, diameter2, mass1, mass2, temperature, pressure):
    """Return collision coefficient [m³/s] for positive SI inputs.

    Diameters are in m, masses in kg, temperature in K and pressure in Pa.
    Inputs broadcast; scalar inputs return a scalar array. Callers must enable
    ``jax_enable_x64`` for Fortran parity. This is only the collision kernel,
    not SALSA's bin transfer or complete coagulation time step.
    """
    diameter1, diameter2, mass1, mass2, temperature, pressure = jnp.broadcast_arrays(
        *[jnp.asarray(value, dtype=jnp.float64) for value in
          (diameter1, diameter2, mass1, mass2, temperature, pressure)]
    )
    diameter = jnp.stack((diameter1, diameter2))
    mass = jnp.stack((mass1, mass2))
    viscosity = 7.44523e-3 * temperature**1.5 / (5093.0 * (temperature + 110.4))
    mean_free_path = (1.656e-10 * temperature + 1.828e-8) * REFERENCE_PRESSURE / pressure
    knudsen = 2.0 * mean_free_path / diameter
    correction = 1.0 + knudsen * (1.142 + 0.558 * jnp.exp(-0.999 / knudsen))
    diffusion = correction * BOLTZMANN * temperature / (3.0 * PI * viscosity * diameter)
    velocity = jnp.sqrt(8.0 * BOLTZMANN * temperature / (PI * mass))
    omega = 8.0 * diffusion / (PI * velocity)
    mean_diameter = 0.5 * (diameter1 + diameter2)
    continuum_flux = 4.0 * PI * mean_diameter * (diffusion[0] + diffusion[1])
    molecular_flux = PI * jnp.sqrt(velocity[0]**2 + velocity[1]**2) * mean_diameter**2
    distance = (
        ((mean_diameter + omega)**3
         - (mean_diameter**2 + omega**2) * jnp.sqrt(mean_diameter**2 + omega**2))
        / (3.0 * mean_diameter * omega) - mean_diameter
    )
    matching_distance = jnp.sqrt(distance[0]**2 + distance[1]**2)
    return continuum_flux / (
        mean_diameter / (mean_diameter + matching_distance) + continuum_flux / molecular_flux
    )
