# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Complete standalone SALSA coagulation on the 17-bin, five-material grid.

Port of mo_ham_salsa_dynamics.F90:coagulation without HAMMOZ/VBS. Hydration
and remapping are separate processes; callers supply the wet diameters used
by Fortran. All leading axes represent independent columns/levels/cases.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salsa_jax.constants import PI, COAGULATION_DENSITY
from salsa_jax.processes.coagulation import coagulation_coefficient

# Zero-based boundaries of the supported native grid: 1a, 2a, 2b.
FIRST_2A = 3
FIRST_2B = 10
BIN_COUNT = 17
MATERIAL_COUNT = 5


class CoagulationState(NamedTuple):
    """Number [particles/m3] and material volume [m3 material/m3 air]."""

    number: jax.Array
    volume: jax.Array


def coefficient_matrix(dry_diameter, wet_diameter, temperature, pressure):
    """Return (...,17,17) coefficients [m3/s], using the native upper triangle.

    Diameters are in m (the legacy ``pdwet`` radius comment is incorrect).
    Temperature is K and pressure Pa. Inputs must be positive and finite;
    callers must enable JAX float64. Dry geometry can be shared across cases.
    The wet correction uses *index* order, including the duplicated 2b grid;
    independently computing the lower triangle would change the algorithm.
    """
    dry = jnp.asarray(dry_diameter, dtype=jnp.float64)
    wet = jnp.asarray(wet_diameter, dtype=jnp.float64)
    if dry.shape[-1:] != (BIN_COUNT,) or wet.shape[-1:] != (BIN_COUNT,):
        raise ValueError("Coagulation requires the 17-bin SALSA grid")
    temperature = jnp.asarray(temperature, dtype=jnp.float64)[..., None, None]
    pressure = jnp.asarray(pressure, dtype=jnp.float64)[..., None, None]
    mass = PI / 6.0 * dry**3 * COAGULATION_DENSITY
    coefficients = coagulation_coefficient(
        dry[..., :, None], dry[..., None, :],
        mass[..., :, None], mass[..., None, :], temperature, pressure,
    ) * dry[..., :, None] * wet[..., None, :] / (dry[..., None, :] * wet[..., :, None])
    upper = jnp.arange(BIN_COUNT)[:, None] <= jnp.arange(BIN_COUNT)[None, :]
    return jnp.where(upper, coefficients, jnp.swapaxes(coefficients, -1, -2))


def _update_column(number, volume, coefficients, timestep):
    bins = jnp.arange(BIN_COUNT)

    def update_bin(k, state):
        number, volume = state
        regime1 = k < FIRST_2A
        regime2a = (k >= FIRST_2A) & (k < FIRST_2B)
        paired_bin = jnp.where(regime2a, k + FIRST_2B - FIRST_2A,
                              k - FIRST_2B + FIRST_2A)
        first_loss = (bins > k) & jnp.where(regime2a, bins < FIRST_2B, True)
        second_loss = jnp.where(
            regime2a, bins > paired_bin,
            (~regime1) & (bins >= paired_bin) & (bins < FIRST_2B),
        )
        gains = jnp.where(
            regime1, bins < k,
            jnp.where(regime2a,
                      (bins < k) | ((bins >= FIRST_2B) & (bins <= paired_bin)),
                      (bins < paired_bin) | ((bins >= FIRST_2B) & (bins < k))),
        )
        # Bin updates are sequential: later bins see already updated smaller
        # bins, including 2a before 2b. Reduction roundoff may differ by backend.
        loss_terms = coefficients[k] * number
        loss = (jnp.sum(jnp.where(first_loss, loss_terms, 0.0)) +
                jnp.sum(jnp.where(second_loss, loss_terms, 0.0)))
        gain = jnp.sum(jnp.where(gains[:, None], coefficients[:, k, None] * volume, 0.0), axis=0)
        new_volume = (volume[k] + timestep * gain * number[k]) / (1.0 + timestep * loss)
        new_number = number[k] / (1.0 + timestep * loss + 0.5 * timestep * coefficients[k, k] * number[k])
        return number.at[k].set(new_number), volume.at[k].set(new_volume)

    number, volume = jax.lax.fori_loop(0, BIN_COUNT, update_bin, (number, volume))
    # Legacy approximation: salt arriving in insoluble bins becomes sulfate
    # by volume. Retain the final native nonnegative-volume clamp as well.
    volume = volume.at[FIRST_2B:, 0].add(volume[FIRST_2B:, 3])
    volume = volume.at[FIRST_2B:, 3].set(0.0)
    return CoagulationState(number, jnp.maximum(volume, 0.0))


def coagulation_update(number, volume, coefficients, timestep):
    """Sequential bin transfer using supplied coefficients; supports leading axes.

    Number ends in (17,), volume in (17,5), with material order SO4/OC/BC/SS/DU.
    Leading state axes must match; coefficients and timestep may broadcast to
    them. The dry-core-volume argument in Fortran is unused and is omitted.
    """
    number = jnp.asarray(number, dtype=jnp.float64)
    volume = jnp.asarray(volume, dtype=jnp.float64)
    if number.shape[-1:] != (BIN_COUNT,) or volume.shape != number.shape + (MATERIAL_COUNT,):
        raise ValueError("Expected number (...,17) and volume (...,17,5) with matching leading axes")
    shape = number.shape[:-1]
    coefficients = jnp.broadcast_to(jnp.asarray(coefficients, dtype=jnp.float64), shape + (BIN_COUNT, BIN_COUNT))
    timestep = jnp.broadcast_to(jnp.asarray(timestep, dtype=jnp.float64), shape)
    updated = jax.vmap(_update_column)(number.reshape(-1, BIN_COUNT),
                                     volume.reshape(-1, BIN_COUNT, MATERIAL_COUNT),
                                     coefficients.reshape(-1, BIN_COUNT, BIN_COUNT), timestep.reshape(-1))
    return CoagulationState(updated.number.reshape(number.shape), updated.volume.reshape(volume.shape))


def coagulation_step(number, volume, wet_diameter, dry_diameter, temperature, pressure, timestep, *, enabled=True):
    """Run complete standalone coagulation, or preserve inputs when disabled.

    Enable ``jax_enable_x64`` before use. Inputs must be physical: nonnegative
    number/volume and timestep, positive diameters, temperature and pressure.
    ``enabled`` may be a scalar JAX boolean. Disabling skips the native process,
    including salt relabeling and clipping; enabled dt=0 still applies those.
    Hydration is an input dependency, not computed here. No inputs are mutated.
    """
    number = jnp.asarray(number, dtype=jnp.float64)
    volume = jnp.asarray(volume, dtype=jnp.float64)
    return jax.lax.cond(
        enabled,
        lambda: coagulation_update(number, volume,
                                   coefficient_matrix(dry_diameter, wet_diameter, temperature, pressure), timestep),
        lambda: CoagulationState(number, volume),
    )
