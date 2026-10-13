# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Ordered native SALSA bin redistribution; fixed 17-bin/five-material layout."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from salsa_jax.constants import (
    PI, GAS_CONSTANT, SULFATE_MOLAR_MASS, SULFATE_DENSITY,
    WATER_MOLAR_MASS, WATER_DENSITY, WATER_SURFACE_TENSION,
    ORGANIC_SOLUBLE_FRACTION, HYDRATION_NUMBER_THRESHOLD,
    REMAPPING_SATURATION_LIMIT, REMAPPING_ION_FACTOR,
)
from salsa_jax.grid import SizeGrid


class RemappingParameters(NamedTuple):
    """Native properties; scalar fields broadcast over leading case axes."""
    sulfate_molar_mass: float = SULFATE_MOLAR_MASS  # g/mol
    sulfate_density: float = SULFATE_DENSITY  # kg/m3
    water_molar_mass: float = WATER_MOLAR_MASS  # g/mol
    water_density: float = WATER_DENSITY  # kg/m3
    surface_tension: float = WATER_SURFACE_TENSION  # J/m2
    organic_soluble_fraction: float = ORGANIC_SOLUBLE_FRACTION
    saturation_limit: float = REMAPPING_SATURATION_LIMIT
    ion_factor: float = REMAPPING_ION_FACTOR
    number_threshold: float = HYDRATION_NUMBER_THRESHOLD  # particles/m3


class RemappingState(NamedTuple):
    number: jax.Array
    volume: jax.Array
    iterations: jax.Array


def _material_sum(volume):
    total = volume[..., 0]
    for species in range(1, 5):
        total = total + volume[..., species]
    return total


def remap(number, volume, temperature, grid: SizeGrid, *, transfer_soluble=False,
          parameters=RemappingParameters()):
    """Apply one native distr_update call, including optional 2b→2a transfer.

    Number (...,17) is particles/m3; volume (...,17,5) is material m3/air m3
    in SO4/OC/BC/SS/DU order. Temperature is K; grid uses size_grid's units.
    Leading state axes match. Geometry, properties and transfer_soluble broadcast
    across cases. Enable float64. Outputs are new arrays; iterations counts the
    native descending scans per cell, including the first scan of empty cells.

    Native scans 16→1, skips bin 10, and does not revisit all receiving bins.
    Two calls and surrounding cleanup/hydration belong to the SALSA caller.
    Large transfers can retain orphan material at zero number; no conservation
    or empty-bin repair is added. Positive finite ordered geometry is required;
    active soluble-transfer bins require positive total material volume; a cell
    violating that native domain returns NaN number and volume.
    """
    number, volume = (jnp.asarray(v, dtype=jnp.float64) for v in (number, volume))
    if number.shape[-1:] != (17,) or volume.shape != number.shape + (5,):
        raise ValueError('Expected number (...,17) and volume (...,17,5)')
    leading = number.shape[:-1]
    def case(value):
        return jnp.broadcast_to(jnp.asarray(value, dtype=jnp.float64), leading)
    low, high, midpoint = (jnp.broadcast_to(jnp.asarray(value, dtype=jnp.float64), number.shape)
                           for value in (grid.vlolim, grid.vhilim, grid.dpmid))
    threshold = case(parameters.number_threshold)
    temperature = case(temperature)
    counts = jnp.zeros(leading, dtype=jnp.int32)
    pending = jnp.ones(leading, dtype=bool)

    def scan(state):
        number, volume, pending, counts = state
        def bin_update(index, carry):
            number, volume, again = carry
            kk = 15 - index
            n = number[..., kk]
            v = volume[..., kk, :]
            total = _material_sum(v)
            eligible = pending & (n > threshold) & (kk != 9)
            mean = total / jnp.where(eligible, n, 1.0)
            skip_lowest = (kk == 0) & (low[..., kk] > mean)
            down = eligible & ~skip_lowest & (mean < low[..., kk])
            target = jnp.where(kk == 10, 2, jnp.maximum(kk - 1, 0))
            moved_n = jnp.where(down, n, 0.0)
            moved_v = jnp.where(down[..., None], v, 0.0)
            number = number.at[..., target].add(moved_n)
            volume = volume.at[..., target, :].add(moved_v)
            number = number.at[..., kk].set(jnp.where(down, 0.0, n))
            volume = volume.at[..., kk, :].set(jnp.where(down[..., None], 0.0, v))
            # The source retains its pre-transfer mean and CYCLEs below midvolume.
            grow = eligible & ~skip_lowest & (mean > (PI / 6.0) * midpoint[..., kk]**3)
            ratio = high[..., kk] / low[..., kk]
            vlo = 2.0 * mean / (1.0 + ratio)
            vhi = ratio * vlo
            excess = 0.5 * (vhi + high[..., kk])
            fraction = jnp.minimum(1.0, (vhi - high[..., kk]) / jnp.where(grow, vhi - vlo, 1.0))
            n, v = number[..., kk], volume[..., kk, :]
            total = _material_sum(v)
            transfer = fraction[..., None] * n[..., None] * excess[..., None] * v / jnp.where(grow, total, 1.0)[..., None]
            transfer = jnp.where(grow[..., None], transfer, 0.0)
            volume = volume.at[..., kk + 1, :].add(transfer)
            volume = volume.at[..., kk, :].add(-transfer)
            moved_n = jnp.where(grow, fraction * n, 0.0)
            number = number.at[..., kk + 1].add(moved_n)
            number = number.at[..., kk].set(jnp.where(grow, n * (1.0 - fraction), n))
            n = number[..., kk]
            # CYCLE after nongrowth also skips this native repeat test.
            check = grow & (n > threshold)
            remaining = _material_sum(volume[..., kk, :]) / jnp.where(check, n, 1.0)
            again = again | (check & (remaining > high[..., kk]))
            return number, volume, again
        number, volume, again = jax.lax.fori_loop(0, 16, bin_update, (number, volume, jnp.zeros_like(pending)))
        return number, volume, again, counts + pending.astype(jnp.int32)

    number, volume, _, counts = jax.lax.while_loop(lambda s: jnp.any(s[2]), scan, (number, volume, pending, counts))
    enabled = jnp.broadcast_to(jnp.asarray(transfer_soluble, dtype=bool), leading)
    # The checked native soluble-fraction division traps for active zero-volume
    # bins. Signal that invalid cell rather than silently treating NaN as false.
    invalid = enabled & jnp.any((number[..., 10:] > threshold[..., None]) &
                                (_material_sum(volume[..., 10:, :]) <= 0), axis=-1)
    water_mass, water_density = case(parameters.water_molar_mass), case(parameters.water_density)
    sulfate_mass, sulfate_density = case(parameters.sulfate_molar_mass), case(parameters.sulfate_density)
    # XLA's literal-division reciprocal rewrite shifts the strict transfer
    # threshold by one ULP. An array-shaped denominator barrier retains the
    # native divisions for scalar and batched cases (no tolerance at the branch).
    thousand = jax.lax.optimization_barrier(jnp.full_like(water_mass, 1000.0))
    aa = 4.0 * water_mass / thousand * case(parameters.surface_tension) / (GAS_CONSTANT * water_density)
    bb = 6.0 * water_mass / thousand * case(parameters.ion_factor) * sulfate_density / (PI * water_density * sulfate_mass / thousand)
    def soluble_transfer(index, state):
        number, volume = state
        kk, target = 10 + index, 3 + index
        n, v = number[..., kk], volume[..., kk, :]
        active = enabled & (n > threshold)
        soluble_fraction = (v[..., 0] + case(parameters.organic_soluble_fraction) * v[..., 1]) / jnp.where(active, _material_sum(v), 1.0)
        critical = 4.0 * aa**3 / (27.0 * bb * jnp.log(case(parameters.saturation_limit))**2 * ((PI / 6.0) * midpoint[..., kk]**3))
        move = active & (critical / temperature**3 < soluble_fraction)
        number = number.at[..., target].add(jnp.where(move, n, 0.0))
        volume = volume.at[..., target, :].add(jnp.where(move[..., None], v, 0.0))
        number = number.at[..., kk].set(jnp.where(move, 0.0, n))
        volume = volume.at[..., kk, :].set(jnp.where(move[..., None], 0.0, v))
        return number, volume
    number, volume = jax.lax.fori_loop(0, 7, soluble_transfer, (number, volume))
    number = jnp.where(invalid[..., None], jnp.nan, number)
    volume = jnp.where(invalid[..., None, None], jnp.nan, volume)
    return RemappingState(number, volume, counts)
