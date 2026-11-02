"""Opt-in repairs S4 (SALSA#14), S10 (SALSA#19) and D011 (SALSA#15) against Fortran.

The fixtures are the outputs of the standalone Fortran tests behind the upstream issues
(SALSA-JAX branch upstream-salsa-issue-tests, folders upstream/s4_kelvin_hydration,
upstream/s10_cond_particle_diffusion and upstream/s5_remap_orphan), built from TUT-ISI/SALSA
d87dc07 as committed and with each issue's patch. With every flag off the JAX port must match
the as-committed Fortran; with the flag on it must match the patched Fortran. The flags are off
by default, and turning one on must not change the processes the other two govern.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from salsa_jax.constants import PI
from salsa_jax.grid import DEFAULT_REGIME_LIMITS, size_grid
from salsa_jax.interface import volume_to_mass
from salsa_jax.model import FIX_OPTIONS, JIT_STATIC_OPTIONS, STATIC_OPTIONS, salsa_step
from salsa_jax.processes.condensation import condensation_step
from salsa_jax.processes.hydration import equilibrate
from salsa_jax.processes.remapping import remap

jax.config.update('jax_enable_x64', True)
FIXTURES = Path(__file__).parent / 'fixtures' / 'paper-fixes'
VARIANTS = (('as_committed', False), ('fixed', True))


def table(name):
    return np.genfromtxt(FIXTURES / name, delimiter=',', names=True)


@pytest.mark.parametrize('variant, flag', VARIANTS)
def test_s4_kelvin_hydration_matches_fortran(variant, flag):
    """One sulfate particle population in bin 1a1, 293.15 K, RH 50-95 %, dry 3 nm to 1 um."""
    rows = table(f's4-kelvin-{variant}.csv')
    grid = size_grid(DEFAULT_REGIME_LIMITS)
    count = len(rows)
    number = jnp.zeros((count, 17)).at[:, 0].set(1e9)
    dry = rows['dry_nm'] * 1e-9
    volume = jnp.zeros((count, 17, 5)).at[:, 0, 0].set(1e9 * PI / 6.0 * dry**3)
    state = jax.jit(equilibrate, static_argnames=('fix_kelvin_water',))(
        number, volume, rows['rh'], 293.15, grid.dpmid, fix_kelvin_water=flag)
    np.testing.assert_allclose(np.asarray(state.wet_diameter)[:, 0], rows['wet_nm'] * 1e-9, rtol=1e-12, atol=0)


def test_s4_fix_converges_to_water_kelvin_reference():
    """The repaired fixed point agrees with the bisection on the water Kelvin term (issue #14)."""
    rows = table('s4-kelvin-fixed.csv')
    assert np.max(np.abs(rows['wet_nm'] / rows['wet_water_kelvin_nm'] - 1)) < 2.5e-3
    native = table('s4-kelvin-as_committed.csv')
    assert np.min(native['wet_nm'] / native['wet_water_kelvin_nm'] - 1) < -0.08


@pytest.mark.parametrize('variant, flag', VARIANTS)
def test_s10_particle_diffusion_matches_fortran(variant, flag):
    """Collision rate per particle in bins 1a1-1a3, 293.15 K/1013 hPa and 220 K/50 hPa."""
    rows = table(f's10-cond-dp-{variant}.csv')
    grid = size_grid(DEFAULT_REGIME_LIMITS)
    count, c0 = len(rows), 1e13
    d = rows['diameter_nm'] * 1e-9
    index = rows['bin'].astype(int) - 1
    cases = np.arange(count)
    number = jnp.zeros((count, 17)).at[cases, index].set(1e9)
    volume = jnp.zeros((count, 17, 5)).at[cases, index, 0].set(1e9 * PI / 6.0 * d**3)
    wet = jnp.broadcast_to(grid.dpmid, (count, 17)).at[cases, index].set(d)
    gas = jnp.zeros((count, 3)).at[:, 1].set(c0)
    step = jax.jit(condensation_step, static_argnames=('fix_particle_diffusion',))
    state = step(number, volume, gas, wet, rows['temperature_K'], rows['pressure_Pa'], 1.0,
                 fix_particle_diffusion=flag)
    rate = (c0 / np.asarray(state.gas)[:, 1] - 1.0) / 1e9
    np.testing.assert_allclose(rate, rows['rate_per_particle_m3_s'], rtol=1e-11, atol=0)


def test_s10_fix_changes_only_regime_1a_small_bins():
    fixed, native = table('s10-cond-dp-fixed.csv'), table('s10-cond-dp-as_committed.csv')
    change = fixed['rate_per_particle_m3_s'] / native['rate_per_particle_m3_s'] - 1
    assert np.all(change[native['bin'] == 3] == 0)
    cold = (native['temperature_K'] == 220.0) & (native['diameter_nm'] < 3.0 + 1e-9)
    np.testing.assert_allclose(change[cold], 0.0719, atol=5e-4)


def remap_summary(number, volume, grid):
    """The diagnostics the Fortran test writes after each distr_update call."""
    n, v = np.asarray(number), np.asarray(volume).sum(axis=-1)
    threshold = 1.0  # nlim [#/m3]
    orphan = np.where(n <= threshold, v, 0.0).sum(axis=-1)
    active = (n > threshold) & ~np.isin(np.arange(17), (9, 16))
    over = np.where(active, v / np.where(active, n, 1.0) / np.asarray(grid.vhilim), 0.0).max(axis=-1)
    return orphan, over, np.argmax(n, axis=-1) + 1, n.sum(axis=-1), v.sum(axis=-1)


@pytest.mark.parametrize('variant, flag', VARIANTS)
def test_d011_remap_matches_fortran(variant, flag):
    """Monodisperse sulfate in bin 1a1 at 6 nm to 400 nm, two distr_update calls at 280 K."""
    rows = table(f's5-remap-{variant}.csv')
    grid = size_grid(DEFAULT_REGIME_LIMITS)
    first = rows['call'] == 1
    d = rows['diameter_nm'][first] * 1e-9
    count = len(d)
    number = jnp.zeros((count, 17)).at[:, 0].set(1e9)
    volume = jnp.zeros((count, 17, 5)).at[:, 0, 0].set(1e9 * PI / 6.0 * d**3)
    n0, v0 = 1e9, 1e9 * PI / 6.0 * d**3
    call = jax.jit(remap, static_argnames=('fix_remap_orphan',))
    for index in (1, 2):
        state = call(number, volume, 280.0, grid, fix_remap_orphan=flag)
        number, volume = state.number, state.volume
        expected = rows[rows['call'] == index]
        orphan, over, most, n_total, v_total = remap_summary(number, volume, grid)
        np.testing.assert_allclose(orphan / v0, expected["orphan_volume_fraction"], rtol=1e-9, atol=1e-13)
        np.testing.assert_allclose(over, expected['max_mean_over_upper'], rtol=1e-9, atol=0)
        np.testing.assert_array_equal(most, expected['bin_most_number'].astype(int))
        np.testing.assert_allclose(n_total / n0, expected['number_rel'], rtol=1e-14)
        np.testing.assert_allclose(v_total / v0, expected['volume_rel'], rtol=1e-14)
    if flag:
        assert np.max(orphan / v0) < 1e-12 and np.max(over) <= 1.0
    else:
        assert np.max(orphan / v0) > 0.1


def full_step_inputs():
    from salsa_jax.interface import validate_component_map
    component_map = np.full((17, 5), -1, dtype=int)
    index = 0
    for b in range(17):                       # the native represented slots, in order
        for s in range(5):
            if (b >= 3 or s < 2) and (b < 10 or s != 3):
                component_map[b, s] = index
                index += 1
    validate_component_map(component_map)
    grid = size_grid(DEFAULT_REGIME_LIMITS)
    seed = jnp.array([3e7, 5e7, 2e7])
    number = jnp.zeros((1, 17)).at[0, :3].set(seed)
    volume = jnp.zeros((1, 17, 5)).at[0, :3, 0].set(seed * PI / 6.0 * grid.dpmid[:3]**3)
    gas = jnp.zeros((1, 3)).at[0, 1].set(1e14)
    return volume_to_mass(volume, component_map), number, gas, component_map


def test_flags_are_static_and_default_off():
    """All flags off is the default call, bit for bit; S4 and S10 act in a full step."""
    assert set(JIT_STATIC_OPTIONS) == set(STATIC_OPTIONS) | set(FIX_OPTIONS)
    mass, number, gas, component_map = full_step_inputs()
    step = jax.jit(salsa_step, static_argnames=JIT_STATIC_OPTIONS)
    args = (mass, number, gas, 230.0, 0.5, 5000.0, 60.0, component_map)
    common = dict(scheme=0, coagulation_enabled=True, condensation_enabled=True)
    default = step(*args, **common)
    off = step(*args, **common, **{name: False for name in FIX_OPTIONS})
    for a, b in zip(default, off):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    kelvin = step(*args, **common, fix_kelvin_water=True)
    diffusion = step(*args, **common, fix_particle_diffusion=True)
    every = step(*args, **common, **{name: True for name in FIX_OPTIONS})
    for state in (kelvin, diffusion, every):
        assert bool(np.all(np.asarray(state.valid)))
    assert np.all(np.asarray(kelvin.wet_diameter)[0, :3] > np.asarray(default.wet_diameter)[0, :3])
    assert not np.array_equal(np.asarray(diffusion.gas), np.asarray(default.gas))


def test_coagulation_density_option():
    """Kernel particle-mass density: None and 1500 are the native kernel; 1770 rescales mass only."""
    from salsa_jax.constants import COAGULATION_DENSITY
    from salsa_jax.processes.coagulation import coagulation_coefficient
    from salsa_jax.processes.coagulation_step import coefficient_matrix
    grid = size_grid(DEFAULT_REGIME_LIMITS)
    dry = jnp.asarray(grid.dpmid)
    wet = dry * 1.06
    native = np.asarray(coefficient_matrix(dry, wet, 220.0, 5000.0))
    explicit = np.asarray(jax.jit(coefficient_matrix)(dry, wet, 220.0, 5000.0, density=COAGULATION_DENSITY))
    np.testing.assert_allclose(explicit, native, rtol=1e-13, atol=0)  # jit vs eager: ULP
    dense = np.asarray(coefficient_matrix(dry, wet, 220.0, 5000.0, density=1770.0))
    mass = PI / 6.0 * dry**3 * 1770.0
    direct = np.asarray(coagulation_coefficient(dry[:, None], dry[None, :], mass[:, None], mass[None, :], 220.0, 5000.0))
    upper = np.triu(np.ones((17, 17), dtype=bool))
    scale = np.asarray(dry[:, None] * wet[None, :] / (dry[None, :] * wet[:, None]))
    np.testing.assert_allclose(dense[upper], (direct * scale)[upper], rtol=1e-14)
    assert np.all(dense[:3, :3] < native[:3, :3])     # heavier particles move more slowly
