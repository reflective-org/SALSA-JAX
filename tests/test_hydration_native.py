"""Native initial hydration and composed pre-remapping microphysics acceptance."""
from pathlib import Path

import jax
import numpy as np
import pytest

from salsa_jax.grid import size_grid
from salsa_jax.processes.hydration import HydrationParameters, equilibrate, solubility_mask
from salsa_jax.processes.coagulation_step import coagulation_step
from salsa_jax.processes.condensation import CondensationParameters, condensation_step
from tools.hydration_fixture import load_native_case, record
from tools.hydration_oracle import build_oracle, run_oracle
from tools.coagulation_capture import load_fixture as load_coagulation_fixture
from tools.condensation_capture import load_fixture as load_condensation_fixture

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[1]
INPUTS = ('number', 'volume', 'relative_humidity', 'temperature', 'dry_diameter')


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    cases, expected, metadata = load_native_case()
    executable, provenance = build_oracle(tmp_path_factory.mktemp('hydration-native'))
    oracle, _ = run_oracle(executable, cases)
    return cases, expected, metadata, oracle, provenance


def test_actual_initial_boundary_matches_extracted_fortran(native):
    _, expected, _, oracle, provenance = native
    assert provenance['checked']
    for field in ('core', 'wet_diameter'):
        np.testing.assert_allclose(oracle[field], expected[field], rtol=3e-13, atol=0)
    assert np.all(oracle['iterations'] > 0)


@pytest.mark.parametrize('compiled', [False, True])
def test_actual_initial_hydration_matches_jax(native, compiled):
    cases, expected, metadata, oracle, _ = native
    kernel = jax.jit(equilibrate) if compiled else equilibrate
    actual = kernel(*(cases[key] for key in INPUTS), soluble_mask=cases['soluble_mask'],
                    parameters=HydrationParameters(**{name: cases[name] for name in HydrationParameters._fields}))
    for field in ('core', 'wet_diameter'):
        values = np.asarray(getattr(actual, field))
        assert np.all(np.isfinite(values))
        assert np.all(values > 0)
        np.testing.assert_allclose(values, expected[field], rtol=3e-13, atol=0)
    np.testing.assert_array_equal(actual.iterations, oracle['iterations'])
    np.testing.assert_array_equal(cases['soluble_mask'],
                                  np.broadcast_to(solubility_mask(metadata['hydrate_insoluble']), cases['soluble_mask'].shape))


@pytest.mark.parametrize('compiled', [False, True])
def test_grid_hydration_coagulation_condensation_chain_matches_native_boundaries(native, compiled):
    cases, expected, metadata, _, _ = native
    coagulation = load_coagulation_fixture(ROOT / 'tests/fixtures/coagulation-default')
    condensation = load_condensation_fixture(ROOT / 'tests/fixtures/condensation-default')
    count = len(cases['number'])
    def coag(stage, name):
        value = record(coagulation, stage, name)
        return value.reshape((count,) + value.shape[2:]) if value.ndim >= 2 else value
    def cond(stage, name):
        value = record(condensation, stage, name)
        return value.reshape((count,) + value.shape[2:]) if value.ndim >= 2 else value
    # Hydration changes neither number nor material volume. At recalc=False,
    # condensation consumes the initial hydrated diameter after coagulation.
    assert not metadata['recalculate']
    np.testing.assert_array_equal(cases['number'], coag('entry', 'pnaero'))
    np.testing.assert_array_equal(cases['volume'], coag('entry', 'zvols'))
    np.testing.assert_array_equal(cases['temperature'], coag('entry', 'ptemp'))
    np.testing.assert_array_equal(cases['relative_humidity'], cond('entry', 'prh'))
    np.testing.assert_array_equal(expected['wet_diameter'], coag('entry', 'zdwet'))
    np.testing.assert_array_equal(expected['wet_diameter'], cond('entry', 'zdwet'))
    np.testing.assert_array_equal(coag('exit', 'pnaero'), cond('entry', 'pnaero'))
    np.testing.assert_array_equal(coag('exit', 'zvols'), cond('entry', 'zvols'))
    np.testing.assert_array_equal(cond('entry', 'options'), [0, 0, 64])
    np.testing.assert_array_equal(cond('entry', 'gas_indices'), [2, 0])
    gas = np.zeros((count, 3))
    gas[:, :2] = cond('entry', 'pc_gas')
    hydration_parameters = HydrationParameters(**{name: cases[name] for name in HydrationParameters._fields})
    condensation_parameters = CondensationParameters(
        sulfate_molar_mass=cases['species_molar_mass'][:, 0], organic_molar_mass=cases['species_molar_mass'][:, 1],
        sulfate_density=cases['species_density'][:, 0], organic_density=cases['species_density'][:, 1],
        mass_accommodation=cond('entry', 'massacc'))
    np.testing.assert_array_equal(condensation_parameters.sulfate_molar_mass, np.full(count, cond('entry', 'species_molar_mass')[0]))
    np.testing.assert_array_equal(condensation_parameters.organic_molar_mass, np.full(count, cond('entry', 'species_molar_mass')[1]))

    def advance(number, volume, rh, temperature, pressure, timestep, gas, limits, hydrate_insoluble, hydration_parameters, condensation_parameters):
        # Both geometry and wet diameter are computed by the ported kernels;
        # captured midpoint/wet arrays appear only in assertions below.
        grid = size_grid(limits)
        hydrated = equilibrate(number, volume, rh, temperature, grid.dpmid,
                               soluble_mask=solubility_mask(hydrate_insoluble), parameters=hydration_parameters)
        coagulated = coagulation_step(number, volume, hydrated.wet_diameter, grid.dpmid, temperature, pressure, timestep)
        condensed = condensation_step(coagulated.number, coagulated.volume, gas, hydrated.wet_diameter,
                                      temperature, pressure, timestep, organic_enabled=False, parameters=condensation_parameters)
        return grid, hydrated, coagulated, condensed

    kernel = jax.jit(advance) if compiled else advance
    grid, hydrated, coagulated, condensed = kernel(
        cases['number'], cases['volume'], cases['relative_humidity'], cases['temperature'],
        coag('entry', 'ppres'), coag('entry', 'ptstep'), gas,
        metadata['regime_limits'], metadata['hydrate_insoluble'], hydration_parameters, condensation_parameters)
    np.testing.assert_allclose(grid.dpmid, coag('entry', 'dpmid'), rtol=3e-14, atol=0)
    for field in ('core', 'wet_diameter'):
        np.testing.assert_allclose(getattr(hydrated, field), expected[field], rtol=3e-13, atol=0)
    for field, source in (('number', 'pnaero'), ('volume', 'zvols')):
        np.testing.assert_allclose(getattr(coagulated, field), coag('exit', source), rtol=3e-13, atol=0)
        np.testing.assert_allclose(getattr(condensed, field), cond('exit', source), rtol=3e-13, atol=0)
    np.testing.assert_allclose(np.asarray(condensed.gas)[:, :2], cond('exit', 'pc_gas'), rtol=3e-13, atol=0)
    np.testing.assert_array_equal(np.asarray(condensed.gas)[:, 2], 0)


@pytest.mark.parametrize('field', ['species_indices', 'soluble', 'ak'])
def test_adapter_rejects_incompatible_species_flags_or_constants(field):
    from copy import deepcopy
    from tools.hydration_capture import load_fixture
    from tools.hydration_fixture import adapt_capture
    capture = deepcopy(load_fixture(ROOT / 'tests/fixtures/hydration-default'))
    values = capture['entry'][field]['values']
    if field == 'species_indices':
        values[0], values[1] = values[1], values[0]
        message = 'species mapping'
    elif field == 'soluble':
        values[-1] = 1 - values[-1]
        message = 'Soluble flags'
    else:
        values[0] *= 1.01
        message = 'fixed constant'
    with pytest.raises(ValueError, match=message):
        adapt_capture(capture)
