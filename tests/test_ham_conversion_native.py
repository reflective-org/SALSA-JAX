"""Actual HAM boundaries and number→material→gas initialization composition."""
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from salsa_jax.aerosol_initialization import initialize_aerosol
from salsa_jax.gas_conversion import gas_to_tracers
from salsa_jax.initialization import size_distribution
from tools.aerosol_initialization_oracle import build_oracle as build_aerosol, run_oracle as run_aerosol
from tools.gas_conversion_oracle import build_oracle as build_gas, run_oracle as run_gas
from tools.ham_conversion_fixture import GAS_INPUTS, aerosol_arguments, load_native_cases
from tools.initialization_fixture import load_native_case as load_number

jax.config.update('jax_enable_x64', True)
FIXTURES = Path(__file__).resolve().parent/'fixtures'


@pytest.fixture(scope='module')
def references(tmp_path_factory):
    aerosol, _ = build_aerosol(tmp_path_factory.mktemp('native-conc2mmr'))
    gas, _ = build_gas(tmp_path_factory.mktemp('native-gas2mmr'))
    return aerosol, gas


@pytest.mark.parametrize('profile', ['default', 'active'])
def test_original_references_replay_captured_boundaries(references, profile):
    c, expected, g, gas_expected, _ = load_native_cases(FIXTURES/f'ham-conversion-{profile}')
    actual, _ = run_aerosol(references[0], c)
    for name in ('number', 'mass', 'tracers'):
        np.testing.assert_allclose(actual[name], expected[name], rtol=3e-13, atol=0)
    actual_gas = run_gas(references[1], g)
    np.testing.assert_allclose(actual_gas['tracers'], gas_expected, rtol=3e-15, atol=0)


@pytest.mark.parametrize('profile', ['default', 'active'])
@pytest.mark.parametrize('compiled', [False, True])
def test_native_number_material_and_gas_chain(profile, compiled):
    directory = FIXTURES/f'ham-conversion-{profile}'
    c, expected, g, gas_expected, metadata = load_native_cases(directory)
    number_args, number_expected, _ = load_number(directory/'number-initialization')
    number_manifest = json.loads((directory/'number-initialization/manifest.json').read_text())
    assert number_manifest['run_manifest_sha256'] == metadata['run_manifest_sha256']
    assert number_manifest['input_sha256'] == metadata['input_sha256']
    assert number_manifest['source_sha256'] == metadata['source_sha256']
    args, kwargs = aerosol_arguments(c)
    def initialize(number_args, core, tracers, density, mapping, component_indices,
                   number_indices, codes, parameters, mw_so4, gas_args):
        n = size_distribution(*number_args)
        n = jax.numpy.broadcast_to(n, c['number'].shape)
        aerosol = initialize_aerosol(n, core, tracers, density, mapping, component_indices,
                                     number_indices, codes, parameters=parameters, mw_so4=mw_so4)
        final = gas_to_tracers(aerosol.tracers, *gas_args)
        return n, aerosol, final
    kernel = jax.jit(initialize) if compiled else initialize
    number, aerosol, final = kernel(number_args, *args[1:], kwargs['parameters'], kwargs['mw_so4'],
                                    tuple(g[key] for key in GAS_INPUTS[1:]))
    np.testing.assert_allclose(number, number_expected.reshape(c['number'].shape), rtol=3e-13, atol=0)
    np.testing.assert_array_equal(expected['tracers'], g['tracers'])
    for name in ('number', 'mass', 'tracers'):
        np.testing.assert_allclose(getattr(aerosol, name), expected[name], rtol=3e-13, atol=0)
    np.testing.assert_allclose(final, gas_expected, rtol=3e-13, atol=0)
    # Pure gas leaf at its exact native input has its tighter independent bound.
    gas_kernel = jax.jit(gas_to_tracers) if compiled else gas_to_tracers
    np.testing.assert_allclose(gas_kernel(*(g[key] for key in GAS_INPUTS)), gas_expected, rtol=3e-15, atol=0)


def test_actual_profiles_preserve_initialization_and_change_gas_forcing():
    default = load_native_cases(FIXTURES/'ham-conversion-default')
    active = load_native_cases(FIXTURES/'ham-conversion-active')
    for field in ('number', 'mass', 'tracers'):
        np.testing.assert_array_equal(default[1][field], active[1][field])
    forcing = active[4]['reference_configuration']['settings']['sulfate_molecules_m3']
    assert active[2]['gas'][0, 1] == forcing*1e-6
    assert active[2]['gas'][0, 1] != default[2]['gas'][0, 1]
    assert active[4]['reference_configuration']


@pytest.mark.parametrize('change', ['source', 'unchecked'])
def test_adapter_rejects_incompatible_reference_provenance(tmp_path, change):
    import shutil
    directory = tmp_path/'fixture'
    shutil.copytree(FIXTURES/'ham-conversion-default', directory)
    path = directory/'manifest.json'
    manifest = json.loads(path.read_text())
    if change == 'source':
        manifest['source_sha256']['mo_convert_concentrations.F90'] = 'mismatched-source'
    else:
        manifest['config']['build']['flags'].remove('-fcheck=all')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='source|Checked'):
        load_native_cases(directory)
