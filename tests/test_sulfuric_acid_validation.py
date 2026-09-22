"""Independent report gates and reproducible user-facing trajectory artifacts."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from salsa_jax import constants
from salsa_jax.sulfuric_acid import initial_state, load_config, sulfur_inventory
from tools import run_sulfuric_acid as cli
from tools.salsa_oracle import OUTPUTS
from tools.sulfuric_acid_validation import compare, reference_case, summarize

jax.config.update('jax_enable_x64', True)
CONFIG = Path(__file__).resolve().parents[1] / 'configs/sulfuric-acid.toml'


def copies(values):
    return {key: np.array(value, copy=True) for key, value in values.items()}


def fake_outputs(steps=3):
    return {key: np.ones((steps, 1, width)) for key, width in OUTPUTS.items()}


@pytest.mark.parametrize('field', list(OUTPUTS))
def test_each_field_discrepancy_fails_every_step_comparison(field):
    expected = fake_outputs()
    actual = copies(expected)
    actual[field][1, 0, -1] *= 1.01
    result = compare(actual, expected)
    assert not result['passed']
    assert not result['fields'][field]['passed']
    assert result['fields'][field]['largest_absolute_error_index'] == [1, 0, OUTPUTS[field]-1]


@pytest.mark.parametrize('field', [key for key in OUTPUTS if key != 'water_mass'])
def test_native_zero_requires_exact_preservation(field):
    expected = fake_outputs()
    expected[field][1, 0, 0] = 0.
    actual = copies(expected)
    actual[field][1, 0, 0] = 1e-30
    result = compare(actual, expected)
    assert not result['passed']
    assert result['fields'][field]['nonzero_at_zero_reference'] == 1


@pytest.mark.parametrize('change', ['shape', 'actual_nonfinite', 'reference_nonfinite'])
def test_shape_and_nonfinite_outputs_are_rejected(change):
    expected = fake_outputs()
    actual = copies(expected)
    if change == 'shape':
        actual['number'] = actual['number'][:-1]
    elif change == 'actual_nonfinite':
        actual['gas'][0, 0, 1] = np.inf
    else:
        expected['mass'][1, 0, 0] = np.nan
    with pytest.raises(ValueError, match='shape/nonfinite'):
        compare(actual, expected)


def test_nonzero_signed_water_uses_only_documented_cancellation_allowance():
    expected = fake_outputs()
    expected['dry_radius_cm'][:] = 1e-5
    expected['number'][:] = 1e8
    scale = ((constants.PI/6) * (.02 * expected['dry_radius_cm'])**3
             * constants.WATER_DENSITY * expected['number'])
    allowance = 32 * np.finfo(float).eps * scale
    expected['water_mass'][:] = -1e-30
    actual = copies(expected)
    actual['water_mass'] += allowance / 2
    assert compare(actual, expected)['passed']
    actual['water_mass'] += 2 * allowance
    assert not compare(actual, expected)['passed']



def test_zero_water_has_documented_dry_volume_cancellation_exception():
    expected = fake_outputs()
    expected['water_mass'][:] = 0.
    expected['dry_radius_cm'][:] = 1e-5
    expected['number'][:] = 1e8
    dry_scale = ((constants.PI/6) * (.02 * expected['dry_radius_cm'])**3
                 * constants.WATER_DENSITY * expected['number'])
    allowance = 32 * np.finfo(float).eps * dry_scale
    actual = copies(expected)
    actual['water_mass'][:] = allowance / 2
    result = compare(actual, expected)
    assert result['passed']
    assert result['fields']['water_mass']['nonzero_at_zero_reference'] == actual['water_mass'].size
    actual['water_mass'][:] = 2 * allowance
    assert not compare(actual, expected)['passed']
    expected['number'][:] = 0.
    actual['number'][:] = 0.
    actual['water_mass'][:] = 1e-30
    assert not compare(actual, expected)['passed']


def budget_case(*, source):
    config = replace(load_config(CONFIG), steps=3, timestep_s=2.,
                     h2so4_source_molecules_m3_s=source)
    mass, number, gas, mapping, _ = initial_state(config)
    initial = {key: np.array(value) for key, value in
               (('mass', mass), ('number', number), ('gas', gas))}
    values = {key: np.zeros((config.steps, 1, width)) for key, width in OUTPUTS.items()}
    for key in initial:
        values[key][:] = initial[key]
    values['gas'][:, 0, 1] += np.arange(1, config.steps+1) * config.timestep_s * source
    return config, initial, values, mapping


@pytest.mark.parametrize('source', [0., 1e11])
def test_source_and_closed_sulfur_budget_independent_of_native_parity(source):
    config, initial, values, mapping = budget_case(source=source)
    report = summarize(config, initial, values, mapping)
    assert report['sulfur_budget_passed'] and report['pure_sulfate_soluble_only']
    assert report['supplied_sulfur_molecules_m3'] == source * config.steps * config.timestep_s
    np.testing.assert_allclose(report['final_sulfur_molecules_m3'],
        report['initial_sulfur_molecules_m3'] + report['supplied_sulfur_molecules_m3'], rtol=3e-15)
    values['gas'][-1, 0, 1] += 1e10
    # Equal implementation outputs would still fail the physical budget gate.
    assert compare(values, values)['passed']
    assert not summarize(config, initial, values, mapping)['sulfur_budget_passed']


def test_sulfur_transfer_and_permuted_storage_preserve_budget_and_purity():
    config, initial, values, mapping = budget_case(source=1e11)
    # Move one million molecules/m3 from vapor into aerosol SO4 molecules/cm3.
    values['gas'][..., 1] -= 1e6
    values['mass'][..., mapping[3, 0]] += 1.
    permutation = np.random.default_rng(831).permutation(69)
    changed_map = np.where(mapping >= 0, permutation[np.maximum(mapping, 0)], -1)
    initial_mass = initial['mass'].copy()
    initial['mass'][:, permutation] = initial_mass
    mass = values['mass'].copy()
    values['mass'][..., permutation] = mass
    report = summarize(config, initial, values, changed_map)
    assert report['sulfur_budget_passed'] and report['pure_sulfate_soluble_only']
    expected = sulfur_inventory(initial['mass'], initial['gas'], changed_map)
    np.testing.assert_allclose(report['initial_sulfur_molecules_m3'], expected[0], rtol=0, atol=0)


@pytest.mark.parametrize('impurity', ['material', 'gas', 'insoluble_number'])
def test_purity_gate_is_independent_of_sulfur_conservation(impurity):
    config, initial, values, mapping = budget_case(source=0.)
    if impurity == 'material':
        values['mass'][1, 0, mapping[3, 1]] = 1.
    elif impurity == 'gas':
        values['gas'][1, 0, 2] = 1.
    else:
        values['number'][1, 0, 10] = 1.
    report = summarize(config, initial, values, mapping)
    assert report['sulfur_budget_passed']
    assert not report['pure_sulfate_soluble_only']


def test_insoluble_sulfate_mass_fails_purity_even_without_particle_number():
    config, initial, values, mapping = budget_case(source=0.)
    # A spurious 2b sulfate residue must not pass just because its number is zero.
    values['mass'][1, 0, mapping[3, 0]] -= 1.
    values['mass'][1, 0, mapping[10, 0]] += 1.
    report = summarize(config, initial, values, mapping)
    assert report['sulfur_budget_passed']
    assert np.all(values['number'][..., 10:] == 0)
    assert not report['pure_sulfate_soluble_only']


@pytest.mark.parametrize('field', ['mass', 'number', 'gas'])
def test_negative_prognostic_outputs_rejected(field):
    config, initial, values, mapping = budget_case(source=0.)
    values[field][1, 0, 0] = -1.
    with pytest.raises(ValueError, match='Negative trajectory'):
        summarize(config, initial, values, mapping)


def test_negative_water_is_reported_without_silent_clipping():
    config, initial, values, mapping = budget_case(source=0.)
    values['water_mass'][1, 0, 0] = -1e-30
    report = summarize(config, initial, values, mapping)
    assert report['negative_water_values'] == 1
    assert values['water_mass'][1, 0, 0] == -1e-30


def test_reference_adapter_uses_exact_initial_arrays_and_switches():
    config = replace(load_config(CONFIG), steps=3, nucleation_scheme=3, growth_option=2,
                     coagulation_enabled=False)
    mass, number, gas, mapping, _ = initial_state(config)
    case = reference_case(config)
    for key, expected in (('mass', mass), ('number', number), ('gas', gas)):
        np.testing.assert_array_equal(case[key], expected)
    np.testing.assert_array_equal(case['component_map'], mapping[None])
    assert case['scheme'][0] == 3 and case['growth'][0] == 2
    assert not case['coagulation_enabled'][0]
    assert not case['organic_enabled'][0]
    assert case['boundary_layer'][0]
    assert case['timestep'][0] == config.timestep_s


@pytest.fixture(scope='module')
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp('sulfur-cli')
    config = root/'config.toml'
    config.write_text(CONFIG.read_text().replace('steps = 600', 'steps = 3'))
    output = root/'output'
    report = cli.run(config, output, compare_fortran=True)
    return config, output, report


def test_cli_actual_fortran_run_writes_complete_verified_artifacts(artifacts):
    config_path, output, report = artifacts
    assert report['accepted'] and report['comparison']['passed']
    assert report['jax']['sulfur_budget_passed'] and report['fortran']['sulfur_budget_passed']
    assert report['fortran_provenance']['checked']
    assert '-fcheck=all' in report['fortran_provenance']['flags']
    assert json.loads(json.dumps(report)) == json.loads((output/'report.json').read_text())
    assert report['config_sha256'] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    for name, sha in report['files'].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest() == sha
    assert 'molecules/cm³' in (output/'README.md').read_text()
    timeseries = np.loadtxt(output/'timeseries.csv', delimiter=',', skiprows=1)
    assert timeseries.shape == (4, 7)
    np.testing.assert_array_equal(timeseries[:, 0], [0., 1., 2., 3.])
    assert timeseries[0, 1] == 1e14
    np.testing.assert_allclose(timeseries[:, 4], timeseries[:, 1] + timeseries[:, 2], rtol=3e-15)
    np.testing.assert_array_equal(timeseries[:, 6], timeseries[:, 4] - timeseries[:, 5])
    distribution = np.loadtxt(output/'number-distribution.csv', delimiter=',', skiprows=1)
    assert distribution.shape == (4, 18)
    np.testing.assert_allclose(distribution[:, 1:].sum(-1), timeseries[:, 3], rtol=3e-15)
    bins = np.loadtxt(output/'bins.csv', delimiter=',', skiprows=1)
    assert bins.shape == (17, 4)
    assert np.all(bins[:, 1] < bins[:, 2]) and np.all(bins[:, 2] < bins[:, 3])
    with np.load(output/'trajectory.npz') as actual, np.load(output/'fortran-trajectory.npz') as native:
        for key, width in OUTPUTS.items():
            assert actual[key].shape == (3, 1, width)
        assert compare({key: actual[key] for key in OUTPUTS}, {key: native[key] for key in OUTPUTS})['passed']
        np.testing.assert_array_equal(actual['initial_gas'], [[0., 1e14, 0.]])
        np.testing.assert_array_equal(actual['time_s'], [1., 2., 3.])
        assert actual['component_map'].shape == (17, 5)


def test_cli_requires_fresh_output_and_preserves_existing_artifacts(artifacts):
    config_path, output, _ = artifacts
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    with pytest.raises(FileExistsError):
        cli.run(config_path, output, compare_fortran=True)
    after = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    assert after == before


@pytest.mark.parametrize('accepted,exit_code', [(True, 0), (False, 1)])
def test_cli_exit_status_reflects_validation_gate(monkeypatch, tmp_path, accepted, exit_code):
    calls = []
    def run(config, output, *, compare_fortran):
        calls.append((config, output, compare_fortran))
        return {'accepted': accepted, 'jax': {'sulfur_budget_passed': accepted}}
    monkeypatch.setattr(cli, 'run', run)
    monkeypatch.setattr('sys.argv', ['run_sulfuric_acid', '--config', 'settings.toml',
                                   '--output', str(tmp_path/'new'), '--compare-fortran'])
    assert cli.main() == exit_code
    assert calls == [(Path('settings.toml'), tmp_path/'new', True)]
