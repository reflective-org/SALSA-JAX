"""Actual Fortran trajectory reference preserves native calls and source units."""
import hashlib

import numpy as np
import pytest

from tools import salsa_oracle as salsa
from tools import sulfuric_acid_oracle as sulfur


def subset(cases, labels):
    indices = [cases['labels'].index(label) for label in labels]
    return {key: np.asarray(value)[indices].copy() for key, value in cases.items()
            if key != 'labels'}


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    directory = tmp_path_factory.mktemp('sulfur-trajectory')
    executable, provenance = sulfur.build_oracle(directory)
    cases = subset(salsa.reference_cases(), ['captured-default', 'active-scheme1-growth1',
                   'active-scheme2-growth1', 'active-scheme3-growth1', 'active-above-boundary'])
    native_executable, _ = salsa.build_oracle(tmp_path_factory.mktemp('sulfur-native'))
    return directory, executable, provenance, cases, native_executable


def test_unchanged_scientific_modules_and_wrapper_provenance(oracle):
    directory, _, provenance, _, _ = oracle
    assert provenance['checked'] and '-fcheck=all' in provenance['flags']
    assert provenance['compiled_constants']['avogadro'] == 6.02214179e23
    for name in salsa.PROCESS_SOURCES:
        actual = (directory / name).read_bytes()
        assert actual == (salsa.ROOT / name).read_bytes()
        assert hashlib.sha256(actual).hexdigest() == provenance['source_sha256'][name]
    assert hashlib.sha256((directory / 'driver.f90').read_bytes()).hexdigest() == provenance['wrapper_sha256']['driver.f90']
    assert provenance['parent_driver_sha256'] == hashlib.sha256(salsa.DRIVER.encode()).hexdigest()


def test_zero_source_every_saved_step_matches_existing_native_driver(oracle):
    _, executable, _, cases, native_executable = oracle
    output, _ = sulfur.run_oracle(executable, cases, steps=5, all_levels=True)
    for step in range(1, 6):
        expected, _ = salsa.run_oracle(native_executable, cases, steps=step, all_levels=True)
        for key, size in salsa.OUTPUTS.items():
            assert output[key].shape == (5, len(cases['number']), 2, size)
            np.testing.assert_array_equal(output[key][step - 1], expected[key])


def test_one_forced_step_matches_manually_advanced_gas(oracle):
    _, executable, _, cases, native_executable = oracle
    rates = np.arange(1, len(cases['number']) + 1) * 1e11
    original = {key: value.copy() for key, value in cases.items()}
    advanced = {key: value.copy() for key, value in cases.items()}
    advanced['gas'][:, 1] += rates * advanced['timestep']
    actual, _ = sulfur.run_oracle(executable, cases, source_rate=rates, all_levels=True)
    expected, _ = salsa.run_oracle(native_executable, advanced, all_levels=True)
    for key in salsa.OUTPUTS:
        np.testing.assert_array_equal(actual[key][0], expected[key])
    for key in cases:
        np.testing.assert_array_equal(cases[key], original[key])


def test_forcing_applied_each_step_and_only_to_sulfuric_acid(oracle):
    _, executable, _, cases, _ = oracle
    one = {key: value[:1].copy() for key, value in cases.items()}
    one['condensation_enabled'][:] = False
    one['coagulation_enabled'][:] = False
    one['scheme'][:] = 0
    one['gas'][:] = [2e10, 3e10, 4e10]
    one['timestep'][:] = 2.
    output, _ = sulfur.run_oracle(executable, one, source_rate=5e10, steps=7)
    expected = np.broadcast_to(one['gas'], (7, 1, 3)).copy()
    expected[:, 0, 1] += np.arange(1, 8) * 1e11
    np.testing.assert_array_equal(output['gas'], expected)


def test_selected_level_and_native_boundary_behavior(oracle):
    _, executable, _, cases, _ = oracle
    full, _ = sulfur.run_oracle(executable, cases, source_rate=1e11, steps=4, all_levels=True)
    selected, _ = sulfur.run_oracle(executable, cases, source_rate=1e11, steps=4)
    for key in salsa.OUTPUTS:
        np.testing.assert_array_equal(selected[key], full[key][:, np.arange(len(cases['number'])), cases['boundary_layer'].astype(int)])
    # Activation nucleation remains controlled by the actual native level.
    assert full['number'][0, 2, 1, 0] > full['number'][0, 2, 0, 0]


def test_checked_optimized_and_repeated_resets_exact(oracle, tmp_path):
    _, executable, _, cases, _ = oracle
    checked, timing = sulfur.run_oracle(executable, cases, source_rate=1e11, steps=8, all_levels=True)
    repeat, _ = sulfur.run_oracle(executable, cases, source_rate=1e11, steps=8,
                                  repetitions=3, all_levels=True)
    optimized, provenance = sulfur.build_oracle(tmp_path, checked=False)
    fast, _ = sulfur.run_oracle(optimized, cases, source_rate=1e11, steps=8, all_levels=True)
    assert not provenance['checked']
    for key in salsa.OUTPUTS:
        np.testing.assert_array_equal(checked[key], repeat[key])
        np.testing.assert_array_equal(checked[key], fast[key])
    for key in ('mass', 'number', 'gas'):
        # Native SUM walks the saved Fortran array sequentially; NumPy sum
        # uses a different reduction tree over this larger trajectory.
        assert timing[key + '_checksum'] == sum(checked[key].ravel())
    assert timing['checksums_include_all_steps_and_levels']
    assert timing['forcing_included'] and timing['trajectory_recording_included']


@pytest.mark.parametrize('source_rate', [-1., float('nan'), float('inf'), [1., 2.], [[1.]]])
def test_invalid_forcing_rejected_without_subprocess(source_rate):
    cases = subset(salsa.reference_cases(), ['captured-default'])
    with pytest.raises(ValueError, match='source_rate'):
        sulfur.run_oracle('must-not-execute', cases, source_rate)


@pytest.mark.parametrize('steps', [0, -1, 1.5, True])
def test_invalid_steps_rejected_without_subprocess(steps):
    cases = subset(salsa.reference_cases(), ['captured-default'])
    with pytest.raises(ValueError, match='positive integers'):
        sulfur.run_oracle('must-not-execute', cases, steps=steps)


@pytest.mark.parametrize('field', ['mass', 'gas', 'temperature', 'scheme', 'component_map'])
def test_invalid_cases_rejected_without_subprocess(field):
    cases = subset(salsa.reference_cases(), ['captured-default'])
    cases[field] = np.full_like(cases[field], -1)
    with pytest.raises(ValueError):
        sulfur.run_oracle('must-not-execute', cases)


def test_wrapper_anchor_changes_fail_closed(monkeypatch):
    monkeypatch.setattr(salsa, 'DRIVER', salsa.DRIVER.replace('read(*,*) initial_gas(:,i)', ''))
    with pytest.raises(ValueError, match='exactly one native wrapper anchor'):
        sulfur.trajectory_driver()
