# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Saved long-experiment reports retain native evidence and strict failures."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from salsa_jax.sulfuric_acid import sulfur_inventory
from tools.long_sulfuric_acid import load_experiment
from tools.long_sulfuric_acid_oracle import build_oracle
from tools import run_long_sulfuric_acid as runner
from tools.salsa_oracle import OUTPUTS
from tools.sulfuric_acid_validation import compare

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def references(tmp_path_factory):
    directory = tmp_path_factory.mktemp('long-runner-native')
    return (build_oracle(directory/'checked'),
            build_oracle(directory/'optimized', checked=False))


def short_experiment(constant):
    name = 'sulfuric-acid-six-hours.toml' if constant else 'coagulation-24-hours.toml'
    return replace(load_experiment(ROOT/'configs'/name), duration_hours=2/3600,
                   timestep_s=1., save_interval_s=1.)


@pytest.mark.parametrize('constant', [False, True])
def test_actual_native_runs_write_reproducible_units_budgets_and_gates(tmp_path, references, constant):
    experiment = short_experiment(constant)
    output = tmp_path/'run'
    report = runner.run(experiment, output, references=references)
    assert json.loads((output/'report.json').read_text()) == report
    assert report['steps'] == report['snapshots'] == 2
    assert report['checked_optimized_exact']
    assert '-fcheck=all' in report['checked_provenance']['flags']
    assert '-fcheck=all' not in report['optimized_provenance']['flags']
    assert report['configuration']['constant_h2so4'] is constant
    for name, expected_hash in report['files'].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest() == expected_hash
    for name, expected_hash in report['implementation_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected_hash
    with np.load(output/'initial.npz') as saved:
        initial = {key: saved[key] for key in saved.files}
    results = {}
    for label in ('jax', 'fortran'):
        with np.load(output/f'{label}.npz') as saved:
            values = {key: saved[key] for key in saved.files}
        results[label] = values
        np.testing.assert_array_equal(values['time_s'], [1., 2.])
        for key, width in OUTPUTS.items():
            assert values[key].shape == (2, 1, width)
        reservoir = values['cumulative_reservoir_added']
        if constant:
            np.testing.assert_array_equal(values['gas'][..., 1], 1e14)
            assert np.all(reservoir > 0)
            assert np.all(values['h2so4_before_restore'] < 1e14)
            assert report[label]['constant_vapor_at_checkpoints']
        else:
            np.testing.assert_array_equal(values['gas'], 0.)
            np.testing.assert_array_equal(reservoir, 0.)
            assert report[label]['constant_vapor_at_checkpoints'] is None
        expected = sulfur_inventory(initial['mass'], initial['gas'], initial['component_map'])+reservoir
        np.testing.assert_allclose(sulfur_inventory(values['mass'], values['gas'], initial['component_map']),
                                   expected, rtol=3e-13)
        assert report[label]['sulfur_budget_passed']
        assert report[label]['pure_sulfate_soluble_only']
        counts = np.loadtxt(output/f'{label}-number-distribution.csv', delimiter=',', skiprows=1)
        np.testing.assert_array_equal(counts[:, 0], [0., 1/3600, 2/3600])
        np.testing.assert_array_equal(counts[0, 1:], initial['number'][0]*1e-6)
        np.testing.assert_array_equal(counts[1:, 1:], values['number'][:, 0]*1e-6)
        series = np.loadtxt(output/f'{label}-timeseries.csv', delimiter=',', skiprows=1)
        np.testing.assert_array_equal(series[:, 2], values['gas'][:, 0, 1]*1e-6)
        np.testing.assert_array_equal(series[:, 4], reservoir[:, 0])
    # Do not require this coarse-tail case to pass a strict relative gate;
    # require the persisted verdict to agree with the unmodified gate.
    comparison = compare(results['jax'], results['fortran'])
    assert report['comparison']['passed'] is comparison['passed']
    assert report['strict_acceptance_passed'] is comparison['passed']
    assert report['comparison']['relative_tolerance'] == 3e-11
    for key in OUTPUTS:
        assert report['comparison']['fields'][key]['passed'] is comparison['fields'][key]['passed']
    with pytest.raises(FileExistsError):
        runner.run(experiment, output, references=references)


def test_tiny_tail_difference_is_saved_as_strict_failure(tmp_path, references, monkeypatch):
    original = runner.run_jax
    def perturb_tail(experiment):
        initial, actual, monitors, cases, mapping, grid = original(experiment)
        actual = {key: value.copy() for key, value in actual.items()}
        # Tiny nonzero uppermost-bin number: negligible total error, but the
        # existing per-bin relative gate must still reject it without clipping.
        actual['number'][-1, 0, 9] *= 1.001
        return initial, actual, monitors, cases, mapping, grid
    monkeypatch.setattr(runner, 'run_jax', perturb_tail)
    output = tmp_path/'strict-failure'
    report = runner.run(short_experiment(False), output, references=references)
    assert report['checked_optimized_exact']
    assert not report['strict_acceptance_passed']
    assert not report['comparison']['passed']
    assert not report['comparison']['fields']['number']['passed']
    assert report['comparison']['fields']['number']['max_relative_error_nonzero'] >= 9e-4
    assert report['comparison']['fields']['number']['max_relative_L1_error'] < 1e-12
    assert report['jax']['sulfur_budget_passed'] and report['fortran']['sulfur_budget_passed']
    assert json.loads((output/'report.json').read_text())['strict_acceptance_passed'] is False
    with np.load(output/'jax.npz') as actual, np.load(output/'fortran.npz') as native:
        assert actual['number'][-1, 0, 9] != native['number'][-1, 0, 9]
