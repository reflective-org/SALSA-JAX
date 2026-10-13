# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Native hydration extraction, iteration diagnostics and supported state checks."""
import subprocess

import numpy as np
import pytest

from tools.hydration_oracle import ROOT, _extract, build_oracle, reference_cases, run_molality, run_oracle
from tools.state_capture import load_fixture


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    directory = tmp_path_factory.mktemp('hydration-reference')
    executable, provenance = build_oracle(directory)
    cases = reference_cases()
    outputs, timing = run_oracle(executable, cases)
    return directory, executable, provenance, cases, outputs, timing


def test_actual_routines_extracted_and_checked(oracle):
    directory, _, provenance, _, _, _ = oracle
    source = (ROOT / 'mo_ham_salsa_properties.F90').read_text()
    assert (directory / 'equilibration_verbatim.inc').read_text() == _extract(source, 'SUBROUTINE', 'equilibration') + '\n'
    assert (directory / 'molality_verbatim.inc').read_text() == _extract(source, 'FUNCTION', 'zbinmol_inv') + '\n'
    assert '-fcheck=all' in provenance['flags']
    assert '-ffpe-trap=invalid,zero,overflow' in provenance['flags']
    assert provenance['compiled_constants']['avogadro'] == 6.02214179e23
    assert provenance['configuration']['soluble_mask_controls_hydration']
    assert not provenance['configuration']['lhydr2b_directly_used']


def test_native_final_state_reproduces_captured_geometry(oracle):
    _, _, _, cases, outputs, _ = oracle
    fixture = load_fixture(ROOT / 'tests/fixtures/native-default')
    index = cases['labels'].index('native-final-state')
    for name, native_name in (('core', 'zcore'), ('wet_diameter', 'zdwet')):
        item = fixture['exit'][native_name]
        actual = np.array(item['values']).reshape(item['shape'], order='F')[0, 0]
        np.testing.assert_array_equal(outputs[name][index], actual)
    assert np.all(cases['soluble_mask'][index])


def test_optimized_and_repeated_calls_match_checked_results(oracle, tmp_path):
    _, executable, _, cases, expected, _ = oracle
    repeated, timing = run_oracle(executable, cases, repetitions=3)
    optimized, provenance = build_oracle(tmp_path, checked=False)
    fast, _ = run_oracle(optimized, cases)
    assert not provenance['checked']
    for field in expected:
        np.testing.assert_array_equal(repeated[field], expected[field])
        np.testing.assert_array_equal(fast[field], expected[field])
    for field in ('core', 'wet_diameter'):
        np.testing.assert_allclose(timing[field + '_checksum'], expected[field].sum(), rtol=3e-15)
    assert timing['kernel_wall_seconds'] > 0
    assert not timing['iteration_capture_included']


def test_pure_insoluble_material_in_soluble_bin_stays_dry(oracle):
    _, _, _, cases, outputs, _ = oracle
    for label in ('pure-BC-soluble', 'pure-DU-soluble'):
        index = cases['labels'].index(label)
        dry = (outputs['core'][index] / (np.pi / 6))**(1/3)
        np.testing.assert_allclose(outputs['wet_diameter'][index], dry, rtol=2e-15, atol=0)
        np.testing.assert_array_equal(outputs['iterations'][index], np.full(17, 2))
    index = cases['labels'].index('all-insoluble')
    np.testing.assert_array_equal(outputs['iterations'][index], np.zeros(17, dtype=int))


def test_count_and_rh_branches(oracle):
    _, _, _, cases, outputs, _ = oracle
    for label in ('empty', 'below-count-threshold', 'at-count-threshold'):
        index = cases['labels'].index(label)
        np.testing.assert_array_equal(outputs['wet_diameter'][index], cases['dry_diameter'][index])
        np.testing.assert_array_equal(outputs['iterations'][index], np.zeros(17, dtype=int))
        np.testing.assert_allclose(outputs['core'][index], np.pi / 6 * cases['dry_diameter'][index]**3, rtol=2e-15, atol=0)
    above = cases['labels'].index('above-count-threshold')
    assert np.all(outputs['iterations'][above] > 0)
    for left, right in (('rh--0.1', 'rh-0.05'), ('rh-0.0', 'rh-0.05'), ('rh-1.0', 'rh-0.95'), ('rh-1.2', 'rh-0.95')):
        li, ri = cases['labels'].index(left), cases['labels'].index(right)
        for field in outputs:
            np.testing.assert_array_equal(outputs[field][li], outputs[field][ri])


def test_molality_leaf_and_organic_scaling(oracle):
    _, executable, _, _, _, _ = oracle
    activities = np.array([[.01], [.05], [.3], [.8], [.95]])
    reference = run_molality(executable, activities)
    assert reference.shape == (5, 1, 5)
    assert np.all(reference[..., [0, 1, 3]] > 0)
    assert not reference[..., [2, 4]].any()
    varied = run_molality(executable, activities, water_molar_mass=np.array([18., 36.]), organic_soluble_fraction=.3)
    np.testing.assert_allclose(varied[:, 0, 1], 2 * reference[:, 0, 1], rtol=3e-15)
    np.testing.assert_allclose(varied[:, 1, 1], 4 * reference[:, 0, 1], rtol=3e-15)
    zero = run_molality(executable, .5, organic_soluble_fraction=0.)
    assert zero[1] == 0


def test_active_soluble_zero_volume_is_a_native_invalid_state(oracle):
    _, executable, _, cases, _, _ = oracle
    case = {key: value[:1].copy() for key, value in cases.items() if key != 'labels'}
    case['volume'][:] = 0
    # The native soluble branch evaluates Kelvin/diameter after diameter=0.
    # Keep the strict trap visible; neither sanitize the input nor alter source.
    with pytest.raises(subprocess.CalledProcessError) as failure:
        run_oracle(executable, case)
    assert failure.value.returncode < 0
    assert 'Program received signal' in failure.value.stderr


@pytest.mark.parametrize('field', ['number', 'volume', 'temperature', 'dry_diameter', 'species_molar_mass', 'water_density'])
def test_invalid_inputs_rejected(field):
    cases = reference_cases()
    value = np.full_like(cases[field], -1.)
    with pytest.raises(ValueError, match=field):
        run_oracle('must-not-run', {**cases, field: value})
