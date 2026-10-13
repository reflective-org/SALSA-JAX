# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Actual checked model-boundary acceptance for active nested nucleation."""
from copy import deepcopy
from pathlib import Path

import jax
import numpy as np
import pytest

from salsa_jax.constants import AVOGADRO
from salsa_jax.processes.condensation import CondensationParameters
from salsa_jax.processes.condensation_nucleation import condensation_nucleation_step
from salsa_jax.processes.nucleation import nucleation_sources
from tools.condensation_capture import load_fixture as load_condensation_fixture
from tools.nucleation_capture import load_fixture as load_nucleation_fixture
from tools.nucleation_fixture import adapt_captures, load_native_case
from tools.nucleation_process_oracle import build_oracle, run_oracle

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[1]
SOURCE_INPUTS = ('number', 'gas', 'wet_diameter', 'temperature', 'relative_humidity', 'pressure', 'timestep')
COUPLED_INPUTS = ('number', 'volume', 'gas', 'wet_diameter', 'temperature', 'relative_humidity', 'pressure', 'timestep')


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    cases, expected, metadata = load_native_case()
    executable, provenance = build_oracle(tmp_path_factory.mktemp('active-native-nucleation'))
    return cases, expected, metadata, executable, provenance


def test_actual_boundary_matches_extracted_oracle_before_jax(native):
    cases, expected, metadata, executable, provenance = native
    # This verifies that data stubs and isolated extraction reproduce the actual
    # initialized full-model call, before asserting any translated-code parity.
    assert metadata['scheme'] == 2
    assert metadata['growth'] == 1
    assert metadata['fixed_constants']['act_coeff'] == provenance['compiled_constants']['activation_coefficient']
    isolated, _ = run_oracle(executable, cases, operation='isolated')
    coupled, _ = run_oracle(executable, cases, operation='coupled')
    for field in ('number', 'volume', 'gas'):
        np.testing.assert_array_equal(isolated[field], cases[field])
        np.testing.assert_allclose(coupled[field], expected[field], rtol=3e-13, atol=0)
    for field in ('molecular_rate', 'fractions'):
        np.testing.assert_allclose(isolated[field], expected[field], rtol=3e-13, atol=0)
    assert np.all(expected['molecular_rate'][:, 0] > 0)
    assert np.all(expected['number'][:, 0] > cases['number'][:, 0])
    assert np.all(cases['boundary_layer'])
    assert not np.any(cases['organic_enabled'])
    assert not np.any(cases['gas'][:, 2])


@pytest.mark.parametrize('compiled', [False, True])
def test_native_sources_coupled_state_and_budgets(native, compiled):
    cases, expected, metadata, _, _ = native
    kwargs = {'scheme': metadata['scheme'], 'growth': metadata['growth'],
              'boundary_layer': cases['boundary_layer'], 'organic_enabled': cases['organic_enabled'],
              'activation_coefficient': metadata['fixed_constants']['act_coeff'],
              'parameters': CondensationParameters(**{name: cases[name] for name in CondensationParameters._fields})}
    source_kernel = jax.jit(nucleation_sources, static_argnames=('scheme', 'growth')) if compiled else nucleation_sources
    coupled_kernel = jax.jit(condensation_nucleation_step, static_argnames=('scheme', 'growth')) if compiled else condensation_nucleation_step
    source = source_kernel(*(cases[key] for key in SOURCE_INPUTS), **kwargs)
    state = coupled_kernel(*(cases[key] for key in COUPLED_INPUTS), **kwargs)
    for field in ('molecular_rate', 'fractions'):
        np.testing.assert_allclose(getattr(source, field), expected[field], rtol=3e-13, atol=0)
    for field in ('number', 'volume', 'gas'):
        actual = np.asarray(getattr(state, field))
        assert np.all(np.isfinite(actual))
        assert np.all(actual >= 0)
        np.testing.assert_allclose(actual, expected[field], rtol=3e-13, atol=0)
    np.testing.assert_array_equal(state.number[:, 1:], cases['number'][:, 1:])
    np.testing.assert_array_equal(state.gas[:, 0], cases['gas'][:, 0])
    np.testing.assert_array_equal(state.volume[..., 2:], cases['volume'][..., 2:])
    for material, name in enumerate(('sulfate', 'organic')):
        molecular_volume = cases[name + '_molar_mass'] / 1000 / AVOGADRO / cases[name + '_density']
        before = cases['gas'][:, material + 1] + cases['volume'][:, :, material].sum(axis=1) / molecular_volume
        for output in (expected, state._asdict()):
            after = np.asarray(output['gas'])[:, material + 1] + np.asarray(output['volume'])[:, :, material].sum(axis=1) / molecular_volume
            np.testing.assert_allclose(after, before, rtol=3e-14, atol=0)
    assert np.all(np.asarray(source.molecular_rate)[:, 0] > 0)
    assert np.all(np.asarray(state.number)[:, 0] > cases['number'][:, 0])


@pytest.mark.parametrize('field', ['act_coeff', 'n3', 'reglim1', 'avo'])
def test_unsupported_globals_are_not_silently_replaced(field):
    nested = deepcopy(load_nucleation_fixture(ROOT / 'tests/fixtures/nucleation-active'))
    parent = load_condensation_fixture(ROOT / 'tests/fixtures/condensation-active')
    nested['entry'][field]['values'][0] *= 1.01
    with pytest.raises(ValueError, match=field):
        adapt_captures(nested, parent)


def test_mismatched_parent_capture_is_rejected():
    nested = load_nucleation_fixture(ROOT / 'tests/fixtures/nucleation-active')
    parent = deepcopy(load_condensation_fixture(ROOT / 'tests/fixtures/condensation-active'))
    parent['entry']['ptemp']['values'][0] += 1
    with pytest.raises(ValueError, match='ptemp'):
        adapt_captures(nested, parent)


def test_adapter_preserves_horizontal_vertical_order_and_boundary_level():
    nested = deepcopy(load_nucleation_fixture(ROOT / 'tests/fixtures/nucleation-active'))
    parent = deepcopy(load_condensation_fixture(ROOT / 'tests/fixtures/condensation-active'))
    # Expand the captured cell to two columns/two levels with distinct number
    # markers; only the adapter contract is tested here, not synthetic physics.
    for capture in (nested, parent):
        for stage in ('entry', 'exit'):
            for name, item in capture[stage].items():
                shape = item['shape']
                if len(shape) >= 2 and shape[:2] == [1, 1]:
                    value = np.asarray(item['values']).reshape(shape, order='F')
                    value = np.tile(value, (2, 2) + (1,) * (len(shape) - 2))
                    item['shape'] = list(value.shape)
                    item['values'] = value.ravel(order='F').tolist()
                elif name == 'dimensions':
                    item['values'][:3] = [2, 2, 2]
                elif name == 'ppbl':
                    item['shape'] = [2]
                    item['values'] = [1, 2]
    markers = np.zeros((2, 2, 17))
    markers[:, :, 0] = [[11., 12.], [21., 22.]]
    for capture, stage in ((nested, 'entry'), (nested, 'exit'), (parent, 'entry')):
        capture[stage]['pnaero']['values'] = markers.ravel(order='F').tolist()
    cases, _, metadata = adapt_captures(nested, parent)
    assert metadata['native_cell_shape'] == [2, 2]
    assert cases['labels'] == ['native-h1-level1', 'native-h1-level2', 'native-h2-level1', 'native-h2-level2']
    np.testing.assert_array_equal(cases['number'][:, 0], [11., 12., 21., 22.])
    np.testing.assert_array_equal(cases['boundary_layer'], [True, True, False, True])
