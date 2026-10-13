"""Complete composition matches every captured process boundary in native calls."""
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from salsa_jax.model import STATIC_OPTIONS, salsa_step
from tools import coagulation_capture, condensation_capture, hydration_capture
from tools.hydration_fixture import record
from tools.remapping_fixture import load_native_case as load_remapping
from tools.salsa_fixture import load_native_case
from tools.salsa_inputs import arguments

jax.config.update('jax_enable_x64', True)
FIXTURES = Path(__file__).resolve().parent / 'fixtures'


def field(capture, stage, name):
    array = record(capture, stage, name)
    array = array.reshape((-1,) + array.shape[2:])
    if name == 'pc_gas' and array.shape[-1] == 2:
        array = np.pad(array, ((0, 0), (0, 1)))
    return array


def assert_fields(actual, expected, water_density):
    for name, reference in expected.items():
        value = np.asarray(getattr(actual, name))
        if name == 'water_mass':
            scale = expected['core'] * water_density[:, None] * expected['number']
            bound = 3e-13*np.abs(reference) + 32*np.finfo(float).eps*scale
            assert np.all(np.abs(value-reference) <= bound)
        else:
            np.testing.assert_allclose(value, reference, rtol=3e-13, atol=0, err_msg=name)


@pytest.mark.parametrize('profile', ['default', 'active', 'recalc', 'active-recalc'])
@pytest.mark.parametrize('compiled', [False, True])
def test_complete_native_call_and_all_process_boundaries(profile, compiled):
    native_dir = FIXTURES / f'native-{profile}'
    def process_dir(process):
        return (FIXTURES/f'{process}-default' if profile == 'default' else
                FIXTURES/f'salsa-boundaries-{profile}'/process)
    cases, expected, _ = load_native_case(native_dir)
    native_manifest = json.loads((native_dir/'manifest.json').read_text())
    for process in ('hydration', 'coagulation', 'condensation', 'remapping'):
        manifest = json.loads((process_dir(process)/'manifest.json').read_text())
        assert manifest['source_sha256'] == native_manifest['source_sha256']
        assert manifest['input_sha256'] == native_manifest['input_sha256']
        if profile != 'default':
            assert manifest['run_manifest_sha256'] == native_manifest['run_manifest_sha256']
    args, kwargs = arguments(cases, return_trace=True)
    kernel = jax.jit(salsa_step, static_argnames=STATIC_OPTIONS) if compiled else salsa_step
    result, trace = kernel(*args, **kwargs)
    assert np.all(result.valid)
    assert_fields(result, expected, cases['water_density'])

    hydration = hydration_capture.load_fixture(process_dir('hydration'))
    coagulation = coagulation_capture.load_fixture(process_dir('coagulation'))
    condensation = condensation_capture.load_fixture(process_dir('condensation'))
    assert_fields(trace['initial_cleanup'], {
        'number': field(hydration, 'entry', 'pnaero'),
        'volume': field(hydration, 'entry', 'pvols')}, cases['water_density'])
    assert_fields(trace['initial_hydration'], {
        'core': field(hydration, 'exit', 'pcore'),
        'wet_diameter': field(hydration, 'exit', 'pdwet')}, cases['water_density'])
    assert_fields(trace['coagulation'], {
        'number': field(coagulation, 'exit', 'pnaero'),
        'volume': field(coagulation, 'exit', 'zvols')}, cases['water_density'])
    assert_fields(trace['condensation_hydration'], {
        'wet_diameter': field(condensation, 'entry', 'zdwet')}, cases['water_density'])
    assert_fields(trace['condensation'], {
        'number': field(condensation, 'exit', 'pnaero'),
        'volume': field(condensation, 'exit', 'zvols'),
        'gas': field(condensation, 'exit', 'pc_gas')}, cases['water_density'])
    stages, _ = load_remapping(process_dir('remapping'))
    for index, stage in enumerate(stages):
        assert_fields(trace['remapping'][index], {
            name: stage['remapped'][name] for name in ('number', 'volume')}, cases['water_density'])
        assert_fields(trace['correction'][index], stage['corrected'], cases['water_density'])
