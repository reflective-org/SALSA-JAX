"""Native component conversion and final-density parity, including absent slots."""
import hashlib
from pathlib import Path

import jax
import numpy as np
import pytest

from salsa_jax.constants import AVOGADRO, DENSITY_VOLUME_THRESHOLD
from salsa_jax.interface import mass_to_volume, volume_to_mass, particle_density, validate_component_map
from salsa_jax.processes.hydration import HydrationParameters
from tools.hydration_fixture import load_native_case
from tools.interface_oracle import build_oracle, native_blocks, native_cases, reference_cases, run_oracle
from tools.state_capture import load_fixture

jax.config.update('jax_enable_x64', True)
ROOT = Path(__file__).resolve().parents[1]


def parameters(cases):
    return HydrationParameters(**{key: cases[key] for key in ('species_molar_mass', 'species_density', 'water_density')})


def evaluate(cases):
    props = parameters(cases)
    return {'volume_from_mass': mass_to_volume(cases['mass'], cases['component_map'], props),
            'mass_from_volume': volume_to_mass(cases['volume'], cases['component_map'], props),
            'density': particle_density(cases['volume'], cases['water_mass'], props)}


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    directory = tmp_path_factory.mktemp('interface')
    executable, provenance = build_oracle(directory)
    cases = reference_cases()
    expected = run_oracle(executable, cases)
    return directory, executable, provenance, cases, expected


def test_reference_uses_verbatim_original_blocks_and_compiled_constants(reference):
    directory, _, provenance, _, _ = reference
    for name, block in native_blocks().items():
        assert (directory / f'{name}_verbatim.inc').read_text() == block
        assert provenance['block_sha256'][name] == hashlib.sha256(block.encode()).hexdigest()
    assert provenance['compiled_constants']['avogadro'] == AVOGADRO
    assert '-fcheck=all' in provenance['flags']
    assert '-ffpe-trap=invalid,zero,overflow' in provenance['flags']
    assert DENSITY_VOLUME_THRESHOLD == 1e-30
    assert '1.e-30_dp' in native_blocks()['particle_density']


@pytest.mark.parametrize('compiled', [False, True])
def test_interface_matches_checked_fortran(reference, compiled):
    _, _, _, cases, expected = reference
    data = {key: value for key, value in cases.items() if key != 'labels'}
    original = {key: value.copy() for key, value in data.items()}
    actual = (jax.jit(evaluate) if compiled else evaluate)(data)
    for key in expected:
        assert actual[key].dtype == np.float64
        np.testing.assert_allclose(actual[key], expected[key], rtol=3e-15, atol=0)
    for key in original:
        np.testing.assert_array_equal(data[key], original[key])


def test_reference_reproduces_actual_native_entry_and_exit(reference):
    _, executable, _, _, _ = reference
    cases = native_cases()
    actual = run_oracle(executable, cases)
    initial, _, _ = load_native_case()
    np.testing.assert_allclose(actual['volume_from_mass'], initial['volume'], rtol=3e-15, atol=0)
    state = load_fixture(ROOT / 'tests/fixtures/native-default')
    def record(name):
        field = state['exit'][name]
        return np.asarray(field['values']).reshape(field['shape'], order='F').reshape(1, -1)
    np.testing.assert_array_equal(actual['mass_from_volume'], record('paerml'))
    # O2 leaf versus the checked native caller may differ by one ulp.
    np.testing.assert_allclose(actual['density'], record('prhop'), rtol=3e-15, atol=0)
    # Defaults must agree with the actual species registry, not just the stubs.
    np.testing.assert_array_equal(cases['species_density'][0], HydrationParameters().species_density)
    np.testing.assert_array_equal(cases['species_molar_mass'][0], HydrationParameters().species_molar_mass)
    assert cases['water_density'][0] == HydrationParameters().water_density


@pytest.mark.parametrize('compiled', [False, True])
def test_native_region_threshold_and_signed_water_policy(reference, compiled):
    _, _, _, cases, expected = reference
    actual = (jax.jit(evaluate) if compiled else evaluate)({k: v for k, v in cases.items() if k != 'labels'})
    labels = cases['labels']
    density = np.asarray(actual['density'])
    fallback = np.r_[np.ones(10), np.full(7, 2.)]
    for label in ('empty', 'density-threshold-below', 'density-threshold-equal', 'signed-water-negative-total'):
        np.testing.assert_array_equal(density[labels.index(label)], fallback)
    np.testing.assert_allclose(density[labels.index('density-threshold-above')], 1.841, rtol=3e-16, atol=0)
    np.testing.assert_array_equal(density[labels.index('pure-material-3'), 10:], 2.)
    np.testing.assert_array_equal(density[labels.index('pure-material-2'), :3], 1.)
    unclipped = particle_density(cases['volume'][labels.index('signed-water-small')], np.zeros(17))
    assert np.any(density[labels.index('signed-water-small')] != np.asarray(unclipped))
    np.testing.assert_allclose(density, expected['density'], rtol=3e-15, atol=0)


@pytest.mark.parametrize('compiled', [False, True])
def test_native_and_permuted_map_roundtrip_with_case_axis_broadcasting(reference, compiled):
    _, _, _, cases, _ = reference
    mass = cases['mass'][:, None, :]
    mapping = cases['component_map'][:, None, :, :]
    props = HydrationParameters(species_molar_mass=cases['species_molar_mass'][:, None, :],
                                species_density=cases['species_density'][:, None, :])
    def roundtrip(m, maps, p):
        volume = mass_to_volume(m, maps, p)
        return volume, volume_to_mass(volume, maps, p)
    volume, restored = (jax.jit(roundtrip) if compiled else roundtrip)(mass, mapping, props)
    np.testing.assert_allclose(restored, mass, rtol=5e-16, atol=0)
    np.testing.assert_array_equal(np.asarray(volume)[-2:, 0], np.broadcast_to(volume[0, 0], (2, 17, 5)))
    np.testing.assert_array_equal(np.asarray(volume)[..., :3, 2:], 0)
    np.testing.assert_array_equal(np.asarray(volume)[..., 10:, 3], 0)
    # No leading case axis and one shared map are both supported.
    single = mass_to_volume(mass[0, 0], mapping[0, 0])
    np.testing.assert_allclose(single, volume[0, 0], rtol=3e-15, atol=0)


def test_unrepresented_material_is_omitted_without_redistribution(reference):
    _, _, _, cases, _ = reference
    mapping = cases['component_map'][0]
    represented = mapping >= 0
    volume = np.where(represented, 0., 1e-12)
    np.testing.assert_array_equal(volume_to_mass(volume, mapping), np.zeros(69))
    np.testing.assert_array_equal(particle_density(volume, np.zeros(17)), np.r_[np.ones(10), np.full(7, 2.)])


@pytest.mark.parametrize('defect', ['shape', 'float', 'missing', 'duplicate', 'range', 'absent'])
def test_map_adapter_rejects_unsupported_mapping(defect):
    mapping = native_cases()['component_map'][0].copy()
    if defect == 'shape':
        mapping = mapping[:-1]
    elif defect == 'float':
        mapping = mapping.astype(float)
    elif defect == 'missing':
        mapping[0, 0] = -1
    elif defect == 'duplicate':
        mapping[0, 0] = mapping[0, 1]
    elif defect == 'range':
        mapping[0, 0] = 69
    else:
        mapping[0, 2] = 0
    with pytest.raises(ValueError):
        validate_component_map(mapping)


def test_validation_accepts_batched_permutations(reference):
    mapping = reference[3]['component_map']
    np.testing.assert_array_equal(validate_component_map(mapping), mapping)


@pytest.mark.parametrize('function, args', [
    (mass_to_volume, (np.zeros(68), np.zeros((17, 5), dtype=int))),
    (mass_to_volume, (np.zeros(69), np.zeros((17, 5), dtype=float))),
    (volume_to_mass, (np.zeros((17, 4)), np.zeros((17, 5), dtype=int))),
    (particle_density, (np.zeros((17, 5)), np.zeros(16))),
])
def test_interface_rejects_incompatible_shapes(function, args):
    with pytest.raises(ValueError):
        function(*args)
