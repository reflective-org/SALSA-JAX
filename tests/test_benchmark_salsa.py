"""Guard complete-call benchmark workload accounting and parity acceptance."""
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import benchmark_salsa as benchmark
from tools.salsa_inputs import arguments, select_cases, static_groups
from tools.salsa_oracle import OUTPUTS, reference_cases


def outputs(count=2):
    state = {name: np.ones((count, 2, width), dtype=np.float64) for name, width in OUTPUTS.items()}
    state['dry_radius_cm'][:] = 1e-5
    state['water_mass'][:] = 0
    return state


def test_paired_workload_keeps_both_levels_and_all_static_cases():
    cases = reference_cases()
    visited = []
    for indices in static_groups(cases):
        selected = select_cases(cases, indices)
        args, kwargs = arguments(selected, all_levels=True)
        visited.extend(indices.tolist())
        assert args[0].shape == (len(indices), 2, 69)
        assert args[1].shape == (len(indices), 2, 17)
        assert args[2].shape == (len(indices), 2, 3)
        np.testing.assert_array_equal(args[0][:, 0], selected['mass'])
        np.testing.assert_array_equal(args[0][:, 1], selected['mass'])
        np.testing.assert_array_equal(kwargs['boundary_layer'], np.broadcast_to([False, True], (len(indices), 2)))
        assert kwargs['return_trace'] is False
    assert sorted(visited) == list(range(len(cases['labels'])))
    assert len(visited) == len(set(visited))


def test_timed_wrapper_returns_every_native_output_and_no_internal_fields(monkeypatch):
    physical = {name: object() for name in OUTPUTS}
    state = SimpleNamespace(**physical, valid=False, volume=object(), core=object(), wet_diameter=object())
    monkeypatch.setattr(benchmark, 'salsa_step', lambda *args, **kwargs: state)
    assert benchmark.physical_outputs() == tuple(physical.values())


def test_collect_restores_original_case_order_and_two_levels():
    groups = [{'indices': np.array([2, 0])}, {'indices': np.array([1])}]
    results = []
    for group in groups:
        results.append(tuple(np.broadcast_to(group['indices'][:, None, None], (len(group['indices']), 2, width))
                             for width in OUTPUTS.values()))
    collected = benchmark._collect(groups, results, 3)
    for name, width in OUTPUTS.items():
        np.testing.assert_array_equal(collected[name], np.broadcast_to(np.arange(3)[:, None, None], (3, 2, width)))


def test_parity_records_zero_and_signed_water_without_clipping():
    expected = outputs()
    expected['water_mass'][0, 0, 0] = -1e-20
    actual = {key: value.copy() for key, value in expected.items()}
    metrics = benchmark.validate_outputs(actual, expected, {'water_density': np.array([1000., 997.])})
    assert metrics['water_mass']['negative_fortran_values'] == 1
    assert metrics['water_mass']['negative_jax_values'] == 1
    assert metrics['water_mass']['max_absolute_over_dry_water_mass_scale'] == 0
    assert set(metrics) == set(OUTPUTS)


@pytest.mark.parametrize('name', list(OUTPUTS))
def test_parity_rejects_any_changed_native_output(name):
    expected = outputs()
    actual = {key: value.copy() for key, value in expected.items()}
    actual[name][0, 0, 0] += 1e-3
    with pytest.raises(AssertionError):
        benchmark.validate_outputs(actual, expected, {'water_density': np.array([1000., 997.])})


def test_water_tolerance_has_no_absolute_floor():
    expected = outputs()
    expected['number'][:] = 0
    actual = {key: value.copy() for key, value in expected.items()}
    actual['water_mass'][0, 0, 0] = 1e-100
    with pytest.raises(AssertionError, match='water_mass'):
        benchmark.validate_outputs(actual, expected, {'water_density': np.array([1000., 997.])})


def test_checksums_include_both_native_levels():
    actual = outputs()
    timing = {name + '_checksum': actual[name].sum() for name in ('mass', 'number', 'gas')}
    checksums = benchmark._checksums(actual, timing)
    assert checksums['mass'] == 2 * 2 * 69
    timing['number_checksum'] /= 2
    with pytest.raises(AssertionError):
        benchmark._checksums(actual, timing)


@pytest.mark.parametrize('repetitions,trials', [(0, 1), (1, 0), (-1, 1), (1.5, 1), (True, 1)])
def test_invalid_timing_counts_fail_before_building_or_compiling(repetitions, trials):
    with pytest.raises(ValueError, match='positive integers'):
        benchmark.benchmark(repetitions, trials)
