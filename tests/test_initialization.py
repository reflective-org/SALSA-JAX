"""Seven-mode initializer: original Fortran parity and native population policy."""
import jax
import numpy as np
import pytest

from salsa_jax.constants import PI
from salsa_jax.grid import SizeGrid, size_grid
from salsa_jax.initialization import size_distribution
from tools.initialization_oracle import build_oracle, reference_cases, run_oracle

jax.config.update('jax_enable_x64', True)


def arguments(cases):
    diameter = cases['dry_diameter']
    middle = (PI/6.0)*diameter**3
    grid = SizeGrid(cases['lower_volume'], cases['upper_volume'], diameter,
                    cases['upper_volume']/middle, cases['lower_volume']/middle)
    return cases['mode_number'], cases['dpg'], cases['sigma'], grid


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    executable, provenance = build_oracle(tmp_path_factory.mktemp('initialization-jax'))
    cases = reference_cases()
    outputs, _ = run_oracle(executable, cases)
    return cases, outputs['number'], provenance


@pytest.mark.parametrize('compiled', [False, True])
def test_original_fortran_matrix(reference, compiled):
    cases, expected, provenance = reference
    assert provenance['checked']
    kernel = jax.jit(size_distribution) if compiled else size_distribution
    actual = kernel(*arguments(cases))
    assert actual.dtype == np.float64
    assert np.all(np.isfinite(actual))
    assert np.all(actual >= 0)
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=0)


def test_empty_and_linear_population_superposition():
    grid = size_grid()
    number = np.arange(1, 8)*1e6
    mean = np.geomspace(8e-9, 2e-6, 7)
    sigma = np.linspace(1.3, 2.1, 7)
    kernel = jax.jit(size_distribution)
    np.testing.assert_array_equal(kernel(np.zeros(7), mean, sigma, grid), 0)
    individual = kernel(np.diag(number), mean, sigma, grid)
    combined = kernel(number, mean, sigma, grid)
    np.testing.assert_allclose(individual.sum(axis=0), combined, rtol=3e-15, atol=0)
    np.testing.assert_allclose(kernel(2*number, mean, sigma, grid), 2*combined, rtol=3e-15, atol=0)
    np.testing.assert_array_equal(combined[3:10], combined[10:17])


def test_broadcasting_vmap_and_immutable_inputs(reference):
    cases, expected, _ = reference
    args = arguments(cases)
    copies = jax.tree.map(lambda a: np.array(a, copy=True), args)
    actual = jax.jit(jax.vmap(size_distribution))(*args)
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=0)
    extended = jax.tree.map(lambda a: np.asarray(a)[:, None], args)
    batched = jax.jit(size_distribution)(*extended)
    np.testing.assert_array_equal(batched[:, 0], actual)
    for before, after in zip(jax.tree.leaves(copies), jax.tree.leaves(args)):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize('index', [0, 1, 2])
def test_invalid_mode_shape(index):
    args = [np.ones(7), np.ones(7)*1e-7, np.ones(7)*1.5, size_grid()]
    args[index] = np.ones(6)
    with pytest.raises(ValueError, match='seven mode'):
        size_distribution(*args)


def test_invalid_grid_shape():
    bad = SizeGrid(*(np.ones(16) for _ in range(5)))
    with pytest.raises(ValueError, match='grid fields'):
        size_distribution(np.ones(7), np.ones(7)*1e-7, np.ones(7)*1.5, bad)
