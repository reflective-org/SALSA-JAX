"""H01 leaf parity with actual gas2mmr arithmetic, mapping and mutation order."""
import hashlib

import jax
import numpy as np
import pytest

from salsa_jax.constants import AVOGADRO, GAS_MOLECULAR_CONVERSION_FACTOR
from salsa_jax.gas_conversion import gas_to_tracers
from tools.gas_conversion_oracle import build_oracle, gas_routine, run_oracle, unit_declarations

jax.config.update("jax_enable_x64", True)
INPUTS = ("tracers", "gas", "air_density", "pressure", "temperature",
          "tracer_indices", "molar_mass", "unit_codes")


def cases():
    return {
        "tracers": np.arange(28, dtype=float).reshape(4, 7) + 0.123,
        "gas": np.array([[4., 1e8, 2e9, 17., -3e6], [0., 1e11, 3e8, 5., 8e4],
                         [8e6, 9e6, 1e6, 2e6, 3e6], [1., 2., 3., 4., 5.]]),
        "air_density": np.array([1.2, .4, 2.7, .85]),
        "pressure": np.array([101325., 45000., 180000., 73000.]),
        "temperature": np.array([298., 240., 315., 271.]),
        "tracer_indices": np.array([[3, 0, 5, 2, 1], [5, 3, 1, 0, 2],
                                    [2, 2, 2, 4, 4], [0, 1, 2, 3, 4]]),
        "molar_mass": np.array([[64., 98., 44., 180., 18.], [12., 96.0631, 64., 58., 35.],
                                [10., 20., 30., 40., 50.], [64., 96., 180., 18., 28.]]),
        "unit_codes": np.array([[1, 2, 3, 0, 2], [2, 3, 1, 99, 2],
                               [1, 2, 3, 2, 99], [0, -1, 4, 99, 0]]),
    }


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    directory = tmp_path_factory.mktemp("gas-conversion")
    executable, provenance = build_oracle(directory)
    c = cases()
    return directory, executable, provenance, c, run_oracle(executable, c)


def evaluate(c, compiled=True):
    return (jax.jit(gas_to_tracers) if compiled else gas_to_tracers)(*(c[key] for key in INPUTS))


def test_verbatim_checked_reference_and_native_code_declarations(reference):
    directory, _, provenance, _, _ = reference
    assert (directory / "gas2mmr_verbatim.inc").read_text() == gas_routine() + "\n"
    assert provenance["routine_sha256"] == hashlib.sha256(gas_routine().encode()).hexdigest()
    assert unit_declarations() in (directory / "stubs.f90").read_text()
    assert "-fcheck=all" in provenance["flags"]
    assert "-ffpe-trap=invalid,zero,overflow" in provenance["flags"]
    assert "1e-3*prhoa" in gas_routine()


@pytest.mark.parametrize("compiled", [False, True])
def test_all_conversion_codes_and_nonunity_environment_match_native(reference, compiled):
    _, _, _, c, expected = reference
    before = {key: value.copy() for key, value in c.items()}
    actual = np.asarray(evaluate(c, compiled))
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected["tracers"], rtol=3e-15, atol=0)
    np.testing.assert_array_equal(expected["inactive_padding"], -4321.)
    for key in c:
        np.testing.assert_array_equal(c[key], before[key])


def test_duplicate_mapping_last_known_write_and_unknown_noop(reference):
    _, _, _, c, expected = reference
    actual = np.asarray(evaluate(c))
    np.testing.assert_array_equal(actual[3], c["tracers"][3])
    # Three writes to tracer2; the final code3 controls the native result.
    single = {key: value[2:3].copy() for key, value in c.items()}
    single["unit_codes"][0, :2] = 0
    np.testing.assert_array_equal(evaluate(single)[0, 2], actual[2, 2])
    # Last gas maps tracer4 with unknown code99, preserving the previous write.
    np.testing.assert_allclose(actual[2, 4], expected["tracers"][2, 4], rtol=3e-15, atol=0)
    for row in range(4):
        selected = set(c["tracer_indices"][row, np.isin(c["unit_codes"][row], [1, 2, 3])])
        untouched = [index for index in range(7) if index not in selected]
        np.testing.assert_array_equal(actual[row, untouched], c["tracers"][row, untouched])


def test_binary32_literal_promotion_is_numerically_observable(reference):
    _, _, _, c, expected = reference
    assert GAS_MOLECULAR_CONVERSION_FACTOR == float(np.float32(1e-3))
    rho, gas, mass = c["air_density"][0], c["gas"][0, 1], c["molar_mass"][0, 1]
    native = gas / (float(np.float32(1e-3)) * rho * AVOGADRO / mass)
    rewritten = gas / (1e-3 * rho * AVOGADRO / mass)
    np.testing.assert_allclose(expected["tracers"][0, 0], native, rtol=3e-16, atol=0)
    assert abs(native / rewritten - 1) > 4e-8


def test_scalar_multiaxis_and_broadcast_registry(reference):
    _, _, _, c, expected = reference
    one = {key: value[0] for key, value in c.items()}
    np.testing.assert_allclose(evaluate(one), expected["tracers"][0], rtol=3e-15, atol=0)
    many = {key: np.broadcast_to(value, (2, 3) + value.shape) for key, value in one.items()}
    # Registry is shared over cells; all other arrays keep explicit cell axes.
    for key in ("tracer_indices", "molar_mass", "unit_codes"):
        many[key] = one[key]
    actual = evaluate(many)
    assert actual.shape == (2, 3, 7)
    np.testing.assert_allclose(actual, np.broadcast_to(expected["tracers"][0], actual.shape), rtol=3e-15, atol=0)


@pytest.mark.parametrize("field", ["tracer_indices", "unit_codes"])
def test_noninteger_mapping_rejected(field):
    c = cases()
    c[field] = c[field].astype(float)
    with pytest.raises(ValueError, match="integer"):
        evaluate(c)


def test_mismatched_case_axes_rejected():
    c = cases()
    c["gas"] = c["gas"][:1]
    with pytest.raises(ValueError, match="matching case axes"):
        evaluate(c)


@pytest.mark.parametrize("change,message", [("index", "Out-of-range"), ("density", "Nonpositive"),
                                            ("mass", "Nonpositive"), ("nonfinite", "Nonfinite")])
def test_reference_rejects_invalid_host_inputs_before_execution(change, message):
    c = cases()
    if change == "index":
        c["tracer_indices"][0, 0] = 7
    elif change == "density":
        c["air_density"][0] = 0
    elif change == "mass":
        c["molar_mass"][0, 0] = 0
    else:
        c["gas"][0, 0] = np.nan
    with pytest.raises(ValueError, match=message):
        run_oracle("must-not-run", c)


def test_empty_gas_registry_leaves_all_tracers_unchanged():
    tracers = np.arange(7, dtype=float)
    actual = jax.jit(gas_to_tracers)(tracers, np.empty(0), 1.2, 100000., 280.,
                                   np.empty(0, dtype=int), np.empty(0), np.empty(0, dtype=int))
    np.testing.assert_array_equal(actual, tracers)
