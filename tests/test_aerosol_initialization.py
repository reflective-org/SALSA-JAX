"""SALSA composition initialization preserves native mixed units and write order."""
import jax
import numpy as np
import pytest

from salsa_jax.interface import mass_to_volume, validate_component_map
from salsa_jax.processes.hydration import HydrationParameters
from salsa_jax.aerosol_initialization import initialize_aerosol
from tools.aerosol_initialization_oracle import build_oracle, reference_cases, run_oracle

jax.config.update("jax_enable_x64", True)
INPUTS = ("number", "core", "tracers", "air_density", "component_map",
          "component_tracer_indices", "number_tracer_indices", "component_unit_codes")


def parameters(cases):
    return HydrationParameters(species_molar_mass=cases["species_molar_mass"],
                               species_density=cases["species_density"])


def evaluate(cases, compiled=True):
    kernel = jax.jit(initialize_aerosol) if compiled else initialize_aerosol
    return kernel(*(cases[key] for key in INPUTS), parameters=parameters(cases), mw_so4=cases["mw_so4"])


def subset(cases, index=0):
    return {key: value[index:index+1].copy() for key, value in cases.items() if key != "labels"}


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    executable, provenance = build_oracle(tmp_path_factory.mktemp("aerosol-initialization-jax"), checked=True)
    cases = reference_cases()
    expected, _ = run_oracle(executable, cases)
    return executable, provenance, cases, expected


@pytest.mark.parametrize("compiled", [False, True])
def test_all_curated_cases_match_actual_conc2mmr(reference, compiled):
    _, provenance, cases, expected = reference
    assert provenance["checked"]
    before = {key: value.copy() for key, value in cases.items() if key != "labels"}
    actual = evaluate(cases, compiled)
    for field in ("number", "mass", "tracers"):
        values = np.asarray(getattr(actual, field))
        assert values.dtype == np.float64
        assert np.all(np.isfinite(values))
        np.testing.assert_allclose(values, expected[field], rtol=4e-15, atol=0)
    for key, value in before.items():
        np.testing.assert_array_equal(cases[key], value)


def test_physical_volume_and_region_material_budgets(reference):
    _, _, cases, _ = reference
    one = subset(cases)
    actual = evaluate(one)
    decoded = np.asarray(mass_to_volume(actual.mass, one["component_map"], parameters(one)))
    expected_total = one["number"] * one["core"]
    fractions = np.zeros((17, 5))
    fractions[:3, :2] = [.2, .8]
    fractions[3:10] = [.5, .05, .05, .1, .3]
    fractions[10:] = [.01, .01, .01, 0, .97]
    np.testing.assert_allclose(decoded.sum(-1), expected_total, rtol=1e-15, atol=0)
    np.testing.assert_allclose(decoded, expected_total[..., None] * fractions, rtol=1e-15, atol=0)
    np.testing.assert_array_equal(actual.number, one["number"] * 1e-6)


def test_initialization_is_not_idempotent_and_returns_number_cm3(reference):
    executable, _, cases, _ = reference
    one = subset(cases)
    expected, _ = run_oracle(executable, one, steps=2)
    first = evaluate(one)
    second_input = {**one, "number": np.asarray(first.number), "tracers": np.asarray(first.tracers)}
    second = evaluate(second_input)
    for name in ("number", "mass", "tracers"):
        np.testing.assert_allclose(getattr(second, name), expected[name], rtol=4e-15, atol=0)
    np.testing.assert_array_equal(second.number, one["number"] * 1e-6 * 1e-6)
    assert not np.array_equal(first.mass, second.mass)


def test_permuted_component_registry_preserves_bin_material_meaning(reference):
    executable, _, cases, _ = reference
    one = subset(cases)
    original = evaluate(one)
    changed = {key: value.copy() for key, value in one.items()}
    # Reindex all component arrays consistently while retaining material/bin slots.
    old_map = changed["component_map"]
    changed["component_map"] = np.where(old_map >= 0, 68 - old_map, -1)
    changed["component_tracer_indices"] = changed["component_tracer_indices"][:, ::-1]
    changed["component_unit_codes"] = changed["component_unit_codes"][:, ::-1]
    validate_component_map(changed["component_map"])
    expected, _ = run_oracle(executable, changed)
    actual = evaluate(changed)
    np.testing.assert_allclose(actual.mass, np.asarray(original.mass)[:, ::-1], rtol=4e-15, atol=0)
    np.testing.assert_allclose(actual.tracers, original.tracers, rtol=4e-15, atol=0)
    for name in ("number", "mass", "tracers"):
        np.testing.assert_allclose(getattr(actual, name), expected[name], rtol=4e-15, atol=0)


def test_component_then_number_writes_win_duplicate_tracer_collisions(reference):
    executable, _, cases, _ = reference
    changed = subset(cases)
    changed["tracers"][:] = np.arange(changed["tracers"].shape[-1]) + 123.
    changed["component_tracer_indices"][:] = 0
    changed["number_tracer_indices"][:] = 0
    expected, _ = run_oracle(executable, changed)
    actual = evaluate(changed)
    np.testing.assert_allclose(actual.tracers, expected["tracers"], rtol=4e-15, atol=0)
    np.testing.assert_array_equal(np.asarray(actual.tracers)[:, 1:], changed["tracers"][:, 1:])
    # Last bin overwrites the same tracer after all aerosol-mass writes.
    expected_last = (changed["number"][:, -1] * 1e-6) / (1e-6 * changed["air_density"])
    np.testing.assert_allclose(np.asarray(actual.tracers)[:, 0], expected_last, rtol=3e-16, atol=0)


def test_unknown_component_codes_follow_native_mass_else_branch(reference):
    executable, _, cases, _ = reference
    standard = subset(cases)
    standard["component_unit_codes"][:] = 1
    unknown = {key: value.copy() for key, value in standard.items()}
    unknown["component_unit_codes"][:] = 99
    actual, original = evaluate(unknown), evaluate(standard)
    np.testing.assert_array_equal(actual.tracers, original.tracers)
    expected, _ = run_oracle(executable, unknown)
    np.testing.assert_allclose(actual.tracers, expected["tracers"], rtol=4e-15, atol=0)


def test_registry_sulfate_mass_and_legacy_mw_so4_are_independent(reference):
    executable, _, cases, _ = reference
    original = subset(cases)
    changed = {key: value.copy() for key, value in original.items()}
    changed["mw_so4"] *= 1.5
    before, actual = evaluate(original), evaluate(changed)
    # mw_so4 is a tracer factor only; registry molecular volume stays fixed.
    np.testing.assert_array_equal(actual.mass, before.mass)
    selected = changed["component_tracer_indices"][changed["component_unit_codes"] == 2]
    assert selected.size > 0
    assert np.any(np.asarray(actual.tracers)[0, selected] != np.asarray(before.tracers)[0, selected])
    expected, _ = run_oracle(executable, changed)
    np.testing.assert_allclose(actual.tracers, expected["tracers"], rtol=4e-15, atol=0)
    changed = {key: value.copy() for key, value in original.items()}
    changed["species_molar_mass"][:, 0] *= 1.25
    registry = evaluate(changed)
    sulfate_components = changed["component_map"][0, :, 0]
    np.testing.assert_allclose(np.asarray(registry.mass)[0, sulfate_components],
                               np.asarray(before.mass)[0, sulfate_components] / 1.25, rtol=4e-15, atol=0)


def test_scalar_and_multiaxis_case_broadcasting(reference):
    _, _, cases, expected = reference
    scalar = {key: value[0] for key, value in cases.items() if key != "labels"}
    one = evaluate(scalar)
    for name in ("number", "mass", "tracers"):
        np.testing.assert_allclose(getattr(one, name), expected[name][0], rtol=4e-15, atol=0)
    multiple = {key: np.broadcast_to(value, (2, 3) + np.shape(value)) for key, value in scalar.items()}
    for key in ("core", "component_map", "component_tracer_indices", "number_tracer_indices", "component_unit_codes"):
        multiple[key] = scalar[key]
    actual = evaluate(multiple)
    assert actual.number.shape == (2, 3, 17)
    for name in ("number", "mass", "tracers"):
        np.testing.assert_allclose(getattr(actual, name),
                                   np.broadcast_to(getattr(one, name), np.shape(getattr(actual, name))),
                                   rtol=4e-15, atol=0)
