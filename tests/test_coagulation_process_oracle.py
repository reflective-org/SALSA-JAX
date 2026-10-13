# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Check the complete-process oracle's extraction, I/O and repetition semantics."""
import hashlib

import numpy as np
import pytest

from tools.coagulation_process_oracle import (
    ROOT, _extract, build_oracle, reference_cases, run_oracle,
)


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("coagulation-process")
    executable, provenance = build_oracle(directory)
    cases = reference_cases()
    outputs, timing = run_oracle(executable, cases)
    return directory, executable, provenance, cases, outputs, timing


def test_original_extraction_and_checked_build(oracle):
    directory, _, provenance, _, _, _ = oracle
    expected = _extract((ROOT / "mo_ham_salsa_dynamics.F90").read_text(), "SUBROUTINE", "coagulation")
    assert (directory / "coagulation_verbatim.inc").read_text() == expected + "\n"
    assert provenance["process_sha256"] == hashlib.sha256(expected.encode()).hexdigest()
    assert "-fcheck=all" in provenance["flags"]
    assert "-ffpe-trap=invalid,zero,overflow" in provenance["flags"]
    assert provenance["separate_compilation_without_lto"]
    assert provenance["configuration"]["HAMMOZ"] is False
    assert provenance["configuration"]["nsoa"] == 0


def test_repetitions_reset_inputs_and_do_not_advance_a_trajectory(oracle):
    _, executable, _, cases, expected, timing = oracle
    repeated, repeated_timing = run_oracle(executable, cases, repetitions=3)
    for name in expected:
        np.testing.assert_array_equal(repeated[name], expected[name])
    assert timing["kernel_wall_seconds"] > 0
    assert repeated_timing["kernel_wall_seconds"] > 0
    assert repeated_timing["input_reset_included"]
    assert not repeated_timing["coefficient_capture_included"]
    np.testing.assert_allclose(repeated_timing["number_checksum"], repeated["number"].sum(), rtol=2e-15)
    np.testing.assert_allclose(repeated_timing["volume_checksum"], repeated["volume"].sum(), rtol=2e-15)


def test_empty_zero_step_and_self_collision(oracle):
    _, _, _, cases, outputs, _ = oracle
    empty = cases["labels"].index("empty")
    assert not outputs["number"][empty].any()
    assert not outputs["volume"][empty].any()
    zero_step = cases["labels"].index("mixed-zero-timestep")
    # This case starts with no sea salt in 2b: the unconditional salt-to-sulfate
    # relabel therefore leaves volumes unchanged even at dt=0.
    np.testing.assert_array_equal(outputs["number"][zero_step], cases["number"][zero_step])
    np.testing.assert_array_equal(outputs["volume"][zero_step], cases["volume"][zero_step])
    for bin_index in range(17):
        case = cases["labels"].index(f"single-bin-{bin_index + 1:02d}")
        initial = cases["number"][case, bin_index]
        expected = initial / (1 + 0.5 * cases["timestep"][case] *
                              outputs["coefficients"][case, bin_index, bin_index] * initial)
        np.testing.assert_allclose(outputs["number"][case, bin_index], expected, rtol=2e-15)
        np.testing.assert_array_equal(outputs["volume"][case], cases["volume"][case])
    np.testing.assert_array_equal(outputs["coefficients"], outputs["coefficients"].swapaxes(1, 2))
    assert np.all(outputs["coefficients"] > 0)


def test_shared_dry_geometry_input(oracle):
    _, executable, _, cases, expected, _ = oracle
    shared = {**cases, "dry_diameter": cases["dry_diameter"][0]}
    actual, _ = run_oracle(executable, shared)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize("field,value", [
    ("number", np.empty((0, 17))), ("volume", np.zeros((31, 17))),
    ("wet_diameter", np.zeros((31, 17))), ("temperature", np.full(31, np.nan)),
    ("timestep", -np.ones(31)),
])
def test_invalid_inputs_rejected_before_execution(field, value):
    with pytest.raises(ValueError, match=field):
        run_oracle("must-not-execute", {**reference_cases(), field: value})


@pytest.mark.parametrize("repetitions", [0, -1, 1.5])
def test_invalid_repetitions(repetitions):
    with pytest.raises(ValueError, match="repetitions"):
        run_oracle("must-not-execute", reference_cases(), repetitions=repetitions)


def test_legacy_cross_regime_volume_creation_is_visible(oracle):
    # D009: 2a gains see the old 2b volume; subsequent 2b losses use the
    # already-reduced 2a number. The native scheme does not conserve volume
    # under stiff cross-regime collisions. This regression exposes that
    # limitation instead of treating matching Fortran as physical acceptance.
    _, _, _, cases, outputs, _ = oracle
    index = cases["labels"].index("cross-regime-budget-stress")
    number = cases["number"][index, 3]
    initial_volume = cases["volume"][index, 3, 0]
    timestep = cases["timestep"][index]
    cross = outputs["coefficients"][index, 3, 10]
    self_coefficient = outputs["coefficients"][index, 3, 3]
    updated_number = number / (1 + 0.5 * timestep * self_coefficient * number)
    expected_2a = initial_volume * (1 + timestep * cross * number)
    expected_2b = initial_volume / (1 + timestep * cross * updated_number)
    np.testing.assert_allclose(outputs["volume"][index, [3, 10], 0],
                               [expected_2a, expected_2b], rtol=2e-15, atol=0)
    relative_gain = outputs["volume"][index].sum() / cases["volume"][index].sum() - 1
    assert relative_gain > 2.0


def test_optimized_build_preserves_checked_results(oracle, tmp_path):
    _, _, _, cases, expected, _ = oracle
    executable, provenance = build_oracle(tmp_path, checked=False)
    actual, _ = run_oracle(executable, cases)
    assert not provenance["checked"]
    assert not any(flag.startswith(("-fcheck", "-ffpe-trap", "-fbacktrace")) for flag in provenance["flags"])
    assert "-fno-fast-math" in provenance["flags"]
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
