# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Verify the native isolated/coupled nucleation oracle and its state contract."""
import numpy as np
import pytest

from tools.nucleation_process_oracle import ROOT, build_oracle, reference_cases, run_oracle


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("nucleation-process")
    executable, provenance = build_oracle(directory)
    cases = reference_cases()
    outputs, timing = run_oracle(executable, cases)
    return directory, executable, provenance, cases, outputs, timing


def test_original_module_and_actual_compiled_constants(oracle):
    directory, _, provenance, _, _, _ = oracle
    assert (directory / "nucleation_verbatim.F90").read_bytes() == (ROOT / "mo_ham_salsa_nucleation.F90").read_bytes()
    assert "-fcheck=all" in provenance["flags"]
    assert provenance["checked"]
    assert provenance["separate_compilation_without_lto"]
    assert provenance["compiled_constants"]["cluster_molecules"] == 158.79
    assert provenance["compiled_constants"]["activation_coefficient"] == 1e-7
    assert provenance["configuration"]["native_levels_per_case"] == 2
    assert provenance["configuration"]["default_species_parameters"]["sulfate_molar_mass"] == 96.0631


def test_isolated_nucleation_preserves_all_input_state(oracle):
    _, executable, _, cases, coupled, _ = oracle
    isolated, timing = run_oracle(executable, cases, operation="isolated")
    for field in ("number", "volume", "gas"):
        np.testing.assert_array_equal(isolated[field], cases[field])
    for field in ("critical_rate", "critical_molecules", "critical_diameter", "molecular_rate", "fractions"):
        np.testing.assert_array_equal(isolated[field], coupled[field])
    repeated, _ = run_oracle(executable, cases, repetitions=3, operation="isolated")
    for field in isolated:
        np.testing.assert_array_equal(repeated[field], isolated[field])
    assert timing["operation"] == "isolated"
    assert timing["kernel_wall_seconds"] > 0


def test_reset_repetitions_and_optimized_outputs(oracle, tmp_path):
    _, executable, _, cases, expected, _ = oracle
    repeated, timing = run_oracle(executable, cases, repetitions=3)
    optimized, provenance = build_oracle(tmp_path, checked=False)
    fast, _ = run_oracle(optimized, cases)
    assert not provenance["checked"]
    for field in expected:
        np.testing.assert_array_equal(repeated[field], expected[field])
        np.testing.assert_array_equal(fast[field], expected[field])
    for field in ("number", "volume", "gas"):
        np.testing.assert_allclose(timing[field + "_checksum"], expected[field].sum(), rtol=3e-15)
    assert timing["input_reset_included"]
    assert not timing["critical_capture_included"]


def test_boundary_layer_overrides_and_sulfate_fractions(oracle):
    _, _, _, cases, output, _ = oracle
    np.testing.assert_array_equal(output["fractions"], np.tile([1., 0.], (len(cases["labels"]), 1)))
    assert not output["critical_organic_molecules"].any()
    assert not output["molecular_rate"][:, 1].any()
    for scheme in (2, 3):
        inside = cases["labels"].index(f"scheme{scheme}-growth1-absent-bl")
        above = cases["labels"].index(f"scheme{scheme}-growth1-absent-above")
        assert output["critical_diameter"][inside] == output["critical_diameter"][above] == 7.9375e-10
        assert output["critical_molecules"][inside] == 2
        expected = 1e-7 * cases["gas"][inside, 1] if scheme == 2 else 5e-13 * (cases["gas"][inside, 1] * 1e-6)**2 * 1e6
        np.testing.assert_allclose(output["critical_rate"][inside], expected, rtol=2e-15)
        binary = cases["labels"].index("scheme1-growth1-absent-above")
        assert output["critical_rate"][above] == output["critical_rate"][binary]


def test_critical_cutoff_and_optional_organic_mapping(oracle):
    _, _, _, cases, outputs, _ = oracle
    indices = [i for i, label in enumerate(cases["labels"]) if label.startswith("activation-cutoff")]
    for i in indices:
        if outputs["critical_rate"][i] < 1e3:
            assert not outputs["molecular_rate"][i].any()
        else:
            assert outputs["molecular_rate"][i, 0] > 0
    absent = ~cases["organic_enabled"].astype(bool)
    np.testing.assert_array_equal(outputs["gas"][absent, 2], cases["gas"][absent, 2])
    np.testing.assert_array_equal(outputs["volume"][absent, :, 1], cases["volume"][absent, :, 1])


def test_coupled_gas_material_budgets_and_only_first_bin_number_growth(oracle):
    _, _, provenance, cases, output, _ = oracle
    np.testing.assert_array_equal(output["number"][:, 1:], cases["number"][:, 1:])
    assert np.all(output["number"][:, 0] >= cases["number"][:, 0])
    assert np.any(output["number"][:, 0] > cases["number"][:, 0])
    np.testing.assert_array_equal(output["gas"][:, 0], cases["gas"][:, 0])
    for material, name in enumerate(("sulfate", "organic")):
        molecular_volume = (cases[name + "_molar_mass"] / 1000 /
                            provenance["compiled_constants"]["avogadro"] / cases[name + "_density"])
        before = cases["gas"][:, material + 1] + cases["volume"][:, :, material].sum(axis=1) / molecular_volume
        after = output["gas"][:, material + 1] + output["volume"][:, :, material].sum(axis=1) / molecular_volume
        np.testing.assert_allclose(after, before, rtol=3e-14, atol=0)


@pytest.mark.parametrize("field,value", [("scheme", 4), ("growth", 0), ("timestep", 0),
                                         ("organic_enabled", 2), ("boundary_layer", 2),
                                         ("relative_humidity", -1)])
def test_unsupported_inputs_fail_before_execution(field, value):
    cases = reference_cases()
    invalid = np.full_like(cases[field], value, dtype=float)
    with pytest.raises(ValueError, match=field):
        run_oracle("must-not-run", {**cases, field: invalid})


def test_active_binary_and_seeded_growth_matrix_has_nonzero_sources(oracle):
    _, _, _, cases, output, _ = oracle
    for scheme in (1, 2, 3):
        for growth in (1, 2, 3):
            indices = [i for i, label in enumerate(cases["labels"])
                       if label.startswith("active-binary-") and
                       label.endswith(f"scheme{scheme}-growth{growth}")]
            assert len(indices) == 3
            assert np.all(output["critical_rate"][indices] > 1e3)
            assert np.all(output["molecular_rate"][indices, 0] > 1e3)
            assert np.all(output["number"][indices, 0] > cases["number"][indices, 0])
    seeded = [i for i, label in enumerate(cases["labels"]) if label.startswith("seeded-growth3-")]
    assert len(seeded) == 6
    assert np.all(output["molecular_rate"][seeded, 0] > 0)
