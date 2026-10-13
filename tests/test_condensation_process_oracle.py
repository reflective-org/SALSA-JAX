"""Independent checks on native condensation extraction and oracle semantics."""
import hashlib

import numpy as np
import pytest

from tools.condensation_process_oracle import ROOT, _extract, build_oracle, reference_cases, run_oracle
from tools.state_capture import load_fixture


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("condensation-process")
    executable, provenance = build_oracle(directory)
    cases = reference_cases()
    outputs, timing = run_oracle(executable, cases)
    return directory, executable, provenance, cases, outputs, timing


def test_native_extraction_constants_and_checked_build(oracle):
    directory, _, provenance, cases, _, _ = oracle
    expected = _extract((ROOT / "mo_ham_salsa_dynamics.F90").read_text(), "SUBROUTINE", "condensation")
    assert (directory / "condensation_verbatim.inc").read_text() == expected + "\n"
    assert provenance["process_sha256"] == hashlib.sha256(expected.encode()).hexdigest()
    assert "-fcheck=all" in provenance["flags"]
    assert provenance["configuration"]["nucleation_stub_aborts"]
    assert provenance["configuration"]["nsnucl"] == 0
    assert provenance["configuration"]["nsoa"] == 0
    constants = provenance["compiled_constants"]
    assert constants["avogadro"] == 6.02214179e23
    assert constants["gas_constant"] == 8.314472
    assert constants["molecule_diameter"] == 5.539376964394570e-10
    fixture = load_fixture(ROOT / "tests/fixtures/native-default")["entry"]
    species = fixture["volume_species"]["values"]
    for name, material in (("sulfate", 0), ("organic", 1)):
        index = species[material] - 1
        assert cases[f"{name}_molar_mass"][0] == fixture["species_molar_mass"]["values"][index]
        assert cases[f"{name}_density"][0] == fixture["species_density"]["values"][index]


def test_repetitions_and_optimized_build_are_identical(oracle, tmp_path):
    _, executable, _, cases, expected, _ = oracle
    repeated, timing = run_oracle(executable, cases, repetitions=4)
    optimized, provenance = build_oracle(tmp_path, checked=False)
    fast, _ = run_oracle(optimized, cases)
    assert not provenance["checked"]
    assert "-fcheck=all" not in provenance["flags"]
    for field in expected:
        np.testing.assert_array_equal(repeated[field], expected[field])
        np.testing.assert_array_equal(fast[field], expected[field])
    for field in ("number", "volume", "gas"):
        np.testing.assert_allclose(timing[f"{field}_checksum"], expected[field].sum(), rtol=2e-15)
    assert timing["kernel_wall_seconds"] > 0
    assert timing["input_reset_included"]
    assert not timing["rate_capture_included"]


def test_inactive_paths_do_not_change_state(oracle):
    _, _, _, cases, outputs, _ = oracle
    np.testing.assert_array_equal(outputs["number"], cases["number"])
    np.testing.assert_array_equal(outputs["gas"][:, 0], cases["gas"][:, 0])
    np.testing.assert_array_equal(outputs["volume"][:, :, 2:], cases["volume"][:, :, 2:])
    for label in ("empty", "zero-vapors", "zero-timestep", "gas-below-threshold", "gas-at-threshold", "sink-below-threshold"):
        index = cases["labels"].index(label)
        np.testing.assert_array_equal(outputs["gas"][index], cases["gas"][index])
        np.testing.assert_array_equal(outputs["volume"][index], cases["volume"][index])
    absent = cases["labels"].index("absent-organic")
    assert cases["gas"][absent, 2] > 0  # positive canonical sentinel must stay inactive
    assert outputs["gas"][absent, 2] == cases["gas"][absent, 2]
    np.testing.assert_array_equal(outputs["volume"][absent, :, 1], cases["volume"][absent, :, 1])


def test_thresholds_and_molecular_budgets(oracle):
    _, _, provenance, cases, outputs, _ = oracle
    for label, expected_sink in (("sink-below-threshold", .99e-30), ("sink-above-threshold", 1.01e-30)):
        index = cases["labels"].index(label)
        np.testing.assert_allclose(outputs["rates"][index].sum(), expected_sink, rtol=3e-15, atol=0)
    for label in ("gas-above-threshold", "sink-above-threshold"):
        index = cases["labels"].index(label)
        assert np.all(outputs["gas"][index, 1:] < cases["gas"][index, 1:])
    index = cases["labels"].index("zero-initial-condensate")
    for material, name in enumerate(("sulfate", "organic")):
        molecular_volume = (cases[f"{name}_molar_mass"][index] / 1000 / provenance["compiled_constants"]["avogadro"] /
                            cases[f"{name}_density"][index])
        gained = outputs["volume"][index, :, material].sum() / molecular_volume
        lost = cases["gas"][index, material + 1] - outputs["gas"][index, material + 1]
        assert gained > 0
        np.testing.assert_allclose(gained, lost, rtol=3e-15, atol=0)


def test_first_two_bins_use_molecule_size_and_particle_diffusion(oracle):
    _, _, _, cases, outputs, _ = oracle
    index = cases["labels"].index("equal-small-diameters")
    rate = outputs["rates"][index]
    np.testing.assert_array_equal(rate[0], rate[1])
    np.testing.assert_array_equal(rate[2:], np.full(15, rate[2]))
    assert rate[0] > rate[2]


@pytest.mark.parametrize("field", ["number", "volume", "gas", "wet_diameter", "mass_accommodation",
                                    "temperature", "pressure", "timestep", "organic_enabled"])
def test_invalid_input_rejected(field):
    cases = reference_cases()
    bad = np.full_like(cases[field], -1, dtype=float)
    with pytest.raises(ValueError, match=field):
        run_oracle("must-not-run", {**cases, field: bad})
