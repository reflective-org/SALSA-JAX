# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Public initial cleanup versus an independently compiled actual native block."""
import hashlib
import json
import subprocess

import jax
import numpy as np
import pytest

from salsa_jax.constants import AVOGADRO, PI
from salsa_jax.interface import mass_to_volume
from salsa_jax.processes.hydration import HydrationParameters
from salsa_jax.processes.remapping_correction import cleanup
from tools.cleanup_oracle import ROOT, cleanup_block, normalized_block, build_oracle, run_oracle
from tools.remapping_correction_oracle import correction_block
from tools.salsa_fixture import load_native_case
from tools.hydration_fixture import load_native_case as load_hydration

jax.config.update("jax_enable_x64", True)


def cases():
    native, _, _ = load_native_case()
    row = {key: value[0].copy() for key, value in native.items() if key != "labels"}
    parameters = HydrationParameters(species_molar_mass=row["species_molar_mass"],
                                     species_density=row["species_density"])
    row["volume"] = np.asarray(mass_to_volume(row["mass"], row["component_map"], parameters)).copy()
    fields = ("number", "volume", "gas", "species_molar_mass", "species_density", "number_threshold", "organic_enabled")
    base = {key: row[key] for key in fields}
    rows = [base]
    labels = ["native-default"]
    for bin_index in (0, 2, 3, 9, 10, 16):
        for diameter in (0.999999e-10, 1e-10, 1.000001e-10):
            case = {key: value.copy() for key, value in base.items()}
            case["number"] = np.zeros(17)
            case["volume"] = np.zeros((17, 5))
            case["number"][bin_index] = 1e6
            case["volume"][bin_index, :2] = 1e6 * (PI / 6) * diameter**3 * np.array([.6, .4])
            case["organic_enabled"] = np.array(True)
            rows.append(case)
            labels.append(f"tiny-{bin_index}-{diameter}")
    for number in (np.nextafter(1., 0.), 1., np.nextafter(1., np.inf)):
        case = {key: value.copy() for key, value in rows[1].items()}
        case["number"][0] = number
        case["volume"][0, :2] = number * (PI / 6) * 1e-33 * np.array([.6, .4])
        rows.append(case)
        labels.append(f"threshold-{number:.17g}")
    case = {key: value.copy() for key, value in rows[1].items()}
    case["number"][:] = 1e6
    case["volume"][:] = 0
    case["volume"][:, 2] = 1e-30
    rows.append(case)
    labels.append("inert-loss")
    return {"labels": labels, **{key: np.stack([row[key] for row in rows]) for key in fields}}


def run_jax(c, compiled=True):
    parameters = HydrationParameters(species_molar_mass=c["species_molar_mass"],
                                     species_density=c["species_density"], number_threshold=c["number_threshold"])
    return (jax.jit(cleanup) if compiled else cleanup)(c["number"], c["volume"], c["gas"],
                     parameters=parameters, organic_enabled=c["organic_enabled"])


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    directory = tmp_path_factory.mktemp("initial-cleanup")
    executable, provenance = build_oracle(directory)
    c = cases()
    expected = run_oracle(executable, c)
    return directory, executable, provenance, c, expected


def test_verbatim_initial_block_and_equivalence_to_postremap(reference):
    directory, _, provenance, _, _ = reference
    block = cleanup_block()
    assert (directory / "cleanup_verbatim.inc").read_text() == block + "\n"
    assert provenance["cleanup_block_sha256"] == hashlib.sha256(block.encode()).hexdigest()
    assert "-fcheck=all" in provenance["flags"]
    assert "-ffpe-trap=invalid,zero,overflow" in provenance["flags"]
    post = correction_block().split("       CALL equilibration", 1)[0]
    assert normalized_block(block) == normalized_block(post)
    assert "CALL equilibration" not in block
    assert "pc_gas(ii,jj,isubm_ocnv)" in block


@pytest.mark.parametrize("compiled", [False, True])
def test_registered_cleanup_matches_native(reference, compiled):
    _, _, _, c, expected = reference
    actual = run_jax(c, compiled)
    assert np.all(actual.valid)
    for name in ("number", "volume", "gas"):
        np.testing.assert_allclose(getattr(actual, name), expected[name], rtol=3e-13, atol=0)
    below = c["labels"].index(f"threshold-{np.nextafter(1.,0.):.17g}")
    equal = c["labels"].index("threshold-1")
    assert expected["number"][below, 0] > 0
    assert expected["number"][equal, 0] == 0
    loss = c["labels"].index("inert-loss")
    assert np.sum(c["volume"][loss, :, 2]) > 0
    assert not expected["volume"][loss].any()
    np.testing.assert_array_equal(expected["gas"][loss], c["gas"][loss])
    for material, gas_slot in ((0, 1), (1, 2)):
        factor = c["species_density"][:, material] / (c["species_molar_mass"][:, material] / 1000) * AVOGADRO
        before = c["gas"][:, gas_slot] + c["volume"][:, :, material].sum(-1) * factor
        after = expected["gas"][:, gas_slot] + expected["volume"][:, :, material].sum(-1) * factor
        np.testing.assert_allclose(after, before, rtol=3e-15, atol=0)


def test_actual_initial_hydration_boundary(reference):
    _, _, _, _, expected = reference
    hydration, _, _ = load_hydration()
    native_manifest = json.loads((ROOT / "tests/fixtures/native-default/manifest.json").read_text())
    hydration_manifest = json.loads((ROOT / "tests/fixtures/hydration-default/manifest.json").read_text())
    assert native_manifest["input_sha256"] == hydration_manifest["input_sha256"]
    for name in ("mo_ham_salsa.F90", "mo_ham_salsa_init.F90", "mo_ham_species.F90"):
        assert native_manifest["source_sha256"][name] == hydration_manifest["source_sha256"][name]
    np.testing.assert_array_equal(expected["number"][:1], hydration["number"])
    np.testing.assert_allclose(expected["volume"][:1], hydration["volume"], rtol=3e-15, atol=0)


def test_D007_native_abort_and_invalid_cell_isolation(reference):
    _, executable, _, c, _ = reference
    bad = {key: value[1:2].copy() for key, value in c.items() if key != "labels"}
    bad["organic_enabled"][:] = False
    with pytest.raises(subprocess.CalledProcessError) as failure:
        run_oracle(executable, bad)
    assert "below lower bound of 1" in failure.value.stderr
    mixed = {key: np.concatenate((value[:1], bad[key]), axis=0) for key, value in c.items() if key != "labels"}
    result = run_jax(mixed)
    np.testing.assert_array_equal(result.valid, [True, False])
    for name in ("number", "volume", "gas"):
        assert np.all(np.isfinite(np.asarray(getattr(result, name))[0]))
        assert np.all(np.isnan(np.asarray(getattr(result, name))[1]))
        np.testing.assert_array_equal(np.asarray(getattr(result, name))[0], mixed[name][0])


def test_scalar_cleanup_and_input_immutability(reference):
    _, _, _, c, expected = reference
    one = {key: value[1].copy() for key, value in c.items() if key != "labels"}
    before = {key: value.copy() for key, value in one.items()}
    result = run_jax(one)
    assert result.valid.shape == ()
    for name in ("number", "volume", "gas"):
        np.testing.assert_allclose(getattr(result, name), expected[name][1], rtol=3e-13, atol=0)
    for name in one:
        np.testing.assert_array_equal(one[name], before[name])
