# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Complete native boundary adaptation keeps registry, flags and cell ordering."""
import json
import shutil

import numpy as np
import pytest

from tools.salsa_fixture import ROOT, adapt_capture, load_native_case
from tools.state_capture import load_fixture

FIXTURE = ROOT / "tests/fixtures/native-default"


def pair():
    return load_fixture(FIXTURE), json.loads((FIXTURE / "manifest.json").read_text())


def test_default_registry_state_and_units():
    cases, expected, metadata = load_native_case()
    assert cases["mass"].shape == (1, 69)
    assert cases["component_map"].shape == (1, 17, 5)
    mapping = cases["component_map"][0]
    np.testing.assert_array_equal(np.sort(mapping[mapping >= 0]), np.arange(69))
    np.testing.assert_array_equal(mapping[:3, 2:], -1)
    np.testing.assert_array_equal(mapping[10:, 3], -1)
    np.testing.assert_array_equal(cases["gas"][:, 2], 0)
    np.testing.assert_array_equal(expected["gas"][:, 2], 0)
    np.testing.assert_array_equal(cases["species_molar_mass"][0], [96.0631, 180, 12.01, 58.443, 250])
    assert cases["water_molar_mass"][0] == 18
    assert cases["water_density"][0] == 1000
    assert cases["boundary_layer"].tolist() == [True]
    assert cases["scheme"].tolist() == [0]
    assert not cases["organic_enabled"].any()
    assert not cases["recalculate"].any()
    assert set(expected) == {"number", "mass", "gas", "volume", "core", "wet_diameter",
                             "dry_radius_cm", "wet_radius_cm", "water_mass", "density"}
    assert metadata["density_units"] == "g cm-3"
    assert metadata["input_sha256"]
    assert metadata["run_manifest_sha256"]


@pytest.mark.parametrize("case,message", [
    ("source", "source"), ("inputs", "provenance"), ("unchecked", "checked"),
    ("manifest_options", "options disagree"), ("runtime", "option mismatch"),
    ("runtime_gas", "gas registry"), ("mapping", "component map"),
    ("registry", "registry"), ("constants", "fixed constant"),
    ("target", "target diameter"), ("water", "properties"),
    ("geometry", "geometry changed"), ("pbl", "boundary-layer"),
])
def test_rejects_incompatible_capture_or_provenance(case, message):
    capture, manifest = pair()
    if case == "source":
        manifest["source_sha256"]["mo_physical_constants.F90"] = "wrong"
    elif case == "inputs":
        manifest["input_sha256"] = {}
    elif case == "unchecked":
        manifest["config"]["build"]["flags"].remove("-fcheck=all")
    elif case == "manifest_options":
        manifest["options"]["recalc"] = 1
    elif case == "runtime":
        manifest["runtime_metadata"]["values"]["recalc"] = True
    elif case == "runtime_gas":
        manifest["runtime_metadata"]["gas_species"][0]["species_index"] = 1
    elif case == "mapping":
        capture["entry"]["material_map"]["values"][0] = 0
    elif case == "registry":
        capture["mapping"]["species_names"][8] = "wrong"
    elif case == "constants":
        capture["entry"]["constants"]["values"][1] = 1
    elif case == "target":
        capture["entry"]["reglim"]["values"][0] = 2e-9
    elif case == "water":
        capture["entry"]["species_density"]["values"][-1] = 0
    elif case == "geometry":
        capture["exit"]["dpmid"]["values"][0] *= 2
    else:
        capture["entry"]["ppbl"]["values"][0] = 0
    with pytest.raises(ValueError, match=message):
        adapt_capture(capture, manifest)


def test_capture_properties_are_not_substituted_defaults():
    capture, manifest = pair()
    capture["entry"]["species_density"]["values"][8] = 1900
    capture["entry"]["species_molar_mass"]["values"][-1] = 18.1
    cases, _, _ = adapt_capture(capture, manifest)
    assert cases["species_density"][0, 0] == 1900
    assert cases["water_molar_mass"][0] == 18.1


def test_supported_active_and_recalculation_flags():
    capture, manifest = pair()
    for options in (capture["options"], manifest["options"]):
        options["nsnucl"] = 2
        options["recalc"] = 1
    capture["entry"]["options"]["values"][0] = 2
    capture["entry"]["options"]["values"][5] = 1
    manifest["runtime_metadata"]["values"]["nsnucl"] = 2
    manifest["runtime_metadata"]["values"]["recalc"] = True
    cases, _, _ = adapt_capture(capture, manifest)
    assert cases["scheme"].tolist() == [2]
    assert cases["recalculate"].tolist() == [True]


def test_fortran_records_reconstruct_before_cell_flattening():
    capture, manifest = pair()
    for dimensions in (capture["dimensions"], manifest["dimensions"]):
        dimensions.update(kproma=2, kbdim=2, klev=3)
    manifest["runtime_metadata"]["values"].update(kproma=2, kbdim=2, klev=3)
    for stage in ("entry", "exit"):
        for name, record in capture[stage].items():
            if name in ("pc_gas", "paerml", "pnaero", "ppres", "prh", "ptemp",
                        "pm6rp", "pm6dry", "prhop", "pww", "zcore", "zdwet", "zvols"):
                array = np.array(record["values"]).reshape(record["shape"], order="F")
                expanded = np.broadcast_to(array, (2, 3) + array.shape[2:]).copy()
                if name == "pnaero":
                    expanded[:] = np.arange(6).reshape(2, 3, 1) + 10
                record["shape"] = list(expanded.shape)
                record["values"] = expanded.ravel(order="F").tolist()
    capture["entry"]["ppbl"] = {"kind": "R", "shape": [2], "values": [2.9, 3]}
    cases, expected, metadata = adapt_capture(capture, manifest)
    np.testing.assert_array_equal(cases["number"][:, 0], np.arange(6) + 10)
    np.testing.assert_array_equal(expected["number"][:, 0], np.arange(6) + 10)
    np.testing.assert_array_equal(cases["boundary_layer"], [False, True, True, False, False, True])
    assert metadata["native_cell_shape"] == [2, 3]
    assert cases["labels"] == ["native-h1-level1", "native-h1-level2", "native-h1-level3",
                               "native-h2-level1", "native-h2-level2", "native-h2-level3"]


def test_loader_rejects_changed_raw_fixture(tmp_path):
    copied = tmp_path / "fixture"
    shutil.copytree(FIXTURE, copied)
    path = copied / "salsa-entry.txt"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_native_case(copied)
