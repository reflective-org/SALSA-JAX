"""Native-state fixtures retain real mappings, precision and material budgets."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from tools.state_capture import FILES, instrument_salsa, load_capture, load_fixture

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/native-default"


def array(state, stage, name):
    field = state[stage][name]
    return np.array(field["values"]).reshape(field["shape"], order="F")


def test_default_fixture_mapping_and_units():
    state = load_fixture(FIXTURE)
    assert state["dimensions"]["naerocomp"] == 69
    assert len(state["mapping"]["tracers"]) == 109
    material = array(state, "entry", "material_map")
    np.testing.assert_array_equal(material[:, 0], np.arange(1, 18))
    np.testing.assert_array_equal(material[:, 1], np.arange(18, 35))
    np.testing.assert_array_equal(material[:3, 2:], 0)
    np.testing.assert_array_equal(material[10:, 3], 0)
    assert state["units"]["pm6rp"] == "cm"
    assert state["units"]["prhop"] == "g cm-3"
    assert not {"pm6rp", "pm6dry", "prhop", "pww"} & state["entry"].keys()
    assert state["options"]["nsnucl"] == 0
    assert state["options"]["dp_bits"] == 64


def test_default_native_material_budgets_and_volume_conversion():
    state = load_fixture(FIXTURE)
    material = array(state, "entry", "material_map")
    species = array(state, "entry", "volume_species")
    densities = array(state, "entry", "species_density")
    mw = array(state, "entry", "species_molar_mass")
    volume = array(state, "exit", "zvols")
    soluble_before = 0.0
    soluble_after = 0.0
    for column in range(5):
        selected = material[:, column] > 0
        components = material[selected, column] - 1
        before = array(state, "entry", "paerml")[:, :, components]
        after = array(state, "exit", "paerml")[:, :, components]
        density = densities[species[column]-1]
        if column == 0:
            molecular_volume = mw[species[column]-1] / 1000 / array(state, "entry", "avogadro") / density
            expected_volume = after * 1e6 * molecular_volume
            gas_index = state["options"]["isubm_so4g"] - 1
            initial = before.sum(axis=2)*1e6 + array(state, "entry", "pc_gas")[:, :, gas_index]
            final = after.sum(axis=2)*1e6 + array(state, "exit", "pc_gas")[:, :, gas_index]
        else:
            expected_volume = after * 1e-9 / density
            initial, final = before.sum(axis=2), after.sum(axis=2)
        if column == 0:
            soluble_before += initial * molecular_volume
            soluble_after += final * molecular_volume
        elif column == 3:
            soluble_before += initial * 1e-9 / density
            soluble_after += final * 1e-9 / density
        else:
            np.testing.assert_allclose(final, initial, rtol=2e-13, atol=0)
        np.testing.assert_allclose(volume[:, :, selected, column], expected_volume, rtol=3e-15, atol=0)
    # Actual coagulation replaces sea-salt volume entering 2b with sulfate
    # (mo_ham_salsa_dynamics.F90:373-376). Conserve the combined volume,
    # including sulfate vapor; do not falsely require separate species mass.
    np.testing.assert_allclose(soluble_after, soluble_before, rtol=2e-13, atol=0)
    assert np.all(array(state, "exit", "pww") >= 0)
    assert np.all(array(state, "exit", "pm6rp") >= array(state, "exit", "pm6dry"))


@pytest.mark.parametrize("name", list(FILES.values()))
def test_fixture_rejects_modified_content(tmp_path, name):
    shutil.copytree(FIXTURE, tmp_path / "case")
    path = tmp_path / "case" / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_fixture(tmp_path / "case")


def test_fixture_rejects_incomplete_inventory(tmp_path):
    shutil.copytree(FIXTURE, tmp_path / "case")
    path = tmp_path / "case/manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["files"][FILES["exit"]]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inventory"):
        load_fixture(tmp_path / "case")


def test_native_overlay_preserves_checkout_and_line_endings(tmp_path):
    source = ROOT / "mo_ham_salsa.F90"
    original = source.read_bytes()
    copied = tmp_path / source.name
    copied.write_bytes(original)
    metadata = instrument_salsa(copied)
    assert source.read_bytes() == original
    assert metadata["original_source_sha256"] == hashlib.sha256(original).hexdigest()
    assert metadata["instrumented_source_sha256"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    assert copied.read_bytes().count(b"\n") == copied.read_bytes().count(b"\r\n")
    with pytest.raises(ValueError, match="already instrumented"):
        instrument_salsa(copied)


def test_native_overlay_refuses_changed_anchor_without_edit(tmp_path):
    path = tmp_path / "mo_ham_salsa.F90"
    original = (ROOT / path.name).read_bytes().replace(b"  END SUBROUTINE salsa", b"  END SUBROUTINE renamed")
    path.write_bytes(original)
    with pytest.raises(ValueError, match="anchor"):
        instrument_salsa(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("mutation", ["missing", "wrong_type", "wrong_shape", "component_bin", "duplicate_component",
                                      "component_tracer_bin", "number_tracer", "gas_conversion", "sulfate_gas"])
def test_native_schema_rejects_corrupted_maps(tmp_path, mutation):
    state = load_fixture(FIXTURE)
    records = state["entry"]
    if mutation == "missing":
        del records["gas_tracer"]
    elif mutation == "wrong_type":
        records["pnaero"]["kind"] = "I"
        records["pnaero"]["values"] = [int(v) for v in records["pnaero"]["values"]]
    elif mutation == "wrong_shape":
        records["pnaero"]["shape"] = [17, 1, 1]
    elif mutation == "component_bin":
        records["component_bin"]["values"][0] = 17
    elif mutation == "duplicate_component":
        records["material_map"]["values"][1] = records["material_map"]["values"][0]
    elif mutation == "component_tracer_bin":
        records["component_tracer"]["values"][0] = records["component_tracer"]["values"][1]
    elif mutation == "number_tracer":
        records["number_tracer"]["values"][0] = 1  # DMS gas, a valid but wrong tracer.
    elif mutation == "gas_conversion":
        records["gas_conversion"]["values"][0] = 1  # Mass-based, not molecular.
    else:
        records["options"]["values"][8] = 1  # SO2, not the H2SO4 slot.
    shutil.copytree(FIXTURE, tmp_path / "case")
    lines = ["SALSA_ARRAYS 1"]
    for name, record in records.items():
        lines.append(" ".join(map(str, [record["kind"], name, len(record["shape"]), *record["shape"]])))
        lines.extend(str(v) for v in record["values"])
    lines.append("END_SALSA_ARRAYS")
    (tmp_path / "case" / FILES["entry"]).write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        load_capture(tmp_path / "case")
