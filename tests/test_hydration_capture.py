"""First hydration capture preserves inputs, source bytes and fixture provenance."""

import hashlib
import json
import math
from pathlib import Path

import pytest

from tools.array_records import HEADER, FOOTER
from tools.hydration_capture import (
    CALL, ENTRY, EXIT, FILES, INTEGERS, export_fixture,
    instrument_salsa_hydration, load_capture, load_fixture,
)
from tools.state_capture import instrument_salsa
from tools.coagulation_capture import instrument_salsa_coagulation
from tools.condensation_capture import instrument_salsa_condensation

ROOT = Path(__file__).resolve().parents[1]


def write_capture(directory, mutate=None):
    sizes = dict(kproma=1, kbdim=1, klev=1, fn2b=17, nspecies=5, nspec=14)
    for stage, fields in (("entry", {**ENTRY, **INTEGERS}), ("exit", EXIT)):
        records = {}
        for name, spec in fields.items():
            shape = [int(dim) if dim.isdigit() else sizes[dim] for dim in spec[1]]
            records[name] = {"kind": "I" if name in INTEGERS else "R", "shape": shape,
                             "values": [1] * math.prod(shape)}
        if stage == "entry":
            for name, values in {
                "dimensions": list(sizes.values()), "boundaries": [1, 3, 4, 10, 11, 17],
                "options": [1, 0, 64], "species_indices": [8, 9, 10, 11, 12],
                "standard_species_indices": [8, 9, 10, 11, 12], "water_species_index": [13],
                "recalc": [0], "reglim": [1, 2, 3, 4],
            }.items():
                records[name]["values"] = values
        if mutate:
            mutate(stage, records)
        lines = [HEADER]
        for name, record in records.items():
            lines.append(" ".join(map(str, [record["kind"], name, len(record["shape"]), *record["shape"]])))
            lines.extend(map(str, record["values"]))
        (directory / FILES[stage]).write_text("\n".join([*lines, FOOTER]) + "\n")


def test_overlay_preserves_crlf_and_only_first_call(tmp_path):
    source = ROOT / "mo_ham_salsa.F90"
    original = source.read_bytes()
    copied = tmp_path / source.name
    copied.write_bytes(original)
    metadata = instrument_salsa_hydration(copied)
    assert source.read_bytes() == original
    assert metadata["original_source_sha256"] == hashlib.sha256(original).hexdigest()
    assert metadata["instrumented_source_sha256"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    assert copied.read_bytes().count(b"\r\n") == copied.read_bytes().count(b"\n")
    text = copied.read_text()
    first = text.index(CALL)
    assert text.index(FILES["entry"]) < first < text.index(FILES["exit"]) < text.index("IF (lscoag) THEN")
    assert text.count("CALL equilibration") == original.decode().count("CALL equilibration")
    assert text.count("IF (recalc) CALL equilibration") == 1
    assert "hyd_capture_count /= 1" in text
    assert "'I soluble 1'" in text and "'R species_molar_mass 1'" in text
    with pytest.raises(ValueError, match="already instrumented"):
        instrument_salsa_hydration(copied)


def test_capture_overlays_compose(tmp_path):
    copied = tmp_path / "mo_ham_salsa.F90"
    copied.write_bytes((ROOT / copied.name).read_bytes())
    instrument_salsa(copied)
    instrument_salsa_coagulation(copied)
    instrument_salsa_condensation(copied)
    instrument_salsa_hydration(copied)
    text = copied.read_text()
    assert text.index("salsa-entry.txt") < text.index(FILES["entry"])
    assert text.index(FILES["exit"]) < text.index("coagulation-entry.txt")
    assert text.index("coagulation-exit.txt") < text.index("condensation-entry.txt")


@pytest.mark.parametrize("copies", [0, 2])
def test_bad_call_anchor_fails_before_writing(tmp_path, copies):
    copied = tmp_path / "mo_ham_salsa.F90"
    original = (ROOT / copied.name).read_text().replace(CALL, CALL * copies)
    copied.write_text(original)
    with pytest.raises(ValueError, match="anchor"):
        instrument_salsa_hydration(copied)
    assert copied.read_text() == original
    assert not (tmp_path / "hydration-state.patch").exists()


def test_capture_contract(tmp_path):
    write_capture(tmp_path)
    state = load_capture(tmp_path)
    assert state["species"] == ["SO4", "OC", "BC", "SS", "DU"]
    assert state["entry"]["soluble"]["shape"] == [17]
    assert state["entry"]["species_indices"]["values"] == [8, 9, 10, 11, 12]
    assert state["options"] == {"lhydr2b": True, "recalc": False, "nsoa": 0, "dp_bits": 64}
    assert state["units"]["pcore"] == "m3 particle-1"


@pytest.mark.parametrize("name,change,message", [
    ("options", {"values": [2, 0, 64]}, "options"),
    ("options", {"values": [1, 2, 64]}, "options"),
    ("options", {"values": [1, 0, 32]}, "options"),
    ("soluble", {"values": [2] * 17}, "solubility"),
    ("recalc", {"values": [2]}, "recalc"),
    ("species_indices", {"values": [9, 8, 10, 11, 12]}, "mapping"),
    ("water_species_index", {"values": [8]}, "mapping"),
    ("water_species_index", {"values": [15]}, "mapping"),
    ("dimensions", {"values": [2, 1, 1, 17, 5, 14]}, "dimensions"),
    ("boundaries", {"values": [1, 3, 4, 10, 12, 17]}, "boundaries"),
    ("species_density", {"values": [0] * 5}, "Nonpositive"),
    ("epsoc", {"values": [1.1]}, "fraction"),
    ("pi", {"kind": "I"}, "shape/type"),
])
def test_bad_contract_rejected(tmp_path, name, change, message):
    def mutate(stage, records):
        if stage == "entry":
            records[name].update(change)
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match=message):
        load_capture(tmp_path)


def test_changed_pure_input_rejected(tmp_path):
    def mutate(stage, records):
        if stage == "exit":
            records["pvols"]["values"][0] = 2
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match="changed pure input"):
        load_capture(tmp_path)


def test_missing_output_rejected(tmp_path):
    def mutate(stage, records):
        if stage == "exit":
            del records["pcore"]
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match="Missing"):
        load_capture(tmp_path)


def checked_manifest(directory):
    manifest = {"status": "first_microphysics_completed_unvalidated",
                "config": {"build": {"flags": ["-fcheck=all"]}},
                "hydration_capture": {"status": "captured", "file_sha256": {
                    name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in FILES.values()}},
                "source_revision": "test", "working_tree_status": "test", "source_sha256": {},
                "input_sha256": {}, "compiler": "test", "runtime_capture": {"metadata": {}},
                "scope_capture": {"state_sha256": "test"}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_fixture_roundtrip_no_overwrite_and_tamper_detection(tmp_path):
    write_capture(tmp_path)
    checked_manifest(tmp_path)
    fixture = tmp_path / "fixture"
    export_fixture(tmp_path, fixture)
    assert load_fixture(fixture) == load_capture(tmp_path)
    with pytest.raises(FileExistsError):
        export_fixture(tmp_path, fixture)
    entry = fixture / FILES["entry"]
    entry.write_bytes(entry.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_fixture(fixture)


@pytest.mark.parametrize("case", ["unchecked", "failed", "changed"])
def test_export_rejects_unverified_capture(tmp_path, case):
    write_capture(tmp_path)
    manifest = checked_manifest(tmp_path)
    if case == "unchecked":
        manifest["config"]["build"]["flags"] = []
    elif case == "failed":
        manifest["hydration_capture"]["status"] = "failed"
    else:
        path = tmp_path / FILES["entry"]
        path.write_bytes(path.read_bytes() + b"\n")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checked|changed"):
        export_fixture(tmp_path, tmp_path / "fixture")
    assert not (tmp_path / "fixture").exists()
