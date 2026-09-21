"""Capture real hydrated coagulation arguments, preserving the production source."""
import hashlib
import json
import math
from pathlib import Path

import pytest

from tools.array_records import HEADER, FOOTER
from tools.coagulation_capture import (
    CALL, ENTRY, EXIT, FILES, INTEGERS, export_fixture,
    instrument_salsa_coagulation, load_capture, load_fixture,
)
from tools.state_capture import instrument_salsa

ROOT = Path(__file__).resolve().parents[1]


def write_capture(directory, mutate=None):
    sizes = {"kproma": 1, "kbdim": 1, "klev": 1, "fn2b": 17, "nspecies": 5}
    for stage, fields in (("entry", {**ENTRY, **INTEGERS}), ("exit", EXIT)):
        records = {}
        for name, spec in fields.items():
            shape = [int(dim) if dim.isdigit() else sizes[dim] for dim in spec[1]]
            records[name] = {"kind": "I" if name in INTEGERS else "R", "shape": shape,
                             "values": [1] * math.prod(shape)}
        if stage == "entry":
            records["dimensions"]["values"] = list(sizes.values())
            records["boundaries"]["values"] = [1, 3, 4, 10, 11, 17]
            records["options"]["values"] = [0, 64]
        if mutate:
            mutate(stage, records)
        lines = [HEADER]
        for name, record in records.items():
            lines.append(" ".join(map(str, [record["kind"], name, len(record["shape"]), *record["shape"]])))
            lines.extend(map(str, record["values"]))
        lines.append(FOOTER)
        (directory / FILES[stage]).write_text("\n".join(lines) + "\n")


def test_overlay_preserves_crlf_and_captures_only_coagulation(tmp_path):
    source = ROOT / "mo_ham_salsa.F90"
    original = source.read_bytes()
    copied = tmp_path / source.name
    copied.write_bytes(original)
    metadata = instrument_salsa_coagulation(copied)
    assert source.read_bytes() == original
    assert metadata["original_source_sha256"] == hashlib.sha256(original).hexdigest()
    assert metadata["instrumented_source_sha256"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    assert copied.read_bytes().count(b"\r\n") == copied.read_bytes().count(b"\n")
    text = copied.read_text()
    assert text.index("CALL equilibration") < text.index(FILES["entry"]) < text.index(CALL)
    assert text.index(CALL) < text.index(FILES["exit"]) < text.index("IF (recalc) CALL equilibration")
    assert text.index(FILES["exit"]) < text.index("IF (lscond) CALL condensation")
    assert text.count(CALL) == 1
    assert "'R zdwet 3'" in text and "'R zcore 3'" in text
    with pytest.raises(ValueError, match="already instrumented"):
        instrument_salsa_coagulation(copied)


def test_overlay_composes_with_native_capture(tmp_path):
    copied = tmp_path / "mo_ham_salsa.F90"
    copied.write_bytes((ROOT / copied.name).read_bytes())
    instrument_salsa(copied)
    instrument_salsa_coagulation(copied)
    text = copied.read_text()
    assert text.count("SALSA_NATIVE_STATE_V1") == 1
    assert text.count("SALSA_COAGULATION_STATE_V1") == 1
    assert text.index("salsa-entry.txt") < text.index("coagulation-entry.txt")
    assert text.index("coagulation-exit.txt") < text.index("salsa-exit.txt")


def test_missing_anchor_fails_without_writing(tmp_path):
    copied = tmp_path / "mo_ham_salsa.F90"
    original = (ROOT / copied.name).read_text().replace(CALL, "")
    copied.write_text(original)
    with pytest.raises(ValueError, match="anchor"):
        instrument_salsa_coagulation(copied)
    assert copied.read_text() == original
    assert not (tmp_path / "coagulation-state.patch").exists()


def test_capture_schema_and_units(tmp_path):
    write_capture(tmp_path)
    state = load_capture(tmp_path)
    assert state["units"]["zdwet"] == "wet diameter m"
    assert state["units"]["zcore"] == "dry m3 particle-1"
    assert state["units"]["zvols"] == "m3 material m-3 air"
    assert state["species"] == ["SO4", "OC", "BC", "SS", "DU"]
    assert state["dimensions"]["nspecies"] == 5
    assert set(state["exit"]) == {"pnaero", "zvols"}


@pytest.mark.parametrize("name,change,message", [
    ("options", {"values": [1, 64]}, "composition"),
    ("boundaries", {"values": [1, 3, 4, 10, 12, 17]}, "boundaries"),
    ("dimensions", {"values": [2, 1, 1, 17, 5]}, "dimensions"),
    ("ptstep", {"values": [-1]}, "Negative"),
    ("dpmid", {"values": [0] * 17}, "Nonpositive"),
    ("pi", {"kind": "I"}, "shape/type"),
    ("zdwet", {"shape": [17]}, "shape/type"),
])
def test_capture_rejects_bad_physical_contract(tmp_path, name, change, message):
    def mutate(stage, records):
        if stage == "entry":
            records[name].update(change)
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match=message):
        load_capture(tmp_path)


def test_capture_rejects_missing_exit_field(tmp_path):
    def mutate(stage, records):
        if stage == "exit":
            del records["zvols"]
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match="Missing"):
        load_capture(tmp_path)


def checked_manifest(directory):
    manifest = {"status": "first_microphysics_completed_unvalidated",
                "config": {"build": {"flags": ["-fcheck=all"]}},
                "coagulation_capture": {"status": "captured", "file_sha256": {
                    name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in FILES.values()}},
                "source_revision": "test", "working_tree_status": "test", "source_sha256": {},
                "input_sha256": {}, "compiler": "test", "runtime_capture": {"metadata": {}},
                "scope_capture": {"state_sha256": "test"}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_export_fixture_roundtrip_and_no_overwrite(tmp_path):
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
def test_export_rejects_unverified_runs(tmp_path, case):
    write_capture(tmp_path)
    manifest = checked_manifest(tmp_path)
    if case == "unchecked":
        manifest["config"]["build"]["flags"] = []
    elif case == "failed":
        manifest["coagulation_capture"]["status"] = "failed"
    else:
        path = tmp_path / FILES["entry"]
        path.write_bytes(path.read_bytes() + b"\n")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checked|changed"):
        export_fixture(tmp_path, tmp_path / "fixture")
    assert not (tmp_path / "fixture").exists()
