"""Nested capture retains actual initialized sources and their unchanged inputs."""
import hashlib
import json
import math
from pathlib import Path

import pytest

from tools.array_records import HEADER, FOOTER
from tools.nucleation_capture import (
    CALL, CALL_INTEGERS, ENTRY, EXIT, FILES, INTEGERS, SOURCES, STATE,
    export_fixture, instrument_condensation_nucleation, load_capture, load_fixture,
)

ROOT = Path(__file__).resolve().parents[1]


def write_capture(directory, mutate=None):
    sizes = {"kproma": 1, "kbdim": 1, "klev": 1, "fn2b": 17, "nspecies": 5, "ngas": 2}
    for stage, fields in (("entry", {**ENTRY, **INTEGERS}), ("exit", {**EXIT, **CALL_INTEGERS})):
        records = {}
        for name, spec in fields.items():
            shape = [int(dim) if dim.isdigit() else sizes[dim] for dim in spec[1]]
            records[name] = {"kind": "I" if name in INTEGERS else "R", "shape": shape,
                             "values": [1] * math.prod(shape)}
        if stage == "entry":
            records["dimensions"]["values"] = list(sizes.values())
            records["boundaries"]["values"] = [1, 3, 4, 10, 11, 17]
            records["options"]["values"] = [2, 1, 0, 64]
            records["gas_indices"]["values"] = [2, 0]
            for name in SOURCES:
                records[name]["values"] = [0] * math.prod(records[name]["shape"])
        else:
            records["zj3n3"]["values"] = [10, 0]
            records["zxocnv"]["values"] = [0]
        if mutate:
            mutate(stage, records)
        lines = [HEADER]
        for name, record in records.items():
            lines.append(" ".join(map(str, [record["kind"], name, len(record["shape"]), *record["shape"]])))
            lines.extend(map(str, record["values"]))
        lines.append(FOOTER)
        (directory / FILES[stage]).write_text("\n".join(lines) + "\n")


def test_overlay_preserves_crlf_and_only_wraps_nested_call(tmp_path):
    source = ROOT / "mo_ham_salsa_dynamics.F90"
    original = source.read_bytes()
    copied = tmp_path / source.name
    copied.write_bytes(original)
    metadata = instrument_condensation_nucleation(copied)
    assert source.read_bytes() == original
    assert metadata["original_source_sha256"] == hashlib.sha256(original).hexdigest()
    assert metadata["instrumented_source_sha256"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    assert copied.read_bytes().count(b"\r\n") == copied.read_bytes().count(b"\n")
    text = copied.read_text()
    before, after = original.decode().replace('\r\n','\n'), text
    anchor = '  SUBROUTINE condensation('
    assert before[:before.index(anchor)] == after[:after.index(anchor)]
    assert text.index("zj3n3 = 0._dp") < text.index(FILES["entry"])
    assert text.index("IF (nsnucl > 0) THEN") < text.index(FILES["entry"]) < text.index("CALL nucleation(")
    assert text.index("CALL nucleation(") < text.index(FILES["exit"]) < text.index("zdvolsa=0._dp")
    assert text.count("CALL nucleation(") == 1
    assert text.index("CLOSE(nuc_capture_unit)", text.index(FILES['exit'])) < text.index("END IF", text.index(FILES['exit']))
    assert "'R zj3n3 3'" in text and "'I ppbl 1'" in text
    assert "ppbl(1:kproma)" in text
    with pytest.raises(ValueError, match="already instrumented"):
        instrument_condensation_nucleation(copied)


def test_missing_anchor_fails_without_writing(tmp_path):
    copied = tmp_path / "mo_ham_salsa_dynamics.F90"
    original = (ROOT / copied.name).read_text().replace(CALL, "")
    copied.write_text(original)
    with pytest.raises(ValueError, match="anchor"):
        instrument_condensation_nucleation(copied)
    assert copied.read_text() == original
    assert not (tmp_path / "nucleation-state.patch").exists()


@pytest.mark.parametrize('scheme', [1,2,3])
@pytest.mark.parametrize('growth', [1,2,3])
def test_capture_schema_units_and_supported_options(tmp_path, scheme, growth):
    def mutate(stage, records):
        if stage == 'entry':
            records['options']['values'] = [scheme,growth,0,64]
    write_capture(tmp_path,mutate)
    state = load_capture(tmp_path)
    assert state["units"]["pdwet"] == "wet diameter m"
    assert state["units"]["pvols"] == "m3 material m-3 air"
    assert state["units"]["zj3n3"] == "SO4,OC molecules m-3 s-1"
    assert state["species"] == ["SO4", "OC", "BC", "SS", "DU"]
    assert state["dimensions"]["nspecies"] == 5
    assert set(state["exit"]) == set(STATE) | set(SOURCES) | set(CALL_INTEGERS)
    assert state["options"]["isubm_so4g"] == 2
    assert state["options"]["isubm_ocnv"] == 0
    assert state["options"]["nsnucl"] == scheme
    assert state["options"]["nj3"] == growth


@pytest.mark.parametrize("name,change,message", [
    ("options", {"values": [0, 1, 0, 64]}, "configuration"),
    ("options", {"values": [4, 1, 0, 64]}, "configuration"),
    ("options", {"values": [2, 0, 0, 64]}, "configuration"),
    ("options", {"values": [2, 4, 0, 64]}, "configuration"),
    ("options", {"values": [2, 1, 2, 64]}, "configuration"),
    ("options", {"values": [2, 1, 0, 32]}, "configuration"),
    ("gas_indices", {"values": [0, 0]}, "gas indices"),
    ("gas_indices", {"values": [2, 2]}, "gas indices"),
    ("gas_indices", {"values": [2, 3]}, "gas indices"),
    ("boundaries", {"values": [1, 3, 4, 10, 12, 17]}, "boundaries"),
    ("dimensions", {"values": [2, 1, 1, 17, 5, 2]}, "dimensions"),
    ("ppbl", {"values": [0]}, "boundary-layer"),
    ("ppbl", {"values": [2]}, "boundary-layer"),
    ("ptstep", {"values": [0]}, "Nonpositive"),
    ("massacc", {"values": [0] * 17}, "Nonpositive"),
    ("act_coeff", {"values": [-1]}, "Negative"),
    ("pi", {"kind": "I"}, "shape/type"),
    ("pdwet", {"shape": [17]}, "shape/type"),
    ("zj3n3", {"values": [1,0]}, "initialization"),
])
def test_capture_rejects_bad_input_contract(tmp_path, name, change, message):
    def mutate(stage, records):
        if stage == "entry":
            records[name].update(change)
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match=message):
        load_capture(tmp_path)


@pytest.mark.parametrize('name,values,message', [
    ('pc_gas',[2,1],'modified input'), ('ppbl',[2],'modified input'),
    ('zj3n3',[10,1],'organic nucleation'), ('zj3n3',[-1,0],'Negative'),
    ('zxsa',[.5],'fractions'), ('zxocnv',[.5],'fractions'),
])
def test_capture_rejects_bad_output_contract(tmp_path,name,values,message):
    def mutate(stage,records):
        if stage=='exit':
            records[name]['values']=values
    write_capture(tmp_path,mutate)
    with pytest.raises(ValueError,match=message):
        load_capture(tmp_path)


def test_capture_rejects_missing_exit_field(tmp_path):
    def mutate(stage, records):
        if stage == "exit":
            del records["zj3n3"]
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match="Missing"):
        load_capture(tmp_path)


def checked_manifest(directory):
    manifest = {"status": "first_microphysics_completed_unvalidated",
                "config": {"build": {"flags": ["-fcheck=all"]}},
                "nucleation_capture": {"status": "captured", "file_sha256": {
                    name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in FILES.values()}},
                "source_revision": "test", "working_tree_status": "test", "source_sha256": {},
                "input_sha256": {}, "compiler": "test", "runtime_capture": {"metadata": {}},
                "reference_configuration": {"profile":"active-nucleation", "forcing":"test"},
                "scope_capture": {"state_sha256": "test"}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_export_fixture_roundtrip_and_no_overwrite(tmp_path):
    write_capture(tmp_path)
    manifest = checked_manifest(tmp_path)
    fixture = tmp_path / "fixture"
    export_fixture(tmp_path, fixture)
    assert load_fixture(fixture) == load_capture(tmp_path)
    exported = json.loads((fixture/'manifest.json').read_text())
    assert exported['reference_configuration'] == manifest['reference_configuration']
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
        manifest["nucleation_capture"]["status"] = "failed"
    else:
        path = tmp_path / FILES["entry"]
        path.write_bytes(path.read_bytes() + b"\n")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checked|changed"):
        export_fixture(tmp_path, tmp_path / "fixture")
    assert not (tmp_path / "fixture").exists()
