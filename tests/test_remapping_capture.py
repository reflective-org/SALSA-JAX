# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Two-pass remapping capture preserves boundaries, pure inputs and provenance."""

import hashlib
import json
import math
from pathlib import Path

import pytest

from tools.array_records import HEADER, FOOTER
from tools.remapping_capture import (
    CALL, HYDRATION_CALL, CORRECTION_END, LOOP, STATE, CONTEXT, CORRECTIONS,
    INTEGERS, STAGES, FILES, instrument_salsa_remapping, load_capture,
    load_fixture, export_fixture,
)
from tools.state_capture import instrument_salsa
from tools.hydration_capture import instrument_salsa_hydration
from tools.coagulation_capture import instrument_salsa_coagulation
from tools.condensation_capture import instrument_salsa_condensation

ROOT = Path(__file__).resolve().parents[1]


def write_capture(directory, mutate=None):
    sizes = dict(kproma=1, kbdim=1, klev=1, fn2b=17, nspecies=5, ngas=2, nspec=14)
    for number in (1, 2):
        for stage in STAGES:
            specs = {**STATE, **(CONTEXT if stage == "entry" else {}),
                     **(INTEGERS if stage == "entry" else {}),
                     **(CORRECTIONS if stage == "correction_exit" else {})}
            records = {"pass_index": {"kind": "I", "shape": [], "values": [number]}}
            for name, spec in specs.items():
                shape = [int(dim) if dim.isdigit() else sizes[dim] for dim in spec[1]]
                records[name] = {"kind": "I" if name in INTEGERS else "R",
                                 "shape": shape, "values": [1] * math.prod(shape)}
            if stage == "entry":
                for name, values in {
                    "dimensions": list(sizes.values()), "boundaries": [1, 3, 4, 10, 11, 17],
                    "options": [0, 1, 0, 0, 64], "species_indices": [8, 9, 10, 11, 12],
                    "standard_species_indices": [8, 9, 10, 11, 12], "water_species_index": [13],
                    "gas_indices": [2, 0], "reglim": [1, 2, 3, 4], "epsoc": [0.1],
                    "slim": [1.01], "vhilim": [2] * 17,
                }.items():
                    records[name]["values"] = values
            if mutate:
                mutate(number, stage, records)
            lines = [HEADER]
            for name, record in records.items():
                lines.append(" ".join(map(str, [record["kind"], name, len(record["shape"]), *record["shape"]])))
                lines.extend(map(str, record["values"]))
            path = directory / FILES[f"pass{number}_{stage}"]
            path.write_text("\n".join([*lines, FOOTER]) + "\n")


def test_overlay_preserves_crlf_hashes_and_actual_loop(tmp_path):
    source = ROOT / "mo_ham_salsa.F90"
    original = source.read_bytes()
    copied = tmp_path / source.name
    copied.write_bytes(original)
    metadata = instrument_salsa_remapping(copied)
    assert source.read_bytes() == original
    assert metadata["original_source_sha256"] == hashlib.sha256(original).hexdigest()
    assert metadata["instrumented_source_sha256"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    assert metadata["overlay_sha256"] == hashlib.sha256((tmp_path / metadata["overlay_file"]).read_bytes()).hexdigest()
    assert copied.read_bytes().count(b"\r\n") == copied.read_bytes().count(b"\n")
    text = copied.read_text()
    assert text.count(LOOP) == 1
    assert text.count(CALL) == 1
    assert text.count("CALL equilibration") == original.decode().count("CALL equilibration")
    assert text.index("'-entry.txt'") < text.index(CALL) < text.index("'-remap-exit.txt'")
    assert text.index("'-cleanup-exit.txt'") < text.index(HYDRATION_CALL) < text.index("'-correction-exit.txt'")
    assert "remap_capture_count /= 1" in text
    added = set(text.splitlines()) - set(original.decode().splitlines())
    assert all(len(line) <= 132 for line in added)
    with pytest.raises(ValueError, match="already instrumented"):
        instrument_salsa_remapping(copied)


def test_capture_overlays_compose(tmp_path):
    copied = tmp_path / "mo_ham_salsa.F90"
    copied.write_bytes((ROOT / copied.name).read_bytes())
    for instrument in (instrument_salsa, instrument_salsa_hydration,
                       instrument_salsa_coagulation, instrument_salsa_condensation):
        instrument(copied)
    before = copied.read_bytes()
    metadata = instrument_salsa_remapping(copied)
    assert metadata["original_source_sha256"] == hashlib.sha256(before).hexdigest()
    text = copied.read_text()
    assert text.index("condensation-exit.txt") < text.index("'-entry.txt'")
    assert text.index("'-correction-exit.txt'") < text.index("salsa-exit.txt")


@pytest.mark.parametrize("anchor", [CALL, HYDRATION_CALL, CORRECTION_END, LOOP])
@pytest.mark.parametrize("copies", [0, 2])
def test_bad_anchor_fails_before_writing(tmp_path, anchor, copies):
    copied = tmp_path / "mo_ham_salsa.F90"
    original = (ROOT / copied.name).read_text().replace(anchor, anchor * copies)
    copied.write_text(original)
    with pytest.raises(ValueError, match="anchor"):
        instrument_salsa_remapping(copied)
    assert copied.read_text() == original
    assert not (tmp_path / "remapping-state.patch").exists()


def test_capture_contract_and_signed_water_roundoff(tmp_path):
    def mutate(number, stage, records):
        if stage == "correction_exit":
            records["pww"]["values"][0] = -1e-30
    write_capture(tmp_path, mutate)
    state = load_capture(tmp_path)
    assert len(state["passes"]) == 2
    assert state["species"] == ["SO4", "OC", "BC", "SS", "DU"]
    assert state["options"] == dict(lsol2b=False, lhydr2b=True, recalc=False,
                                    nsoa=0, dp_bits=64, isubm_so4g=2, isubm_ocnv=0)
    assert state["passes"][1]["entry"]["pass_index"]["values"] == [2]
    assert state["units"]["pm6dry"] == "cm"
    assert state["units"]["pww"] == "kg water m-3 air"


@pytest.mark.parametrize("name,change,message", [
    ("options", {"values": [2, 1, 0, 0, 64]}, "options"),
    ("options", {"values": [0, 1, 0, 0, 32]}, "options"),
    ("species_indices", {"values": [9, 8, 10, 11, 12]}, "mapping"),
    ("water_species_index", {"values": [8]}, "mapping"),
    ("gas_indices", {"values": [2, 2]}, "gas indices"),
    ("soluble", {"values": [2] * 17}, "solubility"),
    ("dimensions", {"values": [2, 1, 1, 17, 5, 2, 14]}, "dimensions"),
    ("boundaries", {"values": [1, 3, 4, 10, 12, 17]}, "boundaries"),
    ("vhilim", {"values": [1] * 17}, "volume limits"),
    ("slim", {"values": [1]}, "supersaturation"),
    ("epsoc", {"values": [1.1]}, "solubility"),
    ("species_density", {"values": [0] * 5}, "Nonpositive"),
    ("pi", {"kind": "I"}, "shape/type"),
])
def test_bad_context_rejected(tmp_path, name, change, message):
    def mutate(number, stage, records):
        if stage == "entry":
            records[name].update(change)
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match=message):
        load_capture(tmp_path)


@pytest.mark.parametrize("stage,name,message", [
    ("remap_exit", "pc_gas", "changed pure input"),
    ("cleanup_exit", "ptemp", "changed meteorology"),
    ("correction_exit", "zvols", "Correction changed pure input"),
])
def test_pure_inputs_cannot_change(tmp_path, stage, name, message):
    def mutate(number, current, records):
        if number == 1 and stage == current:
            records[name]["values"][0] = 2
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match=message):
        load_capture(tmp_path)


def test_passes_must_be_continuous(tmp_path):
    def mutate(number, stage, records):
        if number == 2:
            records["pnaero"]["values"][0] = 2
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match="Discontinuous"):
        load_capture(tmp_path)


@pytest.mark.parametrize("case,message", [
    ("missing", "Missing"), ("index", "pass index"), ("shape", "shape/type"),
    ("negative", "Negative"), ("nonfinite", "finite"), ("context", "context changed"),
])
def test_malformed_stage_rejected(tmp_path, case, message):
    def mutate(number, stage, records):
        if number == 2 and stage == "entry" and case == "context":
            records["nlim"]["values"][0] = 2
        if number == 1 and stage == "correction_exit":
            if case == "missing":
                del records["zcore"]
            elif case == "index":
                records["pass_index"]["values"] = [2]
            elif case == "shape":
                records["zcore"]["shape"] = [17]
            elif case == "negative":
                records["pnaero"]["values"][0] = -1
            elif case == "nonfinite":
                records["zcore"]["values"][0] = float("nan")
    write_capture(tmp_path, mutate)
    with pytest.raises(ValueError, match=message):
        load_capture(tmp_path)


def checked_manifest(directory):
    manifest = {"status": "first_microphysics_completed_unvalidated",
                "config": {"build": {"flags": ["-fcheck=all"]}},
                "remapping_capture": {"status": "captured", "file_sha256": {
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
    entry = fixture / FILES["pass2_entry"]
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
        manifest["remapping_capture"]["status"] = "failed"
    else:
        path = tmp_path / FILES["pass1_entry"]
        path.write_bytes(path.read_bytes() + b"\n")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checked|changed"):
        export_fixture(tmp_path, tmp_path / "fixture")
    assert not (tmp_path / "fixture").exists()
