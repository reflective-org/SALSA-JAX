# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Copy-only capture of nucleation called inside native standalone condensation."""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import shutil

if __package__:
    from .array_records import HEADER, FOOTER, array_lines, parse_records
else:
    from array_records import HEADER, FOOTER, array_lines, parse_records

FILES = {"entry": "nucleation-entry.txt", "exit": "nucleation-exit.txt"}
STATE = {
    "pnaero": ("pnaero(1:kproma,:,:)", ("kproma", "klev", "fn2b"), "particles m-3"),
    "pvols": ("pvols(1:kproma,:,:,:)", ("kproma", "klev", "fn2b", "nspecies"), "m3 material m-3 air"),
    "pc_gas": ("pc_gas(1:kproma,:,:)", ("kproma", "klev", "ngas"), "molecules m-3"),
    "pdwet": ("pdwet(1:kproma,:,:)", ("kproma", "klev", "fn2b"), "wet diameter m"),
    "ptemp": ("ptemp(1:kproma,:)", ("kproma", "klev"), "K"),
    "prh": ("prh(1:kproma,:)", ("kproma", "klev"), "1"),
    "ppres": ("ppres(1:kproma,:)", ("kproma", "klev"), "Pa"),
    "ptstep": ("ptstep", (), "s"),
}
SOURCES = {
    "zj3n3": ("zj3n3(1:kproma,:,:)", ("kproma", "klev", "2"), "SO4,OC molecules m-3 s-1"),
    "zxsa": ("zxsa(1:kproma,:)", ("kproma", "klev"), "sulfate molecular fraction"),
    "zxocnv": ("zxocnv(1:kproma,:)", ("kproma", "klev"), "organic molecular fraction"),
}
ENTRY = {
    **STATE, **SOURCES,
    "massacc": ("massacc", ("fn2b",), "1"),
    "species_molar_mass": ("[speclist(id_so4)%moleweight,speclist(id_oc)%moleweight]", ("2",), "g mol-1; SO4,OC"),
    "species_density": ("[speclist(id_so4)%density,speclist(id_oc)%density]", ("2",), "kg m-3; SO4,OC"),
    "d_sa": ("d_sa", (), "m"),
    "n3": ("n3", (), "molecules per target particle"),
    "reglim1": ("nuc_reglim(1)", (), "target diameter m"),
    "act_coeff": ("nuc_act_coeff", (), "s-1"),
    "avo": ("avo", (), "mol-1"),
    "argas": ("argas", (), "J mol-1 K-1"),
    "ak": ("ak", (), "J K-1"),
    "p0sl_bg": ("p0sl_bg", (), "Pa"),
    "pi": ("pi", (), "1"),
}
EXIT = {**STATE, **SOURCES}
CALL_INTEGERS = {"ppbl": ("ppbl(1:kproma)", ("kproma",)), "krow": ("krow", ())}
INTEGERS = {
    "dimensions": ("[kproma,kbdim,klev,fn2b,subm_naerospec_nowat,subm_ngasspec]", ("6",)),
    "boundaries": ("[in1a,nuc_fn1a,in2a,nuc_fn2a,nuc_in2b,fn2b]", ("6",)),
    "options": ("[nsnucl,nuc_nj3,nsoa,STORAGE_SIZE(1._dp)]", ("4",)),
    "gas_indices": ("[isubm_so4g,isubm_ocnv]", ("2",)),
    **CALL_INTEGERS,
}
CALL = """    IF (nsnucl > 0) CALL nucleation(kproma, kbdim,  klev,   krow,                     &
                                 pnaero, pdwet,                                    &
                                 ptemp,  prh,    ppres,                            &
                                 ptstep, pc_gas,                                   &
                                 zj3n3, zxsa, zxocnv, ppbl                         &
                                 )
"""


def instrument_condensation_nucleation(path: Path) -> dict:
    """Instrument one checked native call without modifying its arithmetic."""
    original = path.read_bytes()
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    text = original.decode().replace("\r\n", "\n")
    if "SALSA_NUCLEATION_STATE_V1" in text:
        raise ValueError("Nucleation source already instrumented")
    start_anchor, end_anchor = "  SUBROUTINE condensation(", "  END SUBROUTINE condensation\n"
    for anchor in (start_anchor, end_anchor):
        if text.count(anchor) != 1:
            raise ValueError(f"Expected one nucleation capture anchor: {anchor.strip()}")
    start = text.index(start_anchor)
    end = text.index(end_anchor, start) + len(end_anchor)
    body = text[start:end]
    import_anchor = "   IMPLICIT NONE\n"
    declaration_anchor = "    INTEGER :: ii, jj                ! loop indices\n"
    for anchor in (import_anchor, declaration_anchor, CALL):
        if body.count(anchor) != 1:
            raise ValueError(f"Expected one nucleation capture anchor: {anchor.strip()}")
    imports = """    USE mo_ham_salsactl, ONLY: nuc_nj3=>nj3,nuc_reglim=>reglim,nuc_act_coeff=>act_coeff, &
         nuc_fn1a=>fn1a,nuc_fn2a=>fn2a,nuc_in2b=>in2b
"""
    declarations = """    INTEGER :: nuc_capture_unit
    INTEGER, SAVE :: nuc_capture_count=0
"""
    stages = {}
    for stage, fields in (("entry", ENTRY), ("exit", EXIT)):
        lines = []
        if stage == "entry":
            lines += ["! SALSA_NUCLEATION_STATE_V1: copy-only instrumentation",
                      "nuc_capture_count=nuc_capture_count+1",
                      "IF (nuc_capture_count /= 1) ERROR STOP 'nucleation capture requires one call'",
                      "IF (nsnucl < 1 .OR. nsnucl > 3 .OR. nuc_nj3 < 1 .OR. nuc_nj3 > 3 .OR. &",
                      "    nsoa /= 0 .OR. subm_naerospec_nowat /= 5) &",
                      "     ERROR STOP 'unsupported nucleation capture configuration'"]
        lines += [f"OPEN(NEWUNIT=nuc_capture_unit,FILE='{FILES[stage]}',STATUS='new',ACTION='write')",
                  f"WRITE(nuc_capture_unit,'(A)') '{HEADER}'"]
        for name, (expression, shape) in (INTEGERS if stage == "entry" else CALL_INTEGERS).items():
            lines += array_lines(name, expression, "I", shape, "nuc_capture_unit")
        for name, (expression, shape, _) in fields.items():
            shape = tuple({"nspecies": "subm_naerospec_nowat", "ngas": "subm_ngasspec"}.get(dim, dim) for dim in shape)
            lines += array_lines(name, expression, "R", shape, "nuc_capture_unit")
        lines += [f"WRITE(nuc_capture_unit,'(A)') '{FOOTER}'", "CLOSE(nuc_capture_unit)"]
        stages[stage] = "".join("       " + line + "\n" for line in lines)
    body = body.replace(import_anchor, imports + import_anchor)
    body = body.replace(declaration_anchor, declaration_anchor + declarations)
    bare_call = CALL.replace("    IF (nsnucl > 0) CALL", "       CALL", 1)
    body = body.replace(CALL, "    IF (nsnucl > 0) THEN\n" + stages["entry"] + bare_call + stages["exit"] + "    END IF\n")
    text = text[:start] + body + text[end:]
    instrumented = text.encode().replace(b"\n", newline)
    patch = "".join(difflib.unified_diff(original.decode().splitlines(True), instrumented.decode().splitlines(True),
                                       fromfile=path.name, tofile=path.name))
    patch_path = path.parent / "nucleation-state.patch"
    patch_path.write_text(patch)
    path.write_bytes(instrumented)
    return {"original_source_sha256": hashlib.sha256(original).hexdigest(),
            "instrumented_source_sha256": hashlib.sha256(instrumented).hexdigest(),
            "overlay_file": patch_path.name, "overlay_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def load_capture(directory: Path) -> dict:
    entry = parse_records((directory / FILES["entry"]).read_text())
    exit_state = parse_records((directory / FILES["exit"]).read_text())
    if set(entry) != set(ENTRY) | set(INTEGERS) or set(exit_state) != set(EXIT) | set(CALL_INTEGERS):
        raise ValueError("Missing or unexpected nucleation field")
    dims = entry["dimensions"]
    if dims["kind"] != "I" or dims["shape"] != [6]:
        raise ValueError("Invalid nucleation dimensions record")
    sizes = dict(zip(("kproma", "kbdim", "klev", "fn2b", "nspecies", "ngas"), dims["values"]))
    if (any(value <= 0 for value in sizes.values()) or sizes["kproma"] > sizes["kbdim"] or
            sizes["nspecies"] != 5 or sizes["fn2b"] != 17):
        raise ValueError("Invalid nucleation dimensions")
    for fields, records, kind in ((ENTRY, entry, "R"), (EXIT, exit_state, "R"),
                                  (INTEGERS, entry, "I"), (CALL_INTEGERS, exit_state, "I")):
        for name, spec in fields.items():
            expected = [int(dim) if dim.isdigit() else sizes[dim] for dim in spec[1]]
            if records[name]["shape"] != expected or records[name]["kind"] != kind:
                raise ValueError(f"Wrong nucleation shape/type: {name}")
    if entry["boundaries"]["values"] != [1, 3, 4, 10, 11, 17]:
        raise ValueError("Unsupported nucleation bin boundaries")
    scheme, growth, nsoa, precision = entry["options"]["values"]
    if scheme not in (1, 2, 3) or growth not in (1, 2, 3) or nsoa != 0 or precision != 64:
        raise ValueError("Unsupported nucleation scheme/growth/composition/precision configuration")
    sulfate, organic = entry["gas_indices"]["values"]
    if not 1 <= sulfate <= sizes["ngas"] or not 0 <= organic <= sizes["ngas"] or sulfate == organic:
        raise ValueError("Invalid nucleation gas indices")
    if any(not 1 <= value <= sizes["klev"] for value in entry["ppbl"]["values"]):
        raise ValueError("Invalid nucleation boundary-layer level")
    for name in ("pdwet", "ptemp", "ppres", "ptstep", "massacc", "species_molar_mass", "species_density",
                 "d_sa", "n3", "reglim1", "avo", "argas", "ak", "p0sl_bg", "pi"):
        if any(value <= 0 for value in entry[name]["values"]):
            raise ValueError(f"Nonpositive nucleation input: {name}")
    for name in ("pnaero", "pvols", "pc_gas", "prh", "act_coeff"):
        if any(value < 0 for value in entry[name]["values"]):
            raise ValueError(f"Negative nucleation input: {name}")
    for name in SOURCES:
        if any(value != 0 for value in entry[name]["values"]):
            raise ValueError(f"Nucleation source caller initialization is not zero: {name}")
        if any(value < 0 for value in exit_state[name]["values"]):
            raise ValueError(f"Negative nucleation output: {name}")
    for name in STATE | CALL_INTEGERS:
        if entry[name] != exit_state[name]:
            raise ValueError(f"Nucleation modified input state: {name}")
    if any(value != 1 for value in exit_state["zxsa"]["values"]) or any(value != 0 for value in exit_state["zxocnv"]["values"]):
        raise ValueError("Unexpected sulfate-only nucleation fractions")
    cells = sizes["kproma"] * sizes["klev"]
    if any(value != 0 for value in exit_state["zj3n3"]["values"][cells:]):
        raise ValueError("Unexpected organic nucleation source")
    return {"schema_version": 1, "order": "Fortran column-major", "dimensions": sizes,
            "species": ["SO4", "OC", "BC", "SS", "DU"],
            "options": {"nsnucl": scheme, "nj3": growth, "nsoa": nsoa, "dp_bits": precision,
                        "isubm_so4g": sulfate, "isubm_ocnv": organic},
            "stage": "inside condensation; immediately around nucleation; before vapor uptake and particle insertion",
            "entry": entry, "exit": exit_state,
            "units": {**{name: spec[2] for name, spec in ENTRY.items()},
                      "ppbl": "one-based level; boundary layer is jj >= ppbl", "krow": "latitude index"}}


def load_fixture(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or set(manifest.get("files", {})) != set(FILES.values()):
        raise ValueError("Invalid nucleation fixture inventory")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Nucleation fixture hash mismatch: {name}")
    return load_capture(directory)


def export_fixture(run_directory: Path, target: Path) -> None:
    """Export only a completed checked capture whose raw files match its manifest."""
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    capture = manifest.get("nucleation_capture", {})
    if (manifest.get("status") != "first_microphysics_completed_unvalidated" or
            capture.get("status") != "captured" or
            "-fcheck=all" not in manifest.get("config", {}).get("build", {}).get("flags", [])):
        raise ValueError("A successful checked nucleation capture is required")
    hashes = {name: hashlib.sha256((run_directory / name).read_bytes()).hexdigest() for name in FILES.values()}
    if hashes != capture.get("file_sha256"):
        raise ValueError("Run nucleation files changed after manifest creation")
    state = load_capture(run_directory)
    fixture = {"schema_version": 1, "scope": "one native nucleation call inside condensation; not trajectory acceptance",
               "files": hashes, "run_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "source_revision": manifest["source_revision"], "working_tree_status": manifest["working_tree_status"],
               "source_sha256": manifest["source_sha256"], "input_sha256": manifest["input_sha256"],
               "compiler": manifest["compiler"], "config": manifest["config"],
               "reference_configuration": manifest.get("reference_configuration"),
               "dimensions": state["dimensions"], "options": state["options"], "stage": state["stage"],
               "nucleation_overlay": {key: value for key, value in capture.items() if key not in ("state", "file_sha256")},
               "runtime_metadata": manifest["runtime_capture"]["metadata"],
               "scope_state_sha256": manifest["scope_capture"]["state_sha256"]}
    target.mkdir(parents=True, exist_ok=False)
    for name in FILES.values():
        shutil.copy2(run_directory / name, target / name)
    (target / "manifest.json").write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export a verified checked nested nucleation fixture")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new fixture directory")
    args = parser.parse_args()
    export_fixture(args.run_directory, args.output)
