"""Capture the actual coagulation boundary in a copied, checked first SALSA call."""
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

FILES = {"entry": "coagulation-entry.txt", "exit": "coagulation-exit.txt"}
NUMBER = ("pnaero(1:kproma,:,:)", ("kproma", "klev", "fn2b"), "particles m-3")
VOLUME = ("zvols(1:kproma,:,:,:)", ("kproma", "klev", "fn2b", "nspecies"), "m3 material m-3 air")
ENTRY = {
    "pnaero": NUMBER,
    "zvols": VOLUME,
    "zdwet": ("zdwet(1:kproma,:,:)", ("kproma", "klev", "fn2b"), "wet diameter m"),
    "zcore": ("zcore(1:kproma,:,:)", ("kproma", "klev", "fn2b"), "dry m3 particle-1"),
    "ptemp": ("ptemp(1:kproma,:)", ("kproma", "klev"), "K"),
    "ppres": ("ppres(1:kproma,:)", ("kproma", "klev"), "Pa"),
    "ptstep": ("ptstep", (), "s"),
    "dpmid": ("dpmid", ("fn2b",), "dry diameter m"),
    "pi": ("pi", (), "1"),
}
EXIT = {"pnaero": NUMBER, "zvols": VOLUME}
INTEGERS = {
    "dimensions": ("[kproma,kbdim,klev,fn2b,subm_naerospec_nowat]", ("5",)),
    "boundaries": ("[in1a,fn1a,in2a,fn2a,in2b,fn2b]", ("6",)),
    "options": ("[nsoa,STORAGE_SIZE(1._dp)]", ("2",)),
}
CALL = """       CALL coagulation(kproma, kbdim,  klev,        &
            pnaero, zvols,  zdwet,       &
            zcore,  ptstep, ptemp, ppres)
"""


def instrument_salsa_coagulation(path: Path) -> dict:
    """Wrap the existing call, after hydration and before condensation, once."""
    original = path.read_bytes()
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    text = original.decode().replace("\r\n", "\n")
    if "SALSA_COAGULATION_STATE_V1" in text:
        raise ValueError("Coagulation source already instrumented")
    declaration_anchor = "    INTEGER :: ii, jj, kk, nn\n"
    for anchor in (declaration_anchor, CALL):
        if text.count(anchor) != 1:
            raise ValueError(f"Expected one coagulation capture anchor: {anchor.strip()}")
    declarations = """    INTEGER :: coag_capture_unit
    INTEGER, SAVE :: coag_capture_count=0
"""
    stages = {}
    for stage, fields in (("entry", ENTRY), ("exit", EXIT)):
        lines = []
        if stage == "entry":
            lines += ["! SALSA_COAGULATION_STATE_V1: copy-only instrumentation",
                      "coag_capture_count=coag_capture_count+1",
                      "IF (coag_capture_count /= 1) ERROR STOP 'coagulation capture requires one call'",
                      "IF (nsoa /= 0 .OR. subm_naerospec_nowat /= 5) &",
                      "     ERROR STOP 'unsupported coagulation capture composition'"]
        lines += [f"OPEN(NEWUNIT=coag_capture_unit,FILE='{FILES[stage]}',STATUS='new',ACTION='write')",
                  f"WRITE(coag_capture_unit,'(A)') '{HEADER}'"]
        if stage == "entry":
            for name, (expression, shape) in INTEGERS.items():
                lines += array_lines(name, expression, "I", shape, "coag_capture_unit")
        for name, (expression, shape, _) in fields.items():
            shape = tuple("subm_naerospec_nowat" if dim == "nspecies" else dim for dim in shape)
            lines += array_lines(name, expression, "R", shape, "coag_capture_unit")
        lines += [f"WRITE(coag_capture_unit,'(A)') '{FOOTER}'", "CLOSE(coag_capture_unit)"]
        stages[stage] = "".join("       " + line + "\n" for line in lines)
    text = text.replace(declaration_anchor, declaration_anchor + declarations)
    text = text.replace(CALL, stages["entry"] + CALL + stages["exit"])
    instrumented = text.encode().replace(b"\n", newline)
    patch = "".join(difflib.unified_diff(original.decode().splitlines(True), instrumented.decode().splitlines(True),
                                       fromfile=path.name, tofile=path.name))
    patch_path = path.parent / "coagulation-state.patch"
    patch_path.write_text(patch)
    path.write_bytes(instrumented)
    return {"original_source_sha256": hashlib.sha256(original).hexdigest(),
            "instrumented_source_sha256": hashlib.sha256(instrumented).hexdigest(),
            "overlay_file": patch_path.name, "overlay_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def load_capture(directory: Path) -> dict:
    entry = parse_records((directory / FILES["entry"]).read_text())
    exit_state = parse_records((directory / FILES["exit"]).read_text())
    if set(entry) != set(ENTRY) | set(INTEGERS) or set(exit_state) != set(EXIT):
        raise ValueError("Missing or unexpected coagulation field")
    dims = entry["dimensions"]
    if dims["kind"] != "I" or dims["shape"] != [5]:
        raise ValueError("Invalid coagulation dimensions record")
    sizes = dict(zip(("kproma", "kbdim", "klev", "fn2b", "nspecies"), dims["values"]))
    if any(value <= 0 for value in sizes.values()) or sizes["kproma"] > sizes["kbdim"] or sizes["nspecies"] != 5:
        raise ValueError("Invalid coagulation dimensions")
    for fields, records in ((ENTRY, entry), (EXIT, exit_state), (INTEGERS, entry)):
        for name, spec in fields.items():
            expected = [int(dim) if dim.isdigit() else sizes[dim] for dim in spec[1]]
            if records[name]["shape"] != expected or records[name]["kind"] != ("I" if fields is INTEGERS else "R"):
                raise ValueError(f"Wrong coagulation shape/type: {name}")
    start, end1, start2a, end2a, start2b, end2b = entry["boundaries"]["values"]
    if (start != 1 or not end1 >= start or start2a != end1 + 1 or end2a < start2a or
            start2b != end2a + 1 or end2b != sizes["fn2b"] or end2b - start2b != end2a - start2a):
        raise ValueError("Invalid coagulation bin boundaries")
    if entry["options"]["values"] != [0, 64]:
        raise ValueError("Unsupported coagulation composition or precision")
    for name in ("zdwet", "ptemp", "ppres", "dpmid", "pi"):
        if any(value <= 0 for value in entry[name]["values"]):
            raise ValueError(f"Nonpositive coagulation input: {name}")
    for name in ("pnaero", "zvols", "zcore", "ptstep"):
        if any(value < 0 for value in entry[name]["values"]):
            raise ValueError(f"Negative coagulation input: {name}")
    for name in EXIT:
        if any(value < 0 for value in exit_state[name]["values"]):
            raise ValueError(f"Negative coagulation output: {name}")
    return {"schema_version": 1, "order": "Fortran column-major", "dimensions": sizes,
            "species": ["SO4", "OC", "BC", "SS", "DU"],
            "stage": "after initial equilibration; immediately around coagulation; before condensation",
            "entry": entry, "exit": exit_state, "units": {name: spec[2] for name, spec in ENTRY.items()}}


def load_fixture(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or set(manifest.get("files", {})) != set(FILES.values()):
        raise ValueError("Invalid coagulation fixture inventory")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Coagulation fixture hash mismatch: {name}")
    return load_capture(directory)


def export_fixture(run_directory: Path, target: Path) -> None:
    """Export only a completed checked capture whose raw files match its manifest."""
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    capture = manifest.get("coagulation_capture", {})
    if (manifest.get("status") != "first_microphysics_completed_unvalidated" or
            capture.get("status") != "captured" or
            "-fcheck=all" not in manifest.get("config", {}).get("build", {}).get("flags", [])):
        raise ValueError("A successful checked coagulation capture is required")
    hashes = {name: hashlib.sha256((run_directory / name).read_bytes()).hexdigest() for name in FILES.values()}
    if hashes != capture.get("file_sha256"):
        raise ValueError("Run coagulation files changed after manifest creation")
    state = load_capture(run_directory)
    fixture = {"schema_version": 1, "scope": "one configured coagulation call; not trajectory acceptance",
               "files": hashes, "run_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "source_revision": manifest["source_revision"], "working_tree_status": manifest["working_tree_status"],
               "source_sha256": manifest["source_sha256"], "input_sha256": manifest["input_sha256"],
               "compiler": manifest["compiler"], "config": manifest["config"],
               **({"reference_configuration": manifest["reference_configuration"]} if "reference_configuration" in manifest else {}),
               "dimensions": state["dimensions"], "stage": state["stage"],
               "coagulation_overlay": {key: value for key, value in capture.items() if key not in ("state", "file_sha256")},
               "runtime_metadata": manifest["runtime_capture"]["metadata"],
               "scope_state_sha256": manifest["scope_capture"]["state_sha256"]}
    target.mkdir(parents=True, exist_ok=False)
    for name in FILES.values():
        shutil.copy2(run_directory / name, target / name)
    (target / "manifest.json").write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export a verified checked coagulation fixture")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new fixture directory")
    args = parser.parse_args()
    export_fixture(args.run_directory, args.output)
