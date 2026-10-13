"""Capture standalone condensation boundaries in a copied source."""
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

FILES = {"entry": "condensation-entry.txt", "exit": "condensation-exit.txt"}
NUMBER = ("pnaero(1:kproma,:,:)", ("kproma", "klev", "fn2b"), "particles m-3")
VOLUME = ("zvols(1:kproma,:,:,:)", ("kproma", "klev", "fn2b", "nspecies"), "m3 material m-3 air")
GAS = ("pc_gas(1:kproma,:,:)", ("kproma", "klev", "ngas"), "molecules m-3")
ENTRY = {
    "pnaero": NUMBER, "zvols": VOLUME, "pc_gas": GAS,
    "zdwet": ("zdwet(1:kproma,:,:)", ("kproma", "klev", "fn2b"), "wet diameter m"),
    "prh": ("prh(1:kproma,:)", ("kproma", "klev"), "1; unused with nucleation disabled"),
    "ptemp": ("ptemp(1:kproma,:)", ("kproma", "klev"), "K"),
    "ppres": ("ppres(1:kproma,:)", ("kproma", "klev"), "Pa"),
    "ptstep": ("ptstep", (), "s"),
    "massacc": ("cond_massacc", ("fn2b",), "1"),
    "d_sa": ("cond_d_sa", (), "m"),
    "species_molar_mass": ("[speclist(id_so4)%moleweight,speclist(id_oc)%moleweight]", ("2",), "g mol-1; SO4,OC"),
    "species_density": ("[speclist(id_so4)%density,speclist(id_oc)%density]", ("2",), "kg m-3; SO4,OC"),
    "avo": ("avo", (), "mol-1"),
    "argas": ("cond_argas", (), "J mol-1 K-1"),
    "ak": ("cond_ak", (), "J K-1"),
    "p0sl_bg": ("cond_p0sl_bg", (), "Pa"),
    "pi": ("pi", (), "1"),
    "time_step_len": ("cond_time_step_len", (), "s; source reads inverse for inactive diagnostics"),
}
EXIT = {"pnaero": NUMBER, "zvols": VOLUME, "pc_gas": GAS}
INTEGERS = {
    "dimensions": ("[kproma,kbdim,klev,fn2b,subm_naerospec_nowat,subm_ngasspec]", ("6",)),
    "boundaries": ("[in1a,fn1a,in2a,fn2a,in2b,fn2b]", ("6",)),
    "options": ("[cond_nsnucl,nsoa,STORAGE_SIZE(1._dp)]", ("3",)),
    "gas_indices": ("[isubm_so4g,isubm_ocnv]", ("2",)),
    "ppbl": ("zpbl(1:kproma)", ("kproma",)),
    "krow": ("krow", ()),
}
CALL = """    IF (lscond) CALL condensation(&
         kproma,   kbdim,   klev,    krow,               &
         pnaero,   zvols,   zdwet,                       &
         pc_gas,                                         &
         prh,      ptemp,   ppres,   ptstep, zpbl        &
    )
"""


def instrument_salsa_condensation(path: Path, *, allow_nucleation=False) -> dict:
    """Wrap the actual conditional call after hydration and before remapping."""
    integers = {**INTEGERS, "nj3": ("cond_nj3", ())} if allow_nucleation else INTEGERS
    original = path.read_bytes()
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    text = original.decode().replace("\r\n", "\n")
    if "SALSA_CONDENSATION_STATE_V1" in text:
        raise ValueError("Condensation source already instrumented")
    import_anchor = "    IMPLICIT NONE\n"
    declaration_anchor = "    INTEGER :: ii, jj, kk, nn\n"
    for anchor in (import_anchor, declaration_anchor, CALL):
        if text.count(anchor) != 1:
            raise ValueError(f"Expected one condensation capture anchor: {anchor.strip()}")
    imports = """    USE mo_ham_salsactl, ONLY: cond_nsnucl=>nsnucl,cond_massacc=>massacc,cond_d_sa=>d_sa
    USE mo_physical_constants, ONLY: cond_argas=>argas,cond_ak=>ak,cond_p0sl_bg=>p0sl_bg
    USE mo_time_control, ONLY: cond_time_step_len=>time_step_len
"""
    if allow_nucleation:
        imports += "    USE mo_ham_salsactl, ONLY: cond_nj3=>nj3\n"
    declarations = """    INTEGER :: cond_capture_unit
    INTEGER, SAVE :: cond_capture_count=0
"""
    option_guard = "cond_nsnucl < 1 .OR. cond_nsnucl > 3 .OR. cond_nj3 < 1 .OR. cond_nj3 > 3" if allow_nucleation else "cond_nsnucl /= 0"
    stages = {}
    for stage, fields in (("entry", ENTRY), ("exit", EXIT)):
        lines = []
        if stage == "entry":
            lines += ["! SALSA_CONDENSATION_STATE_V1: copy-only instrumentation",
                      "cond_capture_count=cond_capture_count+1",
                      "IF (cond_capture_count /= 1) ERROR STOP 'condensation capture requires one call'",
                      f"IF ({option_guard} .OR. nsoa /= 0 .OR. subm_naerospec_nowat /= 5) &",
                      "     ERROR STOP 'unsupported condensation capture configuration'"]
        lines += [f"OPEN(NEWUNIT=cond_capture_unit,FILE='{FILES[stage]}',STATUS='new',ACTION='write')",
                  f"WRITE(cond_capture_unit,'(A)') '{HEADER}'"]
        if stage == "entry":
            for name, (expression, shape) in integers.items():
                lines += array_lines(name, expression, "I", shape, "cond_capture_unit")
        for name, (expression, shape, _) in fields.items():
            shape = tuple({"nspecies": "subm_naerospec_nowat", "ngas": "subm_ngasspec"}.get(dim, dim) for dim in shape)
            lines += array_lines(name, expression, "R", shape, "cond_capture_unit")
        lines += [f"WRITE(cond_capture_unit,'(A)') '{FOOTER}'", "CLOSE(cond_capture_unit)"]
        stages[stage] = "".join("       " + line + "\n" for line in lines)
    text = text.replace(import_anchor, imports + import_anchor)
    text = text.replace(declaration_anchor, declaration_anchor + declarations)
    bare_call = CALL.replace("    IF (lscond) CALL", "       CALL", 1)
    text = text.replace(CALL, "    IF (lscond) THEN\n" + stages["entry"] + bare_call + stages["exit"] + "    END IF\n")
    instrumented = text.encode().replace(b"\n", newline)
    patch = "".join(difflib.unified_diff(original.decode().splitlines(True), instrumented.decode().splitlines(True),
                                       fromfile=path.name, tofile=path.name))
    patch_path = path.parent / "condensation-state.patch"
    patch_path.write_text(patch)
    path.write_bytes(instrumented)
    return {"original_source_sha256": hashlib.sha256(original).hexdigest(),
            "instrumented_source_sha256": hashlib.sha256(instrumented).hexdigest(),
            "overlay_file": patch_path.name, "overlay_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def load_capture(directory: Path) -> dict:
    entry = parse_records((directory / FILES["entry"]).read_text())
    exit_state = parse_records((directory / FILES["exit"]).read_text())
    integers = {**INTEGERS, "nj3": ("cond_nj3", ())} if "nj3" in entry else INTEGERS
    if set(entry) != set(ENTRY) | set(integers) or set(exit_state) != set(EXIT):
        raise ValueError("Missing or unexpected condensation field")
    dims = entry["dimensions"]
    if dims["kind"] != "I" or dims["shape"] != [6]:
        raise ValueError("Invalid condensation dimensions record")
    sizes = dict(zip(("kproma", "kbdim", "klev", "fn2b", "nspecies", "ngas"), dims["values"]))
    if (any(value <= 0 for value in sizes.values()) or sizes["kproma"] > sizes["kbdim"] or
            sizes["nspecies"] != 5 or sizes["fn2b"] != 17):
        raise ValueError("Invalid condensation dimensions")
    for fields, records in ((ENTRY, entry), (EXIT, exit_state), (integers, entry)):
        for name, spec in fields.items():
            expected = [int(dim) if dim.isdigit() else sizes[dim] for dim in spec[1]]
            if records[name]["shape"] != expected or records[name]["kind"] != ("I" if fields is integers else "R"):
                raise ValueError(f"Wrong condensation shape/type: {name}")
    if entry["boundaries"]["values"] != [1, 3, 4, 10, 11, 17]:
        raise ValueError("Unsupported condensation bin boundaries")
    nsnucl, nsoa, bits = entry["options"]["values"]
    active = "nj3" in entry
    if (nsoa != 0 or bits != 64 or
            (active and (nsnucl not in (1, 2, 3) or entry["nj3"]["values"][0] not in (1, 2, 3))) or
            (not active and nsnucl != 0)):
        raise ValueError("Unsupported condensation nucleation/composition/precision configuration")
    sulfate, organic = entry["gas_indices"]["values"]
    if not 1 <= sulfate <= sizes["ngas"] or not 0 <= organic <= sizes["ngas"] or sulfate == organic:
        raise ValueError("Invalid condensation gas indices")
    for name in ("zdwet", "ptemp", "ppres", "massacc", "d_sa", "species_molar_mass", "species_density",
                 "avo", "argas", "ak", "p0sl_bg", "pi", "time_step_len"):
        if any(value <= 0 for value in entry[name]["values"]):
            raise ValueError(f"Nonpositive condensation input: {name}")
    for name in ("pnaero", "zvols", "pc_gas", "prh", "ptstep"):
        if any(value < 0 for value in entry[name]["values"]):
            raise ValueError(f"Negative condensation input: {name}")
    for name in EXIT:
        if any(value < 0 for value in exit_state[name]["values"]):
            raise ValueError(f"Negative condensation output: {name}")
    return {"schema_version": 1, "order": "Fortran column-major", "dimensions": sizes,
            "species": ["SO4", "OC", "BC", "SS", "DU"],
            "options": {"nsnucl": nsnucl, **({"nj3": entry["nj3"]["values"][0]} if active else {}), "nsoa": 0, "dp_bits": 64, "isubm_so4g": sulfate, "isubm_ocnv": organic},
            "stage": "after coagulation and optional second equilibration; immediately around condensation; before remapping",
            "entry": entry, "exit": exit_state, "units": {**{name: spec[2] for name, spec in ENTRY.items()}, **({"prh": "1"} if active else {})}}


def load_fixture(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or set(manifest.get("files", {})) != set(FILES.values()):
        raise ValueError("Invalid condensation fixture inventory")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Condensation fixture hash mismatch: {name}")
    return load_capture(directory)


def export_fixture(run_directory: Path, target: Path) -> None:
    """Export only a completed checked capture whose raw files match its manifest."""
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    capture = manifest.get("condensation_capture", {})
    if (manifest.get("status") != "first_microphysics_completed_unvalidated" or
            capture.get("status") != "captured" or
            "-fcheck=all" not in manifest.get("config", {}).get("build", {}).get("flags", [])):
        raise ValueError("A successful checked condensation capture is required")
    hashes = {name: hashlib.sha256((run_directory / name).read_bytes()).hexdigest() for name in FILES.values()}
    if hashes != capture.get("file_sha256"):
        raise ValueError("Run condensation files changed after manifest creation")
    state = load_capture(run_directory)
    fixture = {"schema_version": 1, "scope": "one condensation call; explicit captured nucleation options; not trajectory acceptance",
               "files": hashes, "run_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "source_revision": manifest["source_revision"], "working_tree_status": manifest["working_tree_status"],
               "source_sha256": manifest["source_sha256"], "input_sha256": manifest["input_sha256"],
               "compiler": manifest["compiler"], "config": manifest["config"],
               **({"reference_configuration": manifest["reference_configuration"]} if "reference_configuration" in manifest else {}),
               "dimensions": state["dimensions"], "options": state["options"], "stage": state["stage"],
               "condensation_overlay": {key: value for key, value in capture.items() if key not in ("state", "file_sha256")},
               "runtime_metadata": manifest["runtime_capture"]["metadata"],
               "scope_state_sha256": manifest["scope_capture"]["state_sha256"]}
    target.mkdir(parents=True, exist_ok=False)
    for name in FILES.values():
        shutil.copy2(run_directory / name, target / name)
    (target / "manifest.json").write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export a verified checked condensation fixture")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new fixture directory")
    args = parser.parse_args()
    export_fixture(args.run_directory, args.output)
