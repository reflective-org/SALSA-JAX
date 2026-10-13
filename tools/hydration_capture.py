"""Capture the first actual equilibration call in a copied SALSA source."""
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

FILES = {"entry": "hydration-entry.txt", "exit": "hydration-exit.txt"}
INPUTS = {
    "pnaero": ("pnaero(1:kproma,:,:)", ("kproma", "klev", "fn2b"), "particles m-3"),
    "pvols": ("zvols(1:kproma,:,:,:)", ("kproma", "klev", "fn2b", "nspecies"), "m3 material m-3 air"),
    "prh": ("prh(1:kproma,:)", ("kproma", "klev"), "1; source clips to [0.05,0.95]"),
    "ptemp": ("ptemp(1:kproma,:)", ("kproma", "klev"), "K"),
}
ENTRY = {
    **INPUTS,
    "species_molar_mass": ("hyd_molar_mass", ("nspecies",), "g mol-1; native nonwater species order"),
    "species_density": ("hyd_density", ("nspecies",), "kg m-3; native nonwater species order"),
    "water_molar_mass": ("speclist(id_wat)%moleweight", (), "g mol-1"),
    "water_density": ("speclist(id_wat)%density", (), "kg m-3"),
    "dpmid": ("dpmid", ("fn2b",), "m"),
    "reglim": ("hyd_reglim", ("4",), "m"),
    "nlim": ("nlim", (), "particles m-3"),
    "surfw0": ("hyd_surfw0", (), "J m-2"),
    "epsoc": ("hyd_epsoc", (), "1"),
    "ak": ("hyd_ak", (), "J K-1"),
    "avo": ("avo", (), "mol-1"),
    "pi": ("pi", (), "1"),
}
EXIT = {
    **INPUTS,
    "pcore": ("zcore(1:kproma,:,:)", ("kproma", "klev", "fn2b"), "m3 particle-1"),
    "pdwet": ("zdwet(1:kproma,:,:)", ("kproma", "klev", "fn2b"), "m"),
}
INTEGERS = {
    "dimensions": ("[kproma,kbdim,klev,fn2b,subm_naerospec_nowat,hyd_nspec]", ("6",)),
    "boundaries": ("[in1a,fn1a,in2a,fn2a,in2b,fn2b]", ("6",)),
    "options": ("[MERGE(1,0,hyd_lhydr2b),nsoa,STORAGE_SIZE(1._dp)]", ("3",)),
    "recalc": ("MERGE(1,0,recalc)", ()),
    "soluble": ("MERGE(1,0,sizeclass(1:fn2b)%lsoluble)", ("fn2b",)),
    "species_indices": ("hyd_map(1:subm_naerospec_nowat)", ("nspecies",)),
    "standard_species_indices": ("[id_so4,id_oc,id_bc,id_ss,id_du]", ("5",)),
    "water_species_index": ("id_wat", ()),
}
CALL = """    CALL equilibration(kproma, kbdim, klev,        &
         pnaero, zvols, prh, ptemp,  &
         zcore,  zdwet) 
"""


def instrument_salsa_hydration(path: Path) -> dict:
    """Wrap only initial hydration; later optional/repeated call sites are intact."""
    original = path.read_bytes()
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    text = original.decode().replace("\r\n", "\n")
    if "SALSA_HYDRATION_STATE_V1" in text:
        raise ValueError("Hydration source already instrumented")
    import_anchor = "    IMPLICIT NONE\n"
    declaration_anchor = "    INTEGER :: ii, jj, kk, nn\n"
    for anchor in (import_anchor, declaration_anchor, CALL):
        if text.count(anchor) != 1:
            raise ValueError(f"Expected one hydration capture anchor: {anchor.strip()}")
    imports = """    USE mo_ham, ONLY: hyd_map=>subm_aerospec_nowat
    USE mo_species, ONLY: hyd_nspec=>nspec
    USE mo_ham_salsactl, ONLY: hyd_lhydr2b=>lhydr2b,hyd_surfw0=>surfw0,hyd_epsoc=>epsoc
    USE mo_ham_salsactl, ONLY: hyd_reglim=>reglim
    USE mo_physical_constants, ONLY: hyd_ak=>ak
"""
    declarations = """    INTEGER :: hyd_capture_unit, hyd_js
    INTEGER, SAVE :: hyd_capture_count=0
    REAL(dp) :: hyd_molar_mass(subm_naerospec_nowat), hyd_density(subm_naerospec_nowat)
"""
    stages = {}
    for stage, fields in (("entry", ENTRY), ("exit", EXIT)):
        lines = []
        if stage == "entry":
            lines += ["! SALSA_HYDRATION_STATE_V1: copy-only instrumentation",
                      "hyd_capture_count=hyd_capture_count+1",
                      "IF (hyd_capture_count /= 1) ERROR STOP 'hydration capture requires one call'",
                      "IF (nsoa /= 0 .OR. subm_naerospec_nowat /= 5 .OR. fn2b /= 17) &",
                      "     ERROR STOP 'unsupported hydration capture configuration'",
                      "DO hyd_js=1,subm_naerospec_nowat",
                      "   hyd_molar_mass(hyd_js)=speclist(hyd_map(hyd_js))%moleweight",
                      "   hyd_density(hyd_js)=speclist(hyd_map(hyd_js))%density",
                      "END DO"]
        lines += [f"OPEN(NEWUNIT=hyd_capture_unit,FILE='{FILES[stage]}',STATUS='new',ACTION='write')",
                  f"WRITE(hyd_capture_unit,'(A)') '{HEADER}'"]
        for name, spec in ({**INTEGERS, **fields} if stage == "entry" else fields).items():
            expression, shape = spec[:2]
            shape = tuple("subm_naerospec_nowat" if dim == "nspecies" else dim for dim in shape)
            lines += array_lines(name, expression, "I" if name in INTEGERS else "R", shape, "hyd_capture_unit")
        lines += [f"WRITE(hyd_capture_unit,'(A)') '{FOOTER}'", "CLOSE(hyd_capture_unit)"]
        stages[stage] = "".join("    " + line + "\n" for line in lines)
    text = text.replace(import_anchor, imports + import_anchor)
    text = text.replace(declaration_anchor, declaration_anchor + declarations)
    text = text.replace(CALL, stages["entry"] + CALL + stages["exit"])
    instrumented = text.encode().replace(b"\n", newline)
    patch = "".join(difflib.unified_diff(original.decode().splitlines(True), instrumented.decode().splitlines(True),
                                       fromfile=path.name, tofile=path.name))
    patch_path = path.parent / "hydration-state.patch"
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
        raise ValueError("Missing or unexpected hydration field")
    dims = entry["dimensions"]
    if dims["kind"] != "I" or dims["shape"] != [6]:
        raise ValueError("Invalid hydration dimensions record")
    sizes = dict(zip(("kproma", "kbdim", "klev", "fn2b", "nspecies", "nspec"), dims["values"]))
    if (any(value <= 0 for value in sizes.values()) or sizes["kproma"] > sizes["kbdim"] or
            sizes["nspecies"] != 5 or sizes["fn2b"] != 17):
        raise ValueError("Invalid hydration dimensions")
    for fields, records in ((ENTRY, entry), (EXIT, exit_state), (INTEGERS, entry)):
        for name, spec in fields.items():
            expected = [int(dim) if dim.isdigit() else sizes[dim] for dim in spec[1]]
            if records[name]["shape"] != expected or records[name]["kind"] != ("I" if fields is INTEGERS else "R"):
                raise ValueError(f"Wrong hydration shape/type: {name}")
    if entry["boundaries"]["values"] != [1, 3, 4, 10, 11, 17]:
        raise ValueError("Unsupported hydration bin boundaries")
    hydration, nsoa, bits = entry["options"]["values"]
    if hydration not in (0, 1) or nsoa != 0 or bits != 64:
        raise ValueError("Unsupported hydration options")
    if any(value not in (0, 1) for value in entry["soluble"]["values"]):
        raise ValueError("Invalid hydration solubility flag")
    if entry["recalc"]["values"][0] not in (0, 1):
        raise ValueError("Invalid hydration recalc flag")
    species = entry["species_indices"]["values"]
    water = entry["water_species_index"]["values"][0]
    if (species != entry["standard_species_indices"]["values"] or len(set(species)) != 5 or
            any(not 1 <= value <= sizes["nspec"] for value in species) or
            not 1 <= water <= sizes["nspec"] or water in species):
        raise ValueError("Invalid hydration species mapping")
    for name in INPUTS:
        if entry[name] != exit_state[name]:
            raise ValueError(f"Hydration changed pure input: {name}")
    for name in ("ptemp", "species_molar_mass", "species_density", "water_molar_mass", "water_density",
                 "dpmid", "reglim", "surfw0", "ak", "avo", "pi"):
        if any(value <= 0 for value in entry[name]["values"]):
            raise ValueError(f"Nonpositive hydration input: {name}")
    for name in ("pnaero", "pvols", "prh", "nlim"):
        if any(value < 0 for value in entry[name]["values"]):
            raise ValueError(f"Negative hydration input: {name}")
    if not 0 <= entry["epsoc"]["values"][0] <= 1:
        raise ValueError("Invalid soluble organic fraction")
    for name in ("pcore", "pdwet"):
        if any(value <= 0 for value in exit_state[name]["values"]):
            raise ValueError(f"Nonpositive hydration output: {name}")
    return {"schema_version": 1, "order": "Fortran column-major", "dimensions": sizes,
            "species": ["SO4", "OC", "BC", "SS", "DU"],
            "options": {"lhydr2b": bool(hydration), "recalc": bool(entry["recalc"]["values"][0]),
                        "nsoa": nsoa, "dp_bits": bits},
            "stage": "first unconditional equilibration; before coagulation; later recalc calls excluded",
            "entry": entry, "exit": exit_state,
            "units": {name: spec[2] for name, spec in {**ENTRY, **EXIT}.items()}}


def load_fixture(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or set(manifest.get("files", {})) != set(FILES.values()):
        raise ValueError("Invalid hydration fixture inventory")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Hydration fixture hash mismatch: {name}")
    return load_capture(directory)


def export_fixture(run_directory: Path, target: Path) -> None:
    """Export a completed checked first-call capture without blessing changed data."""
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    capture = manifest.get("hydration_capture", {})
    if (manifest.get("status") != "first_microphysics_completed_unvalidated" or
            capture.get("status") != "captured" or
            "-fcheck=all" not in manifest.get("config", {}).get("build", {}).get("flags", [])):
        raise ValueError("A successful checked hydration capture is required")
    hashes = {name: hashlib.sha256((run_directory / name).read_bytes()).hexdigest() for name in FILES.values()}
    if hashes != capture.get("file_sha256"):
        raise ValueError("Run hydration files changed after manifest creation")
    state = load_capture(run_directory)
    fixture = {"schema_version": 1, "scope": "first equilibration call only; not trajectory acceptance",
               "files": hashes, "run_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "source_revision": manifest["source_revision"], "working_tree_status": manifest["working_tree_status"],
               "source_sha256": manifest["source_sha256"], "input_sha256": manifest["input_sha256"],
               "compiler": manifest["compiler"], "config": manifest["config"],
               **({"reference_configuration": manifest["reference_configuration"]} if "reference_configuration" in manifest else {}),
               "dimensions": state["dimensions"], "options": state["options"], "stage": state["stage"],
               "hydration_overlay": {key: value for key, value in capture.items() if key not in ("state", "file_sha256")},
               "runtime_metadata": manifest["runtime_capture"]["metadata"],
               "scope_state_sha256": manifest["scope_capture"]["state_sha256"]}
    target.mkdir(parents=True, exist_ok=False)
    for name in FILES.values():
        shutil.copy2(run_directory / name, target / name)
    (target / "manifest.json").write_text(json.dumps(fixture, indent=2) + "\n")
