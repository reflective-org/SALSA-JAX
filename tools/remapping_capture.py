# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Capture both actual remapping/correction passes in a copied SALSA source."""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import shutil

if __package__:
    from .array_records import HEADER, FOOTER, array_lines, parse_records
    from .hydration_capture import ENTRY as HYDRATION_ENTRY, INPUTS as HYDRATION_INPUTS
else:
    from array_records import HEADER, FOOTER, array_lines, parse_records
    from hydration_capture import ENTRY as HYDRATION_ENTRY, INPUTS as HYDRATION_INPUTS

STAGES = ("entry", "remap_exit", "cleanup_exit", "correction_exit")
FILES = {f"pass{number}_{stage}": f"remapping-pass-{number}-{stage.replace('_', '-')}.txt"
         for number in (1, 2) for stage in STAGES}
STATE = {
    "pnaero": ("pnaero(1:kproma,:,:)", ("kproma", "klev", "fn2b"), "particles m-3"),
    "zvols": ("zvols(1:kproma,:,:,:)", ("kproma", "klev", "fn2b", "nspecies"), "m3 material m-3 air"),
    "pc_gas": ("pc_gas(1:kproma,:,:)", ("kproma", "klev", "ngas"), "molecules m-3"),
    "ptemp": ("ptemp(1:kproma,:)", ("kproma", "klev"), "K"),
    "prh": ("prh(1:kproma,:)", ("kproma", "klev"), "1"),
}
CONTEXT = {
    name: (spec[0].replace("hyd_", "remap_"), spec[1], spec[2])
    for name, spec in HYDRATION_ENTRY.items() if name not in HYDRATION_INPUTS
}
CONTEXT.update({
    "vlolim": ("remap_vlolim", ("fn2b",), "m3 particle-1"),
    "vhilim": ("remap_vhilim", ("fn2b",), "m3 particle-1"),
    "ions": ("remap_ions", (), "1"),
    "slim": ("remap_slim", (), "1"),
    "argas": ("remap_argas", (), "J mol-1 K-1"),
})
CORRECTIONS = {
    "zcore": ("zcore(1:kproma,:,:)", ("kproma", "klev", "fn2b"), "m3 particle-1"),
    "zdwet": ("zdwet(1:kproma,:,:)", ("kproma", "klev", "fn2b"), "m"),
    "pm6dry": ("pm6dry(1:kproma,:,:)", ("kproma", "klev", "fn2b"), "cm"),
    "pm6rp": ("pm6rp(1:kproma,:,:)", ("kproma", "klev", "fn2b"), "cm"),
    "pww": ("pww(1:kproma,:,:)", ("kproma", "klev", "fn2b"), "kg water m-3 air"),
}
INTEGERS = {
    "dimensions": ("[kproma,kbdim,klev,fn2b,subm_naerospec_nowat,subm_ngasspec,remap_nspec]", ("7",)),
    "boundaries": ("[in1a,fn1a,in2a,fn2a,in2b,fn2b]", ("6",)),
    "options": ("[MERGE(1,0,remap_lsol2b),MERGE(1,0,remap_lhydr2b),MERGE(1,0,recalc),nsoa,STORAGE_SIZE(1._dp)]", ("5",)),
    "soluble": ("MERGE(1,0,sizeclass(1:fn2b)%lsoluble)", ("fn2b",)),
    "species_indices": ("remap_map(1:subm_naerospec_nowat)", ("nspecies",)),
    "standard_species_indices": ("[id_so4,id_oc,id_bc,id_ss,id_du]", ("5",)),
    "water_species_index": ("id_wat", ()),
    "gas_indices": ("[isubm_so4g,isubm_ocnv]", ("2",)),
}
LOOP = "    DO nn = 1, 2\n"
CALL = """       CALL distr_update(kproma, kbdim, klev, &
                         pnaero, zvols, ptemp)
"""
HYDRATION_CALL = """       CALL equilibration(kproma, kbdim, klev,        &
            pnaero, zvols, prh, ptemp,  &
            zcore,  zdwet) 
"""
CORRECTION_END = """       pww(1:kproma,:,1:fn2b)    = ((pi/6.0)*zdwet(1:kproma,:,1:fn2b)**3 -                            &
                                   zcore(1:kproma,:,1:fn2b))*speclist(id_wat)%density*pnaero(1:kproma,:,1:fn2b)

    END DO
"""


def _stage_lines(stage):
    fields = {**STATE, **(CONTEXT if stage == "entry" else {}),
              **(CORRECTIONS if stage == "correction_exit" else {})}
    lines = [f"WRITE(remap_filename,'(A,I0,A)') 'remapping-pass-',nn,'-{stage.replace('_', '-')}.txt'",
             "OPEN(NEWUNIT=remap_unit,FILE=TRIM(remap_filename),STATUS='new',ACTION='write')",
             f"WRITE(remap_unit,'(A)') '{HEADER}'"]
    lines += array_lines("pass_index", "nn", "I", (), "remap_unit")
    records = {**(INTEGERS if stage == "entry" else {}), **fields}
    for name, spec in records.items():
        expression, shape = spec[:2]
        shape = tuple({"nspecies": "subm_naerospec_nowat", "ngas": "subm_ngasspec"}.get(dim, dim) for dim in shape)
        generated = array_lines(name, expression, "I" if name in INTEGERS else "R", shape, "remap_unit")
        # The long options constructor needs a continuation under free-form limits.
        if name == "options":
            generated[-1] = generated[-1].replace(",MERGE(1,0,recalc)", ", &\n     MERGE(1,0,recalc)")
        lines += generated
    lines += [f"WRITE(remap_unit,'(A)') '{FOOTER}'", "CLOSE(remap_unit)"]
    return "".join("       " + line + "\n" for line in lines)


def instrument_salsa_remapping(path: Path) -> dict:
    """Wrap two remap calls and subsequent cleanup/hydration/conversions only."""
    original = path.read_bytes()
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    text = original.decode().replace("\r\n", "\n")
    if "SALSA_REMAPPING_STATE_V1" in text:
        raise ValueError("Remapping source already instrumented")
    import_anchor = "    IMPLICIT NONE\n"
    declaration_anchor = "    INTEGER :: ii, jj, kk, nn\n"
    for anchor in (import_anchor, declaration_anchor, LOOP, CALL, HYDRATION_CALL, CORRECTION_END):
        if text.count(anchor) != 1:
            raise ValueError(f"Expected one remapping capture anchor: {anchor.strip()}")
    imports = """    USE mo_ham, ONLY: remap_map=>subm_aerospec_nowat
    USE mo_species, ONLY: remap_nspec=>nspec
    USE mo_ham_salsactl, ONLY: remap_lsol2b=>lsol2b,remap_lhydr2b=>lhydr2b
    USE mo_ham_salsactl, ONLY: remap_surfw0=>surfw0,remap_epsoc=>epsoc,remap_reglim=>reglim
    USE mo_ham_salsactl, ONLY: remap_vlolim=>vlolim,remap_vhilim=>vhilim,remap_ions=>ions,remap_slim=>slim
    USE mo_physical_constants, ONLY: remap_ak=>ak,remap_argas=>argas
"""
    declarations = """    INTEGER :: remap_unit, remap_js
    INTEGER, SAVE :: remap_capture_count=0
    CHARACTER(LEN=64) :: remap_filename
    REAL(dp) :: remap_molar_mass(subm_naerospec_nowat), remap_density(subm_naerospec_nowat)
"""
    setup = """    ! SALSA_REMAPPING_STATE_V1: copy-only two-pass capture
    remap_capture_count=remap_capture_count+1
    IF (remap_capture_count /= 1) ERROR STOP 'remapping capture requires one salsa call'
    IF (nsoa /= 0 .OR. subm_naerospec_nowat /= 5 .OR. fn2b /= 17) &
         ERROR STOP 'unsupported remapping capture configuration'
    DO remap_js=1,subm_naerospec_nowat
       remap_molar_mass(remap_js)=speclist(remap_map(remap_js))%moleweight
       remap_density(remap_js)=speclist(remap_map(remap_js))%density
    END DO
"""
    text = text.replace(import_anchor, imports + import_anchor)
    text = text.replace(declaration_anchor, declaration_anchor + declarations)
    text = text.replace(LOOP, setup + LOOP)
    text = text.replace(CALL, _stage_lines("entry") + CALL + _stage_lines("remap_exit"))
    text = text.replace(HYDRATION_CALL, _stage_lines("cleanup_exit") + HYDRATION_CALL)
    text = text.replace(CORRECTION_END, CORRECTION_END.replace("\n    END DO\n",
                        "\n" + _stage_lines("correction_exit") + "    END DO\n"))
    instrumented = text.encode().replace(b"\n", newline)
    patch = "".join(difflib.unified_diff(original.decode().splitlines(True), instrumented.decode().splitlines(True),
                                       fromfile=path.name, tofile=path.name))
    patch_path = path.parent / "remapping-state.patch"
    patch_path.write_text(patch)
    path.write_bytes(instrumented)
    return {"original_source_sha256": hashlib.sha256(original).hexdigest(),
            "instrumented_source_sha256": hashlib.sha256(instrumented).hexdigest(),
            "overlay_file": patch_path.name, "overlay_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "hydration_schema_sha256": hashlib.sha256(Path(__file__).with_name("hydration_capture.py").read_bytes()).hexdigest()}


def load_capture(directory: Path) -> dict:
    records = {stage: parse_records((directory / name).read_text()) for stage, name in FILES.items()}
    first = records["pass1_entry"]
    if "dimensions" not in first or first["dimensions"]["kind"] != "I" or first["dimensions"]["shape"] != [7]:
        raise ValueError("Invalid remapping dimensions record")
    sizes = dict(zip(("kproma", "kbdim", "klev", "fn2b", "nspecies", "ngas", "nspec"),
                     first["dimensions"]["values"]))
    if (any(value <= 0 for value in sizes.values()) or sizes["kproma"] > sizes["kbdim"] or
            sizes["fn2b"] != 17 or sizes["nspecies"] != 5):
        raise ValueError("Invalid remapping dimensions")
    passes = []
    for number in (1, 2):
        stages = {}
        for stage in STAGES:
            state = records[f"pass{number}_{stage}"]
            specs = {**STATE, **(CONTEXT if stage == "entry" else {}),
                     **(INTEGERS if stage == "entry" else {}),
                     **(CORRECTIONS if stage == "correction_exit" else {}),
                     "pass_index": ("nn", ())}
            if set(state) != set(specs):
                raise ValueError(f"Missing or unexpected remapping fields: pass {number} {stage}")
            for name, spec in specs.items():
                expected = [int(dim) if dim.isdigit() else sizes[dim] for dim in spec[1]]
                kind = "I" if name in INTEGERS or name == "pass_index" else "R"
                if state[name]["shape"] != expected or state[name]["kind"] != kind:
                    raise ValueError(f"Wrong remapping shape/type: {name}")
            if state["pass_index"]["values"] != [number]:
                raise ValueError("Incorrect remapping pass index")
            for name in ("pnaero", "zvols", "pc_gas", "prh"):
                if any(value < 0 for value in state[name]["values"]):
                    raise ValueError(f"Negative remapping state: {name}")
            if any(value <= 0 for value in state["ptemp"]["values"]):
                raise ValueError("Nonpositive remapping temperature")
            if stage == "correction_exit":
                for name in ("zcore", "zdwet", "pm6dry", "pm6rp"):
                    if any(value <= 0 for value in state[name]["values"]):
                        raise ValueError(f"Nonpositive corrected property: {name}")
            stages[stage] = state
        entry, remapped, cleaned, corrected = (stages[name] for name in STAGES)
        for name in (*CONTEXT, *INTEGERS):
            if entry[name] != first[name]:
                raise ValueError(f"Remapping context changed: {name}")
        for name in ("pc_gas", "ptemp", "prh"):
            if remapped[name] != entry[name]:
                raise ValueError(f"Remapping changed pure input: {name}")
        for name in ("ptemp", "prh"):
            if cleaned[name] != remapped[name]:
                raise ValueError(f"Cleanup changed meteorology: {name}")
        for name in STATE:
            if corrected[name] != cleaned[name]:
                raise ValueError(f"Correction changed pure input: {name}")
        if number == 2:
            for name in STATE:
                if entry[name] != passes[0]["correction_exit"][name]:
                    raise ValueError(f"Discontinuous remapping passes: {name}")
        passes.append({"index": number, **stages})
    if first["boundaries"]["values"] != [1, 3, 4, 10, 11, 17]:
        raise ValueError("Unsupported remapping bin boundaries")
    soluble_transfer, hydration, recalc, nsoa, bits = first["options"]["values"]
    if any(value not in (0, 1) for value in (soluble_transfer, hydration, recalc)) or nsoa != 0 or bits != 64:
        raise ValueError("Unsupported remapping options")
    if any(value not in (0, 1) for value in first["soluble"]["values"]):
        raise ValueError("Invalid remapping solubility flags")
    species = first["species_indices"]["values"]
    water = first["water_species_index"]["values"][0]
    if (species != first["standard_species_indices"]["values"] or len(set(species)) != 5 or
            any(not 1 <= value <= sizes["nspec"] for value in species) or
            not 1 <= water <= sizes["nspec"] or water in species):
        raise ValueError("Invalid remapping species mapping")
    sulfate, organic = first["gas_indices"]["values"]
    if not 1 <= sulfate <= sizes["ngas"] or not 0 <= organic <= sizes["ngas"] or sulfate == organic:
        raise ValueError("Invalid remapping gas indices")
    for name in CONTEXT:
        if name not in ("nlim", "epsoc") and any(value <= 0 for value in first[name]["values"]):
            raise ValueError(f"Nonpositive remapping context: {name}")
    if first["nlim"]["values"][0] < 0 or not 0 <= first["epsoc"]["values"][0] <= 1:
        raise ValueError("Invalid remapping threshold or organic solubility")
    if first["slim"]["values"][0] <= 1:
        raise ValueError("Remapping supersaturation limit must exceed one")
    if any(lo >= hi for lo, hi in zip(first["vlolim"]["values"], first["vhilim"]["values"])):
        raise ValueError("Invalid remapping bin volume limits")
    return {"schema_version": 1, "order": "Fortran column-major", "dimensions": sizes,
            "species": ["SO4", "OC", "BC", "SS", "DU"],
            "options": {"lsol2b": bool(soluble_transfer), "lhydr2b": bool(hydration),
                        "recalc": bool(recalc), "nsoa": nsoa, "dp_bits": bits,
                        "isubm_so4g": sulfate, "isubm_ocnv": organic},
            "stage": "two distr_update passes, each followed by tiny cleanup and hydration/radius/water correction",
            "context": {name: first[name] for name in (*CONTEXT, *INTEGERS)},
            "passes": passes,
            "units": {name: spec[2] for name, spec in {**STATE, **CONTEXT, **CORRECTIONS}.items()}}


def load_fixture(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or set(manifest.get("files", {})) != set(FILES.values()):
        raise ValueError("Invalid remapping fixture inventory")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Remapping fixture hash mismatch: {name}")
    return load_capture(directory)


def export_fixture(run_directory: Path, target: Path) -> None:
    """Export only a completed checked capture with unchanged raw records."""
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    capture = manifest.get("remapping_capture", {})
    if (manifest.get("status") != "first_microphysics_completed_unvalidated" or
            capture.get("status") != "captured" or
            "-fcheck=all" not in manifest.get("config", {}).get("build", {}).get("flags", [])):
        raise ValueError("A successful checked remapping capture is required")
    hashes = {name: hashlib.sha256((run_directory / name).read_bytes()).hexdigest() for name in FILES.values()}
    if hashes != capture.get("file_sha256"):
        raise ValueError("Run remapping files changed after manifest creation")
    state = load_capture(run_directory)
    fixture = {"schema_version": 1, "scope": "two native remapping/correction passes; not full trajectory acceptance",
               "files": hashes, "run_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "source_revision": manifest["source_revision"], "working_tree_status": manifest["working_tree_status"],
               "source_sha256": manifest["source_sha256"], "input_sha256": manifest["input_sha256"],
               "compiler": manifest["compiler"], "config": manifest["config"],
               **({"reference_configuration": manifest["reference_configuration"]} if "reference_configuration" in manifest else {}),
               "dimensions": state["dimensions"], "options": state["options"], "stage": state["stage"],
               "remapping_overlay": {key: value for key, value in capture.items() if key not in ("state", "file_sha256")},
               "runtime_metadata": manifest["runtime_capture"]["metadata"],
               "scope_state_sha256": manifest["scope_capture"]["state_sha256"]}
    target.mkdir(parents=True, exist_ok=False)
    for name in FILES.values():
        shutil.copy2(run_directory / name, target / name)
    (target / "manifest.json").write_text(json.dumps(fixture, indent=2) + "\n")
