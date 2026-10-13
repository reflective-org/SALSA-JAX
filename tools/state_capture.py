"""Copy-only native SALSA boundary capture for the standalone five-species case."""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

if __package__:
    from .array_records import HEADER, FOOTER, array_lines, parse_records
else:
    from array_records import HEADER, FOOTER, array_lines, parse_records

FILES = {"entry": "salsa-entry.txt", "exit": "salsa-exit.txt", "mapping": "salsa-mapping.txt"}
# Dimensions describe the serialized active domain, omitting inactive padding.
COMMON = {
    "pc_gas": ("pc_gas(1:kproma,:,:)", ("kproma", "klev", "subm_ngasspec"), "molecules m-3"),
    "paerml": ("paerml(1:kproma,:,:)", ("kproma", "klev", "naerocomp"), "component-dependent: sulfate molecules cm-3; others ug m-3"),
    "pnaero": ("pnaero(1:kproma,:,:)", ("kproma", "klev", "fn2b"), "particles m-3"),
}
GEOMETRY = {name: (name, ("fn2b",), unit) for name, unit in
            (("dpmid", "m"), ("vlolim", "m3 particle-1"), ("vhilim", "m3 particle-1"),
             ("vratiohi", "1"), ("vratiolo", "1"))}
ENTRY = {
    **COMMON, **GEOMETRY,
    **{name: (f"{name}(1:kproma,:)", ("kproma", "klev"), unit)
       for name, unit in (("ppres", "Pa"), ("prh", "1"), ("ptemp", "K"))},
    "ppbl": ("ppbl(1:kproma)", ("kproma",), "one-based level index stored as real"),
    "ptstep": ("ptstep", (), "s"),
    "avogadro": ("avo", (), "mol-1"),
    "pi": ("pi", (), "1"),
    "reglim": ("reglim", ("4",), "m"),
    "massacc": ("massacc", ("fn2b",), "1"),
    "constants": ("[nlim,n3,d_sa,epsoc,slim,ions,surfw0,act_coeff]", ("8",),
                  "ordered: nlim particles/m3,n3 molecules/particle,d_sa m,epsoc 1,slim 1,ions 1,surfw0 J/m2,act_coeff s-1"),
    "species_molar_mass": ("speclist(1:capture_nspec)%moleweight", ("capture_nspec",), "g mol-1"),
    "species_density": ("speclist(1:capture_nspec)%density", ("capture_nspec",), "kg m-3 for aerosol species; gas defaults not physical densities"),
}
EXIT = {**COMMON, **GEOMETRY,
        **{name: (f"{name}(1:kproma,:,:)", ("kproma", "klev", "fn2b"), unit)
           for name, unit in (("pm6rp", "cm"), ("pm6dry", "cm"), ("prhop", "g cm-3"),
                              ("pww", "kg water m-3 air"), ("zcore", "m3 particle-1"), ("zdwet", "m"))},
        "zvols": ("zvols(1:kproma,:,:,:)", ("kproma", "klev", "fn2b", "subm_naerospec_nowat"), "m3 material m-3 air")}
INTEGERS = {
    "dimensions": ("[kproma,kbdim,klev,fn2b,naerocomp,subm_ngasspec,capture_nspec,capture_ntrac,subm_naerospec_nowat]", ("9",)),
    "options": ("[nsnucl,nj3,nsoa,MERGE(1,0,lscoag),MERGE(1,0,lscond),MERGE(1,0,recalc), &\n         MERGE(1,0,lhydr2b),MERGE(1,0,lsol2b),isubm_so4g,isubm_ocnv,STORAGE_SIZE(1._dp)]", ("11",)),
    "material_map": ("state_material_map", ("fn2b", "5")),
    "volume_species": ("[id_so4,id_oc,id_bc,id_ss,id_du]", ("5",)),
    "hydration_species": ("capture_aerospec(1:subm_naerospec_nowat)", ("subm_naerospec_nowat",)),
    "gas_species": ("capture_gasspec(1:subm_ngasspec)", ("subm_ngasspec",)),
    "gas_conversion": ("capture_gasconv(1:subm_ngasspec)", ("subm_ngasspec",)),
    "gas_tracer": ("[(speclist(capture_gasspec(state_index))%idt, &\n         state_index=1,subm_ngasspec)]", ("subm_ngasspec",)),
    "component_bin": ("capture_aerocomp(1:naerocomp)%iclass", ("naerocomp",)),
    "component_species": ("capture_aerocomp(1:naerocomp)%spid", ("naerocomp",)),
    "component_tracer": ("capture_aerocomp(1:naerocomp)%idt", ("naerocomp",)),
    "number_tracer": ("sizeclass(1:fn2b)%idt_no", ("fn2b",)),
}


def instrument_salsa(path: Path) -> dict:
    original = path.read_bytes()
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    text = original.decode().replace("\r\n", "\n")
    imports = """    USE mo_ham, ONLY: capture_aerocomp => aerocomp, &
         capture_gasspec => subm_gasspec, capture_gasconv => subm_gasunitconv, &
         capture_aerospec => subm_aerospec_nowat
    USE mo_species, ONLY: capture_nspec => nspec
    USE mo_tracdef, ONLY: capture_ntrac => ntrac, capture_trlist => trlist
    USE mo_ham_salsactl, ONLY: vlolim,vhilim,vratiohi,vratiolo,reglim,massacc, &
         n3,d_sa,epsoc,slim,ions,surfw0,nsnucl,nj3,lhydr2b,lsol2b,act_coeff
"""
    declarations = """    INTEGER :: state_unit, state_index
    INTEGER, SAVE :: state_call_count = 0
    INTEGER :: state_material_map(fn2b,5)
"""
    entry = ["! SALSA_NATIVE_STATE_V1: copy-only instrumentation", "state_call_count=state_call_count+1",
             "IF (state_call_count /= 1) ERROR STOP 'native capture requires exactly one SALSA call'",
             "IF (nsoa /= 0 .OR. subm_naerospec_nowat /= 5) ERROR STOP 'unsupported native capture composition'",
             "state_material_map=0", "state_material_map(:,1)=iso4b", "state_material_map(:,2)=iocb",
             "state_material_map(in2a:fn2b,3)=ibcb(in2a:fn2b)",
             "state_material_map(in2a:fn2a,4)=issb(in2a:fn2a)",
             "state_material_map(in2a:fn2b,5)=idub(in2a:fn2b)"]
    for stage, fields in (("entry", ENTRY), ("exit", EXIT)):
        lines = [f"OPEN(NEWUNIT=state_unit,FILE='{FILES[stage]}',STATUS='new',ACTION='write')",
                 f"WRITE(state_unit,'(A)') '{HEADER}'"]
        if stage == "entry":
            for name, (expression, shape) in INTEGERS.items():
                lines += array_lines(name, expression, "I", shape)
        for name, (expression, shape, _) in fields.items():
            lines += array_lines(name, expression, "R", shape)
        lines += [f"WRITE(state_unit,'(A)') '{FOOTER}'", "CLOSE(state_unit)"]
        if stage == "entry":
            entry += lines
        else:
            exit_lines = lines
    entry += [f"OPEN(NEWUNIT=state_unit,FILE='{FILES['mapping']}',STATUS='new',ACTION='write')",
              "WRITE(state_unit,'(A)') 'SALSA_MAPPING 1'", "DO state_index=1,capture_nspec",
              "  WRITE(state_unit,'(A,1X,I0)') 'S',state_index",
              "  WRITE(state_unit,'(A)') TRIM(speclist(state_index)%shortname)", "END DO",
              "DO state_index=1,capture_ntrac",
              "  WRITE(state_unit,'(A,4(1X,I0))') 'T',state_index,capture_trlist%ti(state_index)%spid, &",
              "       capture_trlist%ti(state_index)%mode,capture_trlist%ti(state_index)%nphase",
              "  WRITE(state_unit,'(A)') TRIM(capture_trlist%ti(state_index)%fullname)",
              "  WRITE(state_unit,'(A)') TRIM(capture_trlist%ti(state_index)%units)", "END DO",
              "WRITE(state_unit,'(A)') 'END_SALSA_MAPPING'", "CLOSE(state_unit)"]
    anchor = "    zmvsu = (speclist(id_so4)%moleweight/1000.)/avo/speclist(id_so4)%density\n"
    replacements = [("    IMPLICIT NONE\n", imports + "    IMPLICIT NONE\n"),
                    ("    INTEGER :: ii, jj, kk, nn\n", "    INTEGER :: ii, jj, kk, nn\n" + declarations),
                    (anchor, "\n".join("    " + line for line in entry) + "\n" + anchor),
                    ("  END SUBROUTINE salsa\n", "\n".join("    " + line for line in exit_lines) + "\n  END SUBROUTINE salsa\n")]
    if "SALSA_NATIVE_STATE_V1" in text:
        raise ValueError("SALSA source already instrumented")
    for anchor, _ in replacements:
        if text.count(anchor) != 1:
            raise ValueError(f"Expected one native capture anchor: {anchor.strip()}")
    for anchor, replacement in replacements:
        text = text.replace(anchor, replacement)
    instrumented = text.encode().replace(b"\n", newline)
    patch = "".join(difflib.unified_diff(original.decode().splitlines(True), instrumented.decode().splitlines(True),
                                       fromfile=path.name, tofile=path.name))
    patch_path = path.parent / "salsa-state.patch"
    patch_path.write_text(patch)
    path.write_bytes(instrumented)
    return {"original_source_sha256": hashlib.sha256(original).hexdigest(),
            "instrumented_source_sha256": hashlib.sha256(instrumented).hexdigest(),
            "overlay_file": patch_path.name, "overlay_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def parse_mapping(text: str, nspec: int, ntrac: int) -> dict:
    lines = text.splitlines()
    if not lines or lines[0] != "SALSA_MAPPING 1" or lines[-1] != "END_SALSA_MAPPING":
        raise ValueError("Invalid mapping header/footer")
    rows = iter(lines[1:-1])
    species, tracers = [], []
    try:
        for index in range(1, nspec + 1):
            if next(rows) != f"S {index}":
                raise ValueError("Missing/duplicate/unordered species record")
            name = next(rows)
            if not name:
                raise ValueError("Empty species name")
            species.append(name)
        for index in range(1, ntrac + 1):
            row = next(rows).split()
            if len(row) != 5 or row[0] != "T" or int(row[1]) != index:
                raise ValueError("Missing/duplicate/unordered tracer record")
            spid, mode, phase = map(int, row[2:])
            name, units = next(rows), next(rows)
            if not name or not 0 <= spid <= nspec:
                raise ValueError("Invalid tracer name/species index")
            tracers.append({"name": name, "units": units, "species": spid, "bin": mode, "phase": phase})
        if next(rows, None) is not None:
            raise ValueError("Unexpected mapping record")
    except StopIteration as error:
        raise ValueError("Truncated mapping") from error
    return {"species_names": species, "tracers": tracers}


def load_capture(directory: Path) -> dict:
    entry = parse_records((directory / FILES["entry"]).read_text())
    exit_state = parse_records((directory / FILES["exit"]).read_text())
    if set(entry) != set(ENTRY) | set(INTEGERS) or set(exit_state) != set(EXIT):
        raise ValueError("Missing or unexpected native state field")
    dims = entry["dimensions"]
    if dims["kind"] != "I" or dims["shape"] != [9]:
        raise ValueError("Invalid native dimensions record")
    names = ("kproma", "kbdim", "klev", "fn2b", "naerocomp", "subm_ngasspec", "capture_nspec", "capture_ntrac", "subm_naerospec_nowat")
    sizes = dict(zip(names, dims["values"]))
    if any(v <= 0 for v in sizes.values()) or sizes["kproma"] > sizes["kbdim"] or sizes["subm_naerospec_nowat"] != 5:
        raise ValueError("Invalid native dimensions")
    for fields, records in ((ENTRY, entry), (EXIT, exit_state), (INTEGERS, entry)):
        for name, spec in fields.items():
            expected = [int(d) if d.isdigit() else sizes[d] for d in spec[1]]
            if records[name]["shape"] != expected or records[name]["kind"] != ("I" if fields is INTEGERS else "R"):
                raise ValueError(f"Wrong shape/type for {name}")
    for name, upper in (("gas_species", "capture_nspec"), ("volume_species", "capture_nspec"),
                        ("component_species", "capture_nspec"), ("component_bin", "fn2b"),
                        ("component_tracer", "capture_ntrac"), ("number_tracer", "capture_ntrac"),
                        ("gas_tracer", "capture_ntrac")):
        if any(not 1 <= v <= sizes[upper] for v in entry[name]["values"]):
            raise ValueError(f"Out-of-range mapping: {name}")
    mapping = parse_mapping((directory / FILES["mapping"]).read_text(), sizes["capture_nspec"], sizes["capture_ntrac"])
    for prefix in ("component", "gas"):
        for species, tracer in zip(entry[prefix + "_species"]["values"], entry[prefix + "_tracer"]["values"]):
            if mapping["tracers"][tracer-1]["species"] != species:
                raise ValueError("Inconsistent tracer/species mapping")
    # HAM tracer phases: GAS=1, AEROSOLMASS=2, AEROSOLNUMBER=-2.
    for index, tracer in enumerate(entry["component_tracer"]["values"]):
        record = mapping["tracers"][tracer-1]
        if (record["bin"] != entry["component_bin"]["values"][index] or
                record["phase"] != 2 or record["units"] != "kg kg-1"):
            raise ValueError("Inconsistent component tracer bin/phase/units")
    for bin_index, tracer in enumerate(entry["number_tracer"]["values"], 1):
        record = mapping["tracers"][tracer-1]
        if (record["bin"] != bin_index or record["species"] != 0 or
                record["phase"] != -2 or record["units"] != "1 kg-1"):
            raise ValueError("Inconsistent number tracer bin/species/phase/units")
    for tracer in entry["gas_tracer"]["values"]:
        record = mapping["tracers"][tracer-1]
        if record["bin"] != 0 or record["phase"] != 1 or record["units"] != "kg kg-1":
            raise ValueError("Inconsistent gas tracer bin/phase/units")
    # The supported native gas boundary uses HAM's molecules/cm3 code (2),
    # followed by the wrapper's factor of 1e6 to molecules/m3.
    if any(code != 2 for code in entry["gas_conversion"]["values"]):
        raise ValueError("Unsupported gas conversion for molecular boundary")
    material = entry["material_map"]["values"]
    if entry["hydration_species"]["values"] != entry["volume_species"]["values"]:
        raise ValueError("Hydration species order differs from native volume order")
    used = [v for v in material if v]
    if sorted(used) != list(range(1, sizes["naerocomp"] + 1)):
        raise ValueError("Material mapping must address each component exactly once")
    for slot, component in enumerate(material):
        if component:
            bin_index = slot % sizes["fn2b"] + 1
            species = entry["volume_species"]["values"][slot // sizes["fn2b"]]
            if (entry["component_bin"]["values"][component-1] != bin_index or
                    entry["component_species"]["values"][component-1] != species):
                raise ValueError("Inconsistent bin/species component mapping")
    option_names = ("nsnucl", "nj3", "nsoa", "lscoag", "lscond", "recalc", "lhydr2b", "lsol2b", "isubm_so4g", "isubm_ocnv", "dp_bits")
    options = dict(zip(option_names, entry["options"]["values"]))
    if options["nsoa"] != 0 or options["dp_bits"] != 64:
        raise ValueError("Unsupported composition or precision")
    if not 1 <= options["isubm_so4g"] <= sizes["subm_ngasspec"] or not 0 <= options["isubm_ocnv"] <= sizes["subm_ngasspec"]:
        raise ValueError("Invalid optional/required gas index")
    sulfate_species = entry["gas_species"]["values"][options["isubm_so4g"]-1]
    if mapping["species_names"][sulfate_species-1] != "H2SO4":
        raise ValueError("Required sulfate gas slot must identify H2SO4")
    return {"schema_version": 1, "order": "Fortran column-major", "dimensions": sizes, "options": options,
            "entry": entry, "exit": exit_state, "mapping": mapping,
            "units": {name: spec[2] for name, spec in (ENTRY | EXIT).items()}}


def load_fixture(directory: Path) -> dict:
    """Verify every captured file against the reviewed fixture manifest."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or set(manifest.get("files", {})) != set(FILES.values()):
        raise ValueError("Invalid fixture file inventory")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Fixture hash mismatch: {name}")
    return load_capture(directory)
