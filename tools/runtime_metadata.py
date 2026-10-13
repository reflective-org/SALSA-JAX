# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Copy-only driver instrumentation and strict parsing of runtime metadata."""

from __future__ import annotations

import difflib
import hashlib
import math
from pathlib import Path

FILENAME = "runtime-metadata.txt"
FIELDS = {
    "I": {
        "dp_kind": "KIND(1._dp)", "dp_bits": "STORAGE_SIZE(1._dp)",
        "dp_precision": "PRECISION(1._dp)", "dp_range": "RANGE(1._dp)",
        "default_real_bits": "STORAGE_SIZE(1.)", "kproma": "kproma",
        "kbdim": "kbdim", "klev": "klev", "krow": "krow",
        "ntrac": "ntrac", "nspec": "nspec", "nclass": "nclass", "nsol": "nsol",
        "naerocomp": "naerocomp", "subm_ngasspec": "subm_ngasspec",
        "nham_subm": "nham_subm", "nsnucl": "nsnucl", "nj3": "nj3",
        "ncd_activ": "ncd_activ", "nactivpdf": "nactivpdf", "nw": "nw",
        "in1a": "in1a", "fn1a": "fn1a", "in2a": "in2a", "fn2a": "fn2a",
        "in2b": "in2b", "fn2b": "fn2b", "isubm_so4g": "isubm_so4g",
        "isubm_ocnv": "capture_isubm_ocnv",
        "idt_cdnc": "idt_cdnc", "idt_icnc": "idt_icnc",
    },
    "L": {
        "lham": "lham", "laero_micro": "laero_micro",
        "lscoag": "capture_lscoag", "lscond": "capture_lscond",
        "locgas": "locgas", "lsol2b": "lsol2b", "lhydr2b": "lhydr2b",
        "recalc": "recalc", "lwetdep": "lwetdep", "ldrydep": "ldrydep",
        "lsedimentation": "lsedimentation",
    },
    "R": {
        "time_step_seconds": "time_step_len", "pressure_pa_1_1": "pap(1,1)",
        "temperature_k_1_1": "pt(1,1)", "specific_humidity_1_1": "pqm1(1,1)",
        "saturation_specific_humidity_1_1": "pqsm1(1,1)",
        "air_density_kg_m3_1_1": "zrhoa(1,1)", "cloud_fraction_1_1": "paclc(1,1)",
    },
}


def instrument_driver(path: Path) -> dict:
    """Overlay a copied driver; refuse missing/ambiguous anchors or repeat use."""
    original = path.read_bytes()
    text = original.decode()
    imports = (
        "  USE mo_ham, ONLY: capture_lscoag => lscoag, capture_lscond => lscond, &\n"
        "       capture_gasspec => subm_gasspec, capture_gasconv => subm_gasunitconv\n"
        "  USE mo_ham_subm_species, ONLY: capture_isubm_ocnv => isubm_ocnv\n"
    )
    lines = [
        "   ! SALSA_RUNTIME_METADATA_V1: generated copy-only instrumentation",
        f"   OPEN(NEWUNIT=capture_unit, FILE='{FILENAME}', STATUS='replace', ACTION='write')",
        "   WRITE(capture_unit,'(A)') 'SALSA_RUNTIME_METADATA 1'",
    ]
    formats = {"I": "I0", "L": "L1", "R": "ES26.17E3"}
    for kind, fields in FIELDS.items():
        for key, expression in fields.items():
            lines.append(f"   WRITE(capture_unit,'(A,1X,{formats[kind]})') '{kind} {key}', {expression}")
    lines.extend([
        "   DO capture_index = 1, subm_ngasspec",
        "      WRITE(capture_unit,'(A,3(1X,I0))') 'G', capture_index, &",
        "           capture_gasspec(capture_index), capture_gasconv(capture_index)",
        "   END DO",
        "   WRITE(capture_unit,'(A)') 'END_SALSA_RUNTIME_METADATA'",
        "   CLOSE(capture_unit)",
    ])
    replacements = [
        ("  IMPLICIT NONE\n", imports + "  IMPLICIT NONE\n"),
        ("  INTEGER :: ii, jj, kk\n", "  INTEGER :: ii, jj, kk\n  INTEGER :: capture_unit, capture_index\n"),
        ("   zfacn  = 1.0e-06_dp\n", "   zfacn  = 1.0e-06_dp\n" + "\n".join(lines) + "\n"),
    ]
    if "SALSA_RUNTIME_METADATA_V1" in text:
        raise ValueError("Driver already contains runtime metadata instrumentation")
    for anchor, _ in replacements:
        if text.count(anchor) != 1:
            raise ValueError(f"Expected exactly one runtime capture anchor: {anchor.strip()}")
    for anchor, replacement in replacements:
        text = text.replace(anchor, replacement, 1)
    instrumented = text.encode()
    patch = "".join(difflib.unified_diff(original.decode().splitlines(keepends=True),
                                       text.splitlines(keepends=True),
                                       fromfile="driver.F90", tofile="driver.F90"))
    overlay_path = path.parent / "runtime-metadata.patch"
    overlay_path.write_text(patch)
    path.write_bytes(instrumented)
    return {
        "original_driver_sha256": hashlib.sha256(original).hexdigest(),
        "instrumented_driver_sha256": hashlib.sha256(instrumented).hexdigest(),
        "overlay_sha256": hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
        "overlay_file": overlay_path.name,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def parse_metadata(text: str) -> dict:
    """Validate typed records, completeness, finite values and bounded maps."""
    lines = text.splitlines()
    if not lines or lines[0] != "SALSA_RUNTIME_METADATA 1" or lines[-1] != "END_SALSA_RUNTIME_METADATA":
        raise ValueError("Missing runtime metadata header/footer or unsupported schema")
    values, gas = {}, []
    for line in lines[1:-1]:
        parts = line.split()
        if len(parts) == 4 and parts[0] == "G":
            index, species, conversion = map(int, parts[1:])
            gas.append({"index": index, "species_index": species, "conversion_code": conversion})
            continue
        if len(parts) != 3 or parts[0] not in FIELDS or parts[1] not in FIELDS[parts[0]]:
            raise ValueError(f"Unknown or malformed metadata record: {line}")
        kind, key, raw = parts
        if key in values:
            raise ValueError(f"Duplicate metadata field: {key}")
        if kind == "L":
            if raw not in ("T", "F"):
                raise ValueError(f"Invalid logical value: {key}")
            value = raw == "T"
        elif kind == "I":
            value = int(raw)
        else:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite metadata field: {key}")
        values[key] = value
    required = {key for fields in FIELDS.values() for key in fields}
    if values.keys() != required:
        raise ValueError(f"Missing metadata fields: {sorted(required - values.keys())}")
    for key in ("dp_kind", "dp_bits", "dp_precision", "dp_range", "default_real_bits",
                "kproma", "kbdim", "klev", "krow", "ntrac", "nspec", "nclass", "nw"):
        if values[key] <= 0:
            raise ValueError(f"Nonpositive metadata field: {key}")
    if values["kproma"] > values["kbdim"] or not 0 <= values["nsol"] <= values["nclass"]:
        raise ValueError("Inconsistent runtime dimensions")
    if values["naerocomp"] < 0 or not 0 <= values["subm_ngasspec"] <= values["nspec"]:
        raise ValueError("Invalid runtime species counts")
    if not (1 <= values["in1a"] <= values["fn1a"] < values["in2a"] <= values["fn2a"]
            < values["in2b"] <= values["fn2b"]):
        raise ValueError("Inconsistent SALSA bin boundaries")
    if [row["index"] for row in gas] != list(range(1, values["subm_ngasspec"] + 1)):
        raise ValueError("Incomplete, duplicate or unordered gas mapping")
    if any(not 1 <= row["species_index"] <= values["nspec"] for row in gas):
        raise ValueError("Out-of-range gas species index")
    for key, upper in (("isubm_so4g", "subm_ngasspec"), ("isubm_ocnv", "subm_ngasspec"),
                       ("idt_cdnc", "ntrac"), ("idt_icnc", "ntrac")):
        if not 0 <= values[key] <= values[upper]:
            raise ValueError(f"Out-of-range runtime index: {key}")
    if values["time_step_seconds"] <= 0 or values["pressure_pa_1_1"] <= 0 or values["temperature_k_1_1"] <= 0:
        raise ValueError("Nonpositive timestep, pressure or temperature")
    return {"schema_version": 1, "stage": "initialized_before_time_loop", "values": values, "gas_species": gas}
