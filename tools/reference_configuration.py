# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Explicit first-call settings applied only to an isolated standalone driver."""
from __future__ import annotations
import difflib
import hashlib
import math
from pathlib import Path


def configure_driver(path: Path, settings: dict | None = None, *, process_switches: dict | None = None) -> dict:
    """Configure supported nucleation forcing and recalc on a copied driver.

    Switches are assigned before start_ham. Sulfate forcing, when requested,
    precedes the unchanged native molecular/tracer conversions. The harness
    restricts either profile section to the first-microphysics scope.
    """
    if settings is None and process_switches is None:
        raise ValueError("A reference_case or process_switches profile is required")
    if settings is not None:
        if set(settings) != {"nsnucl", "nj3", "sulfate_molecules_m3"}:
            raise ValueError("reference_case requires exactly nsnucl, nj3, sulfate_molecules_m3")
        for key in ("nsnucl", "nj3"):
            if type(settings[key]) is not int or settings[key] not in (1, 2, 3):
                raise ValueError(f"reference_case {key} must be an integer in 1,2,3")
        sulfate = settings["sulfate_molecules_m3"]
        if type(sulfate) not in (int, float) or not math.isfinite(sulfate) or sulfate <= 0:
            raise ValueError("reference_case sulfate_molecules_m3 must be finite and positive")
    if process_switches is not None:
        if set(process_switches) != {"recalc"} or type(process_switches["recalc"]) is not bool:
            raise ValueError("process_switches requires exactly a boolean recalc")
    original = path.read_bytes()
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    text = original.decode().replace("\r\n", "\n")
    if "SALSA_REFERENCE_CONFIGURATION_V1" in text:
        raise ValueError("Driver already has reference configuration")
    start = "  CALL start_ham\n"
    forcing = "      zgas(:,:,isubm_so4g) = sulfate_pdf_value * H2SO4_scaling_factor\n"
    anchors = [start] + ([forcing] if settings is not None else [])
    for anchor in anchors:
        if text.count(anchor) != 1:
            raise ValueError(f"Expected one reference configuration anchor: {anchor.strip()}")
    assignments = "  ! SALSA_REFERENCE_CONFIGURATION_V1: explicit copy-only test profile\n"
    if settings is not None:
        assignments += f'  nsnucl = {settings["nsnucl"]}\n  nj3 = {settings["nj3"]}\n'
        text = text.replace(forcing, forcing + f"      IF (ii == 1) zgas(:,:,isubm_so4g) = {sulfate:.17e}_dp\n")
    if process_switches is not None:
        assignments += "  recalc = " + (".TRUE." if process_switches["recalc"] else ".FALSE.") + "\n"
    text = text.replace(start, assignments + start)
    updated = text.encode().replace(b"\n", newline)
    patch = "".join(difflib.unified_diff(original.decode().splitlines(True), updated.decode().splitlines(True),
                                       fromfile=path.name, tofile=path.name))
    overlay = path.parent / "reference-configuration.patch"
    overlay.write_text(patch)
    path.write_bytes(updated)
    return {"settings": settings, **({"process_switches": process_switches} if process_switches is not None else {}),
            "scope": "first-microphysics",
            "original_driver_sha256": hashlib.sha256(original).hexdigest(),
            "configured_driver_sha256": hashlib.sha256(updated).hexdigest(),
            "overlay_file": overlay.name, "overlay_sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            **({"forcing_units": "H2SO4 molecules m-3; before unchanged native gas/tracer conversions"}
               if settings is not None else {})}
