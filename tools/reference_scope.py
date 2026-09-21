"""Explicit first-call reference scope applied only to a copied box driver."""

import difflib
import hashlib
from pathlib import Path

SCOPE = "first-microphysics"
FILENAME = "reference-scope.txt"
FINITE_FIELDS = ("pxtm1", "pxtte", "zrwet", "zm6dry", "zrhop", "zww")
COMPLETION_LINES = (
    "SALSA_REFERENCE_SCOPE 1", "scope first-microphysics",
    "completed_calls 1", "END_SALSA_REFERENCE_SCOPE",
)
CALL_END_ANCHOR = (
    "          paclc, pgrvolm1, zpbl)                                 ! cloud cover, grid box volume, boundary layer top level\n"
    "      DO jn=1, nclass\n"
)


def instrument_scope(path: Path) -> dict:
    """Stop after a finite first microphysics call; preserve overlay hash chain."""
    original = path.read_bytes()
    text = original.decode()
    lines = ["      ! SALSA_FIRST_MICROPHYSICS_SCOPE_V1: copy-only finite-state gate"]
    for field in FINITE_FIELDS:
        lines.append(f"      IF (.NOT. ALL(scope_is_finite({field}))) ERROR STOP 'nonfinite {field}'")
    lines.append(f"      OPEN(NEWUNIT=scope_unit, FILE='{FILENAME}', STATUS='replace', ACTION='write')")
    lines.extend(f"      WRITE(scope_unit,'(A)') '{line}'" for line in COMPLETION_LINES)
    lines.extend(["      CLOSE(scope_unit)", "      STOP"])
    replacements = [
        ("  IMPLICIT NONE\n",
         "  USE, INTRINSIC :: ieee_arithmetic, ONLY: scope_is_finite => ieee_is_finite\n  IMPLICIT NONE\n"),
        ("  INTEGER :: ii, jj, kk\n", "  INTEGER :: ii, jj, kk\n  INTEGER :: scope_unit\n"),
        (CALL_END_ANCHOR, CALL_END_ANCHOR.replace("      DO jn=1, nclass\n",
                                               "\n".join(lines) + "\n      DO jn=1, nclass\n")),
    ]
    if "SALSA_FIRST_MICROPHYSICS_SCOPE_V1" in text:
        raise ValueError("Driver already contains first-microphysics scope instrumentation")
    for anchor, _ in replacements:
        if text.count(anchor) != 1:
            raise ValueError(f"Expected exactly one reference scope anchor: {anchor.strip()}")
    for anchor, replacement in replacements:
        text = text.replace(anchor, replacement, 1)
    patch = "".join(difflib.unified_diff(original.decode().splitlines(keepends=True),
                                       text.splitlines(keepends=True),
                                       fromfile="driver.F90", tofile="driver.F90"))
    overlay = path.parent / "reference-scope.patch"
    overlay.write_text(patch)
    path.write_bytes(text.encode())
    return {
        "input_driver_sha256": hashlib.sha256(original).hexdigest(),
        "instrumented_driver_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "overlay_file": overlay.name, "overlay_sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "finite_fields": list(FINITE_FIELDS),
    }


def parse_completion(text: str) -> dict:
    """Accept only the complete first-call marker, never a full-trajectory claim."""
    if text.splitlines() != list(COMPLETION_LINES):
        raise ValueError("Invalid first-microphysics completion record")
    return {"schema_version": 1, "scope": SCOPE, "completed_calls": 1}
