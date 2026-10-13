# Copyright 2026 Reflective
# SPDX-License-Identifier: Apache-2.0
"""Full-precision, shape-tagged Fortran array records with strict stdlib parsing."""
import math
import re

HEADER = "SALSA_ARRAYS 1"
FOOTER = "END_SALSA_ARRAYS"


def array_lines(name, expression, kind, dimensions, unit="state_unit"):
    if kind not in ("R", "I") or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("Invalid array name/type")
    suffix = ", " + ", ".join(dimensions) if dimensions else ""
    header = f"WRITE({unit},'(A,*(1X,I0))') '{kind} {name} {len(dimensions)}'{suffix}"
    return [header, f"WRITE({unit},'({'ES26.17E3' if kind == 'R' else 'I0'})') {expression}"]


def parse_records(text):
    lines = text.splitlines()
    if not lines or lines[0] != HEADER or lines[-1] != FOOTER:
        raise ValueError("Invalid array header/footer")
    position, result = 1, {}
    while position < len(lines) - 1:
        fields = lines[position].split()
        position += 1
        if len(fields) < 3 or fields[0] not in ("R", "I") or not re.fullmatch(r"[a-z][a-z0-9_]*", fields[1]):
            raise ValueError("Invalid array record")
        kind, name = fields[:2]
        rank = int(fields[2])
        shape = [int(v) for v in fields[3:]]
        if rank < 0 or len(shape) != rank or any(v <= 0 for v in shape) or name in result:
            raise ValueError("Invalid rank/shape or duplicate array")
        count = math.prod(shape)
        if count > len(lines) - 1 - position:
            raise ValueError("Truncated array")
        raw = lines[position:position+count]
        if any(len(value.split()) != 1 for value in raw):
            raise ValueError("Expected one array element per line")
        values = [(int(v) if kind == "I" else float(v)) for v in raw]
        if kind == "R" and not all(math.isfinite(v) for v in values):
            raise ValueError("Nonfinite array element")
        result[name] = {"kind": kind, "shape": shape, "values": values}
        position += count
    return result
