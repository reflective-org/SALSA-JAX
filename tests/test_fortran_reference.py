"""A failed reference setup must leave usable diagnostic provenance."""

import json
from pathlib import Path
import sys

from tools import fortran_reference


def test_missing_reference_source_records_failure(tmp_path, monkeypatch):
    config = Path(fortran_reference.ROOT / "configs/fortran-reference.toml")
    empty_source = tmp_path / "incomplete-checkout"
    empty_source.mkdir()
    destination = tmp_path / "reference"
    monkeypatch.setattr(fortran_reference, "ROOT", empty_source)
    monkeypatch.setattr(fortran_reference, "output", lambda *args: "test-revision")
    monkeypatch.setattr(sys, "argv", ["fortran_reference", "--config", str(config),
                                    "--output", str(destination)])

    assert fortran_reference.main() == 1
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["status"] == "setup_failed"
    assert ".depend" in manifest["error"]
    assert "build" not in manifest
