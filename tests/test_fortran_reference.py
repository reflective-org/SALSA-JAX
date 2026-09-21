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


def test_active_reference_profile_requires_first_call_scope(tmp_path, monkeypatch):
    import pytest
    target = tmp_path / 'must-not-be-created'
    monkeypatch.setattr(sys, 'argv', ['reference', '--config', str(fortran_reference.ROOT / 'configs/fortran-nucleation-active.toml'),
                                    '--output', str(target)])
    with pytest.raises(SystemExit) as error:
        fortran_reference.main()
    assert error.value.code == 2
    assert not target.exists()


def test_nested_capture_requires_explicit_profile(tmp_path, monkeypatch):
    import pytest
    target = tmp_path / 'must-not-be-created'
    monkeypatch.setattr(sys, 'argv', ['reference', '--scope', 'first-microphysics', '--capture-nucleation-state',
                                    '--output', str(target)])
    with pytest.raises(SystemExit) as error:
        fortran_reference.main()
    assert error.value.code == 2
    assert not target.exists()
