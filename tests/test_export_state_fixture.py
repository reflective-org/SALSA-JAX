"""Fixture export requires checked, intact captured evidence and a new directory."""
import json
from pathlib import Path
import shutil

import pytest

from tools.export_state_fixture import export_fixture
from tools.state_capture import load_fixture

FIXTURE = Path(__file__).resolve().parent / "fixtures/native-default"


def make_run(directory):
    shutil.copytree(FIXTURE, directory)
    fixture = json.loads((directory / "manifest.json").read_text())
    manifest = {key: fixture[key] for key in ("source_revision", "working_tree_status", "source_sha256", "input_sha256", "compiler", "config")}
    manifest.update(status="first_microphysics_completed_unvalidated",
                    salsa_capture={"status": "captured", "file_sha256": fixture["files"]},
                    runtime_capture={"metadata": fixture["runtime_metadata"]},
                    scope_capture={"state_sha256": fixture["scope_state_sha256"]})
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_export_roundtrip_and_no_overwrite(tmp_path):
    run, target = tmp_path / "run", tmp_path / "fixture"
    make_run(run)
    export_fixture(run, target)
    assert load_fixture(target) == load_fixture(FIXTURE)
    before = (target / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        export_fixture(run, target)
    assert (target / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("failure", ["unchecked", "failed_run", "changed_data"])
def test_unaccepted_export_leaves_no_target(tmp_path, failure):
    run, target = tmp_path / "run", tmp_path / "fixture"
    manifest = make_run(run)
    if failure == "unchecked":
        manifest["config"]["build"]["flags"].remove("-fcheck=all")
    elif failure == "failed_run":
        manifest["status"] = "run_failed"
    else:
        path = run / "salsa-entry.txt"
        path.write_bytes(path.read_bytes() + b"\n")
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        export_fixture(run, target)
    assert not target.exists()


@pytest.mark.parametrize("active", [False, True])
def test_configured_export_preserves_exact_profile_provenance(tmp_path, active):
    run, target = tmp_path / "run", tmp_path / "fixture"
    manifest = make_run(run)
    manifest["config"]["process_switches"] = {"recalc": True}
    profile = {"settings": None, "process_switches": {"recalc": True},
               "original_driver_sha256": "original", "configured_driver_sha256": "configured",
               "overlay_sha256": "patch", "scope": "first-microphysics"}
    if active:
        settings = {"nsnucl": 2, "nj3": 1, "sulfate_molecules_m3": 1e14}
        manifest["config"]["reference_case"] = settings
        profile["settings"] = settings
    manifest["reference_configuration"] = profile
    manifest["config_path"] = "/captured/profile.toml"
    (run / "manifest.json").write_text(json.dumps(manifest))
    export_fixture(run, target)
    fixture = json.loads((target / "manifest.json").read_text())
    assert fixture["reference_configuration"] == profile
    assert fixture["regeneration"]["config"] == manifest["config"]
    assert fixture["regeneration"]["config_path_at_capture"] == manifest["config_path"]
    assert "default" not in fixture["scope"]
    assert fixture["input_sha256"] == manifest["input_sha256"]
