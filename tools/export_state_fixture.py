"""Export a small verified native-state fixture from a successful checked capture."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

if __package__:
    from .state_capture import FILES, load_capture
else:
    from state_capture import FILES, load_capture


def export_fixture(run_directory: Path, target: Path) -> None:
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    capture = manifest.get("salsa_capture", {})
    if (manifest.get("status") != "first_microphysics_completed_unvalidated" or
            capture.get("status") != "captured" or
            "-fcheck=all" not in manifest.get("config", {}).get("build", {}).get("flags", [])):
        raise ValueError("A successful checked native capture is required")
    hashes = {name: hashlib.sha256((run_directory / name).read_bytes()).hexdigest() for name in FILES.values()}
    if hashes != capture.get("file_sha256"):
        raise ValueError("Run capture files changed after manifest creation")
    state = load_capture(run_directory)
    fixture = {
        "schema_version": 1, "scope": "one configured native SALSA call; not trajectory acceptance",
        "files": hashes, "run_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_revision": manifest["source_revision"], "working_tree_status": manifest["working_tree_status"],
        "source_sha256": manifest["source_sha256"], "input_sha256": manifest["input_sha256"],
        "compiler": manifest["compiler"], "config": manifest["config"],
        **({"reference_configuration": manifest["reference_configuration"]} if "reference_configuration" in manifest else {}),
        "dimensions": state["dimensions"], "options": state["options"],
        "salsa_overlay": {key: value for key, value in capture.items() if key not in ("state", "file_sha256")},
        "runtime_metadata": manifest["runtime_capture"]["metadata"],
        "scope_state_sha256": manifest["scope_capture"]["state_sha256"],
        "regeneration": {
            "config": manifest["config"],
            "config_path_at_capture": manifest.get("config_path"),
            "instructions": "Use the recorded complete TOML configuration with --scope first-microphysics "
                            "--capture-salsa-state --run and a new output directory, then export_state_fixture.py.",
        },
    }
    target.mkdir(parents=True, exist_ok=False)
    for name in FILES.values():
        shutil.copy2(run_directory / name, target / name)
    (target / "manifest.json").write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new fixture directory")
    args = parser.parse_args()
    export_fixture(args.run_directory, args.output)
