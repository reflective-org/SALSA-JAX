"""Adapt a hash-verified complete native SALSA boundary to composition inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np

from salsa_jax import constants
from tools.state_capture import load_fixture

ROOT = Path(__file__).resolve().parents[1]
FIXED_SOURCES = ("mo_physical_constants.F90", "mo_ham_salsa_init.F90", "mo_ham_salsactl.F90")
MATERIALS = ("SO4", "OC", "BC", "SS", "DU")
CONSTANT_NAMES = ("number_threshold", "cluster_molecules", "acid_diameter",
                  "organic_soluble_fraction", "saturation_limit", "ion_count",
                  "water_surface_tension", "activation_coefficient")


def record(capture, stage, name):
    item = capture[stage][name]
    return np.asarray(item["values"]).reshape(item["shape"], order="F")


def adapt_capture(capture, manifest):
    """Return cases, expected complete boundary outputs, and provenance metadata.

    Fixed constants not serialized by the old capture schema are accepted only
    with matching native source hashes. Configurable registry properties come
    from this capture itself; no other process fixture supplies missing values.
    """
    for name in FIXED_SOURCES:
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if manifest.get("source_sha256", {}).get(name) != actual:
            raise ValueError(f"Unsupported fixed-constant/initialization source: {name}")
    if not manifest.get("input_sha256") or not manifest.get("run_manifest_sha256"):
        raise ValueError("Missing native source/input provenance association")
    if "-fcheck=all" not in manifest.get("config", {}).get("build", {}).get("flags", []):
        raise ValueError("A checked native fixture is required")
    def entry(name):
        return record(capture, "entry", name)
    dimensions = capture["dimensions"]
    options = capture["options"]
    option_names = ("nsnucl", "nj3", "nsoa", "lscoag", "lscond", "recalc", "lhydr2b",
                    "lsol2b", "isubm_so4g", "isubm_ocnv", "dp_bits")
    if dict(zip(option_names, entry("options"))) != options:
        raise ValueError("Captured option records disagree")
    if manifest.get("dimensions") != dimensions or manifest.get("options") != options:
        raise ValueError("Manifest and captured dimensions/options disagree")
    runtime = manifest.get("runtime_metadata", {}).get("values", {})
    runtime_dimensions = {
        "kproma": "kproma", "kbdim": "kbdim", "klev": "klev", "fn2b": "nclass",
        "naerocomp": "naerocomp", "subm_ngasspec": "subm_ngasspec",
        "capture_nspec": "nspec", "capture_ntrac": "ntrac",
    }
    for name, runtime_name in runtime_dimensions.items():
        if runtime.get(runtime_name) != dimensions[name]:
            raise ValueError(f"Runtime/capture dimension mismatch: {name}")
    for name in ("nsnucl", "nj3", "lscoag", "lscond", "recalc", "lhydr2b", "lsol2b",
                 "isubm_so4g", "isubm_ocnv", "dp_bits"):
        if runtime.get(name) != options[name]:
            raise ValueError(f"Runtime/capture option mismatch: {name}")
    if (dimensions["fn2b"] != 17 or dimensions["naerocomp"] != 69 or
            dimensions["subm_naerospec_nowat"] != 5 or options["nsoa"] != 0 or options["dp_bits"] != 64):
        raise ValueError("Unsupported native bin/material/precision layout")
    if options["nsnucl"] not in (0, 1, 2, 3) or options["nj3"] not in (1, 2, 3):
        raise ValueError("Unsupported native nucleation options")
    if any(options[name] not in (0, 1) for name in ("lscoag", "lscond", "recalc", "lhydr2b", "lsol2b")):
        raise ValueError("Invalid native process switch")
    for name, expected in (("avogadro", constants.AVOGADRO), ("pi", constants.PI)):
        if float(entry(name)) != expected:
            raise ValueError(f"Unsupported captured fixed constant: {name}")
    captured_constants = dict(zip(CONSTANT_NAMES, entry("constants")))
    for name, expected in (("cluster_molecules", constants.NUCLEATION_CLUSTER_MOLECULES),
                           ("acid_diameter", constants.SULFURIC_ACID_DIAMETER),
                           ("activation_coefficient", constants.NUCLEATION_ACTIVATION_COEFFICIENT)):
        if captured_constants[name] != expected:
            raise ValueError(f"Unsupported captured fixed constant: {name}")
    limits = entry("reglim")
    if limits.shape != (4,) or np.any(limits <= 0) or np.any(np.diff(limits) <= 0):
        raise ValueError("Invalid regime limits")
    if limits[0] != constants.NUCLEATION_TARGET_DIAMETER:
        raise ValueError("Unsupported captured nucleation target diameter")
    names = capture["mapping"]["species_names"]
    ids = entry("volume_species").astype(int)
    if (len(set(names)) != len(names) or ids.shape != (5,) or
            any(not 1 <= index <= len(names) for index in ids) or
            tuple(names[index-1] for index in ids) != MATERIALS or
            not np.array_equal(ids, entry("hydration_species")) or "WAT" not in names):
        raise ValueError("Unsupported native material registry")
    water = names.index("WAT")
    masses, densities = entry("species_molar_mass"), entry("species_density")
    if np.any(masses[ids-1] <= 0) or np.any(densities[ids-1] <= 0) or masses[water] <= 0 or densities[water] <= 0:
        raise ValueError("Nonpositive native material properties")
    component_map = entry("material_map").astype(int)
    present = component_map > 0
    expected_present = np.ones((17, 5), dtype=bool)
    expected_present[:3, 2:] = False
    expected_present[10:, 3] = False
    if (component_map.shape != (17, 5) or not np.array_equal(present, expected_present) or
            not np.array_equal(np.sort(component_map[present]), np.arange(1, 70))):
        raise ValueError("Unsupported native component map")
    for bin_index, material in zip(*np.where(present)):
        component = component_map[bin_index, material] - 1
        if (entry("component_bin")[component] != bin_index+1 or
                entry("component_species")[component] != ids[material]):
            raise ValueError("Inconsistent native component map")
    component_map = component_map - 1  # native absent zero becomes explicit -1
    gas_ids = entry("gas_species").astype(int)
    gas_names = tuple(names[index-1] for index in gas_ids)
    organic = options["isubm_ocnv"]
    if (gas_names not in (("SO2", "H2SO4"), ("SO2", "H2SO4", "OC")) or
            options["isubm_so4g"] != 2 or organic != (3 if len(gas_names) == 3 else 0) or
            np.any(entry("gas_conversion") != 2)):
        raise ValueError("Unsupported canonical gas mapping")
    runtime_gas = manifest["runtime_metadata"].get("gas_species", [])
    if runtime_gas != [{"index": i+1, "species_index": int(index), "conversion_code": 2}
                      for i, index in enumerate(gas_ids)]:
        raise ValueError("Runtime/capture gas registry mismatch")
    number = entry("pnaero")
    horizontal, vertical, bins = number.shape
    count = horizontal * vertical
    pbl = entry("ppbl")
    if pbl.shape != (horizontal,) or np.any(pbl < 1) or np.any(pbl > vertical):
        raise ValueError("Invalid native boundary-layer index")
    for name in ("dpmid", "vlolim", "vhilim", "vratiohi", "vratiolo"):
        if not np.array_equal(entry(name), record(capture, "exit", name)):
            raise ValueError(f"Native geometry changed across SALSA: {name}")
    def scalar(value):
        return np.full(count, value)
    def fixed(array):
        return np.broadcast_to(array, (count,) + np.shape(array)).copy()
    def field(stage, name):
        array = record(capture, stage, name)
        return array.reshape((count,) + array.shape[2:])
    def gas(stage):
        values = field(stage, "pc_gas")
        result = np.zeros((count, 3))
        result[:, :values.shape[-1]] = values
        return result
    cases = {
        "labels": [f"native-h{h+1}-level{k+1}" for h in range(horizontal) for k in range(vertical)],
        "number": number.reshape(count, 17), "mass": field("entry", "paerml"), "gas": gas("entry"),
        "temperature": field("entry", "ptemp"), "relative_humidity": field("entry", "prh"),
        "pressure": field("entry", "ppres"), "timestep": scalar(float(entry("ptstep"))),
        "boundary_layer": (np.arange(1, vertical+1)[None, :] >= np.trunc(pbl)[:, None]).reshape(count),
        "component_map": fixed(component_map), "regime_limits": fixed(limits),
        "dry_diameter": fixed(entry("dpmid")), "lower_volume": fixed(entry("vlolim")),
        "upper_volume": fixed(entry("vhilim")), "mass_accommodation": fixed(entry("massacc")),
        "species_molar_mass": fixed(masses[ids-1]), "species_density": fixed(densities[ids-1]),
        "water_molar_mass": scalar(masses[water]), "water_density": scalar(densities[water]),
        **{name: scalar(captured_constants[name]) for name in (
            "number_threshold", "organic_soluble_fraction", "saturation_limit", "ion_count", "water_surface_tension")},
        **{name: scalar(bool(options[native])) for name, native in (
            ("coagulation_enabled", "lscoag"), ("condensation_enabled", "lscond"),
            ("recalculate", "recalc"), ("transfer_soluble", "lsol2b"), ("hydrate_insoluble", "lhydr2b"))},
        "organic_enabled": scalar(bool(organic)), "scheme": scalar(options["nsnucl"]),
        "growth": scalar(options["nj3"]),
    }
    for name in ("number", "mass", "gas", "relative_humidity"):
        if np.any(cases[name] < 0):
            raise ValueError(f"Negative native input: {name}")
    for name in ("temperature", "pressure", "timestep", "number_threshold"):
        if np.any(cases[name] <= 0):
            raise ValueError(f"Nonpositive native input: {name}")
    expected = {"gas": gas("exit"), **{key: field("exit", native) for key, native in (
        ("number", "pnaero"), ("mass", "paerml"), ("volume", "zvols"),
        ("core", "zcore"), ("wet_diameter", "zdwet"), ("dry_radius_cm", "pm6dry"),
        ("wet_radius_cm", "pm6rp"), ("water_mass", "pww"), ("density", "prhop"))}}
    metadata = {
        "native_cell_shape": [horizontal, vertical], "flatten_order": "C after Fortran record reconstruction",
        "material_order": list(MATERIALS), "gas_order": ["SO2", "H2SO4", "OC"],
        "mass_units": "SO4 molecules cm-3; other materials ug m-3", "density_units": "g cm-3",
        "component_map": "zero-based component indices; -1 means absent", "options": dict(options),
        "source_revision": manifest.get("source_revision"), "source_sha256": manifest["source_sha256"],
        "input_sha256": manifest["input_sha256"], "run_manifest_sha256": manifest["run_manifest_sha256"],
        "reference_configuration": manifest.get("reference_configuration"),
        "fixed_sources_verified": list(FIXED_SOURCES),
        "solubility_source": "native initialized lhydr2b mapping; initialization source hash verified",
    }
    return cases, expected, metadata


def load_native_case(directory=None):
    directory = Path(directory) if directory is not None else ROOT / "tests/fixtures/native-default"
    capture = load_fixture(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    return adapt_capture(capture, manifest)
