"""Build a versioned manifest and factual caption for a synthetic sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy.io import fits

SCHEMA_VERSION = "astrobridge.sample_manifest.v1"
LATER_TODO = (
    "Use particle/SSP or spectral-cube synthesis for spectrally redshifted "
    "observer-frame colors; do not treat broadband geometric reprojection as a K-correction."
)


def json_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.is_file():
        return {}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def image_filters(hdul: fits.HDUList, prefix: str) -> list[str]:
    return sorted(hdu.name.removeprefix(prefix) for hdu in hdul if hdu.name.startswith(prefix))


def dr1_properties(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with fits.open(path, memmap=False) as cube:
        properties: dict[str, float] = {}
        for extension, key in (("STARS_MASS", "stellar_mass_msun"), ("GAS_MASS", "gas_mass_msun"), ("SFR_INST", "sfr_msun_per_year")):
            if extension in cube:
                properties[key] = float(np.nansum(cube[extension].data))
        if "STARS_MASS" in cube and "STARS_MW_ZSOL" in cube:
            mass = np.asarray(cube["STARS_MASS"].data, dtype=float)
            metallicity = np.asarray(cube["STARS_MW_ZSOL"].data, dtype=float)
            total_mass = np.nansum(mass)
            if total_mass > 0:
                properties["stellar_mass_weighted_metallicity_zsun"] = float(np.nansum(mass * metallicity) / total_mass)
        return properties


def atlas_properties(catalog: Path | None, object_id: str) -> dict[str, float]:
    if catalog is None:
        return {}
    rows = np.atleast_2d(np.loadtxt(catalog, comments="#"))
    row = rows[rows[:, 0] == int(object_id)]
    if len(row) != 1:
        raise KeyError(f"Atlas subhalo {object_id} was not found in {catalog}.")
    values = row[0]
    return {
        "stellar_mass_msun": float(values[1]), "gas_mass_msun": float(values[2]),
        "dark_matter_mass_msun": float(values[3]), "sfr_msun_per_year": float(values[4]),
        "stellar_half_mass_radius_kpc": float(values[5]),
    }


def caption_for(manifest: dict[str, Any]) -> str:
    source = manifest["source"]
    observer = manifest["observer"]
    observation = manifest["observation"]
    color_text = {
        "spectrally_redshifted": "Colors were formed after spectral redshifting into the observer filters.",
        "native_observer_frame": "Colors are native to the rendered observer frame.",
        "geometric_reprojection_only": "Colors are from broadband maps reprojected geometrically; they are not K-corrected.",
    }[observer["color_treatment"]]
    kind = source["kind"].upper()
    simulation = source["simulation"]
    object_id = source["object_id"]
    snapshot = source["snapshot"]
    intrinsic_redshift = source["intrinsic_redshift"]
    observer_redshift = observer["redshift"]
    profile_id = observation["profile_id"]
    return (
        f"Synthetic {kind} image of {simulation} object {object_id} "
        f"(snapshot {snapshot}, intrinsic z={intrinsic_redshift:.3f}), "
        f"placed at observer redshift z={observer_redshift:.3f}. "
        f"It uses the {profile_id} observation profile. {color_text}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--source-kind", choices=("dr1", "atlas", "galsyn"), required=True)
    parser.add_argument("--simulation", required=True)
    parser.add_argument("--snapshot", type=int, required=True)
    parser.add_argument("--intrinsic-redshift", type=float, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--observer-redshift", type=float, required=True)
    parser.add_argument("--color-treatment", choices=("spectrally_redshifted", "native_observer_frame", "geometric_reprojection_only"), required=True)
    parser.add_argument("--ideal", type=Path, required=True)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgb", type=Path)
    parser.add_argument("--source-cube", type=Path)
    parser.add_argument("--atlas-catalog", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = yaml.safe_load(args.profile.read_text(encoding="utf-8"))
    with fits.open(args.ideal, memmap=False) as ideal, fits.open(args.observed, memmap=False) as observed:
        ideal_filters = image_filters(ideal, "DUST_LSST_")
        observed_filters = image_filters(observed, "OBS_LSST_")
        if ideal_filters != observed_filters:
            raise ValueError(f"Ideal and observed filters differ: {ideal_filters}, {observed_filters}")
        source_properties = atlas_properties(args.atlas_catalog, args.object_id) if args.source_kind == "atlas" else dr1_properties(args.source_cube)
        source_files = [str(args.ideal), str(args.observed)]
        if args.source_cube:
            source_files.append(str(args.source_cube))
        if args.atlas_catalog:
            source_files.append(str(args.atlas_catalog))
        source_files = list(dict.fromkeys(source_files))
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION, "sample_id": args.sample_id,
            "source": {"kind": args.source_kind, "simulation": args.simulation, "snapshot": args.snapshot, "intrinsic_redshift": args.intrinsic_redshift, "object_id": args.object_id, "physical_properties": source_properties, "source_files": source_files},
            "observer": {"redshift": args.observer_redshift, "color_treatment": args.color_treatment},
            "ideal": {"fits": str(args.ideal), "surface_brightness_unit": str(ideal["DUST_LSST_R"].header.get("BUNIT", "unknown")), "filters": ideal_filters, "pixel_scale_arcsec": float(ideal[0].header["PIXSIZE"])},
            "observation": {"profile_id": profile["profile"]["id"], "products": {"observed_fits": str(args.observed), "rgb": str(args.rgb) if args.rgb else None, "extensions": [hdu.name for hdu in observed]}, "operations": json_sidecar(args.observed).get("operations", []), "random_seed": profile.get("noise", {}).get("random_seed")},
            "provenance": {"profile": str(args.profile), "profile_references": profile["profile"].get("references", []), "ideal_sidecar": json_sidecar(args.ideal), "observation_sidecar": json_sidecar(args.observed)},
            "later_todos": [LATER_TODO],
        }
    manifest["caption"] = {"template_version": "factual_v1", "text": caption_for(manifest)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    caption_path = args.output.with_suffix(".caption.txt")
    caption_path.write_text(manifest["caption"]["text"] + "\n", encoding="utf-8")
    manifest["observation"]["products"]["caption"] = str(caption_path)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {args.output}")
    print(f"Wrote caption: {caption_path}")


if __name__ == "__main__":
    main()
