"""Build a mock-LSST-observation dataset from the TNG50-SKIRT Atlas.

Config-driven batch driver: for every (subhalo_id, observer, redshift)
combination named in the YAML config, this chains the two single-object
scripts that already do the real work --

    prepare_atlas_mock_sky.py  (place the idealized z=0 Atlas map at a mock
                                 redshift, apply cosmological dimming)
    apply_lsst_psf.py          (convolve with the LSST PSF, resample to the
                                 detector, add sky background + noise)

-- and records one manifest row per product, with the galaxy's physical
properties pulled from the Atlas catalog for downstream captioning.

Example:
    uv run scripts/simulations/observation/build_atlas_dataset.py \\
        configs/atlas/tng50_dataset_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import atlas_demo  # noqa: E402  (path must be extended first)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Dataset config must be a YAML mapping.")
    for section in ("run", "atlas", "observation", "manifest"):
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    return config


def redshift_code(redshift: float) -> str:
    return f"z{round(redshift * 100):03d}"


def object_output_dir(
    output_root: Path,
    *,
    subhalo_id: int,
    observer: str,
    redshift: float,
    profile_id: str,
) -> Path:
    name = (
        f"atlas_tng{subhalo_id:06d}_{observer.lower()}_"
        f"{redshift_code(redshift)}_{profile_id}"
    )
    return output_root / name


def ensure_atlas_data(subhalo_id: int, *, auto_download: bool) -> Path:
    """Return the extracted Atlas files directory, downloading if allowed."""
    files_dir = atlas_demo.object_directory(subhalo_id) / "files"

    if files_dir.exists():
        return files_dir

    if not auto_download:
        raise FileNotFoundError(
            f"Atlas archive for subhalo {subhalo_id} is not downloaded "
            f"({files_dir} missing). Run "
            f"`python scripts/atlas_demo.py download --subhalo-id "
            f"{subhalo_id}` first, or set atlas.auto_download: true."
        )

    print(f"Downloading Atlas archive for subhalo {subhalo_id}...")
    download_args = argparse.Namespace(subhalo_id=subhalo_id, force=False)
    atlas_demo.command_download(download_args)
    return files_dir


def run_subprocess(command: list[str]) -> None:
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def clean_for_rgb(image: np.ndarray) -> np.ndarray:
    result = np.asarray(image, dtype=np.float32).copy()
    result[~np.isfinite(result)] = 0
    result[result < 0] = 0
    return result


def write_rgb_preview(
    observed_fits: Path,
    png_path: Path,
    *,
    bands: list[str],
    stretch: float,
    q: float,
    percentile: float,
) -> None:
    """Render a blue/green/red preview from OBS_LSST_* extensions."""
    blue_band, green_band, red_band = bands

    with fits.open(observed_fits, memmap=True) as hdul:
        red = clean_for_rgb(hdul[f"OBS_LSST_{red_band.upper()}"].data)
        green = clean_for_rgb(hdul[f"OBS_LSST_{green_band.upper()}"].data)
        blue = clean_for_rgb(hdul[f"OBS_LSST_{blue_band.upper()}"].data)

    positive = np.concatenate([red[red > 0], green[green > 0], blue[blue > 0]])
    if positive.size == 0:
        raise ValueError(f"All selected bands are non-positive in {observed_fits}.")

    scale = float(np.percentile(positive, percentile))

    rgb = make_lupton_rgb(red / scale, green / scale, blue / scale, stretch=stretch, Q=q)
    Image.fromarray(np.flipud(rgb)).save(png_path)


def build_object(
    *,
    subhalo_id: int,
    observer: str,
    redshift: float,
    config: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    atlas_files = ensure_atlas_data(
        subhalo_id,
        auto_download=bool(config["atlas"]["auto_download"]),
    )

    profile_path = Path(config["observation"]["profile"])
    profile_id = yaml.safe_load(profile_path.read_text())["profile"]["id"]
    crop_kpc = float(config["observation"]["crop_kpc"])

    output_root = REPO_ROOT / config["run"]["output_dir"]
    object_dir = object_output_dir(
        output_root,
        subhalo_id=subhalo_id,
        observer=observer,
        redshift=redshift,
        profile_id=profile_id,
    )

    ideal_path = object_dir / f"ideal_atlas_mock_{redshift_code(redshift)}.fits"
    observed_path = object_dir / "observed_lsst_dust.fits"
    preview_path = object_dir / "lsst_gri_single_visit.png"

    if observed_path.exists() and not force:
        print(f"Skipping (already built): {object_dir.relative_to(REPO_ROOT)}")
    else:
        object_dir.mkdir(parents=True, exist_ok=True)

        run_subprocess(
            [
                sys.executable,
                "scripts/simulations/observation/prepare_atlas_mock_sky.py",
                str(atlas_files),
                str(ideal_path),
                "--observer",
                observer,
                "--redshift",
                str(redshift),
                "--crop-kpc",
                str(crop_kpc),
            ]
        )

        run_subprocess(
            [
                sys.executable,
                "scripts/simulations/observation/apply_lsst_psf.py",
                str(ideal_path),
                str(observed_path),
                "--profile",
                str(profile_path),
                "--dust-state",
                "DUST",
            ]
        )

        if config["diagnostics"]["make_rgb_preview"]:
            write_rgb_preview(
                observed_path,
                preview_path,
                bands=list(config["diagnostics"]["rgb_bands"]),
                stretch=float(config["diagnostics"]["rgb_stretch"]),
                q=float(config["diagnostics"]["rgb_q"]),
                percentile=float(config["diagnostics"]["rgb_percentile"]),
            )

        print(f"Built: {object_dir.relative_to(REPO_ROOT)}")

    catalog_path = atlas_demo.object_directory(subhalo_id) / "atlas_catalog.txt"
    properties = atlas_demo.load_catalog_row(catalog_path, subhalo_id)

    return {
        "subhalo_id": subhalo_id,
        "observer": observer,
        "mock_redshift": redshift,
        "profile_id": profile_id,
        "output_dir": str(object_dir.relative_to(REPO_ROOT)),
        "ideal_fits": str(ideal_path.relative_to(REPO_ROOT)),
        "observed_fits": str(observed_path.relative_to(REPO_ROOT)),
        "rgb_preview": (
            str(preview_path.relative_to(REPO_ROOT))
            if config["diagnostics"]["make_rgb_preview"]
            else None
        ),
        "physical_properties": properties,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Dataset YAML config.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild objects even if their output already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    subhalo_ids = list(config["atlas"]["subhalo_ids"])
    observers = list(config["atlas"]["observers"])
    redshifts = list(config["observation"]["redshifts"])

    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    combinations = [
        (subhalo_id, observer, redshift)
        for subhalo_id in subhalo_ids
        for observer in observers
        for redshift in redshifts
    ]

    for subhalo_id, observer, redshift in combinations:
        try:
            manifest.append(
                build_object(
                    subhalo_id=subhalo_id,
                    observer=observer,
                    redshift=redshift,
                    config=config,
                    force=args.force,
                )
            )
        except Exception as error:
            print(
                f"FAILED subhalo={subhalo_id} observer={observer} "
                f"redshift={redshift}: {error}"
            )
            failures.append(
                {
                    "subhalo_id": subhalo_id,
                    "observer": observer,
                    "mock_redshift": redshift,
                    "error": str(error),
                }
            )

    manifest_path = REPO_ROOT / config["manifest"]["path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "config": str(args.config),
                "objects": manifest,
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nBuilt {len(manifest)}/{len(combinations)} objects. "
        f"{len(failures)} failed."
    )
    print(f"Manifest: {manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
