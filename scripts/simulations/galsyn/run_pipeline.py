"""Run a configuration-driven TNG-to-GalSyn synthesis and diagnostics pipeline."""

from __future__ import annotations

import argparse
import importlib.resources
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np
import requests
import yaml
from astropy.io import fits
from astropy.visualization import make_lupton_rgb, simple_norm
from dotenv import load_dotenv
from galsyn import GalaxySynthesizer
from galsyn.dust import (
    bump_amp_from_dust_index,
    scale_dust_redshift_Vogelsberger20,
)
from galsyn.simutils_tng import (
    get_snap_z,
    make_sim_file_from_tng_data,
)
from PIL import Image


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    for section in ("run", "prepare", "filters", "synthesis", "diagnostics"):
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    return config


def required_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def download_tng_cutout_streaming(
    *,
    simulation: str,
    snapshot: int,
    subhalo_id: int,
    scope: str,
    api_key: str,
    output_path: Path,
) -> None:
    """Download a TNG cutout to a resumable partial file with progress output."""
    cutout_key = {"subhalo": "subhalo", "parent_halo": "parent_halo"}.get(scope)
    if cutout_key is None:
        raise ValueError("prepare.cutout_scope must be 'parent_halo' or 'subhalo'.")

    headers = {"api-key": api_key}
    metadata_url = (
        f"https://www.tng-project.org/api/{simulation}/snapshots/{snapshot}/"
        f"subhalos/{subhalo_id}"
    )
    with requests.Session() as session:
        metadata = session.get(metadata_url, headers=headers, timeout=(30, 120))
        metadata.raise_for_status()
        cutout_url = metadata.json()["cutouts"][cutout_key]
        partial_path = output_path.with_suffix(output_path.suffix + ".part")
        existing_bytes = partial_path.stat().st_size if partial_path.exists() else 0
        request_headers = headers.copy()
        if existing_bytes:
            request_headers["Range"] = f"bytes={existing_bytes}-"

        print(f"Downloading {scope} TNG cutout to {partial_path}...")
        try:
            response = session.get(
                cutout_url,
                headers=request_headers,
                stream=True,
                timeout=(30, 120),
            )
            response.raise_for_status()
            append = existing_bytes > 0 and response.status_code == 206
            if existing_bytes and not append:
                print("Server did not accept resume; restarting partial download.")
                existing_bytes = 0
            content_length = int(response.headers.get("content-length", 0))
            total_bytes = existing_bytes + content_length if content_length else None
            mode = "ab" if append else "wb"
            downloaded = existing_bytes
            last_report = time.monotonic()
            with partial_path.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 5:
                        if total_bytes:
                            print(
                                f"  {downloaded / 1024**2:.1f} / "
                                f"{total_bytes / 1024**2:.1f} MiB"
                            )
                        else:
                            print(f"  {downloaded / 1024**2:.1f} MiB downloaded")
                        last_report = now
        except requests.RequestException as error:
            raise RuntimeError(
                f"TNG cutout download failed; partial data remains at {partial_path}."
            ) from error

    if total_bytes is not None and downloaded != total_bytes:
        raise RuntimeError(
            f"Incomplete TNG cutout: received {downloaded} of {total_bytes} bytes. "
            f"Partial data remains at {partial_path}."
        )
    partial_path.replace(output_path)
    print(f"Downloaded {output_path} ({downloaded / 1024**2:.1f} MiB).")


def prepare_simulation(config: dict[str, Any]) -> tuple[Path, float | None]:
    prepare = config["prepare"]
    standardized = Path(prepare["standardized_file"])
    if standardized.is_file():
        print(f"Using existing standardized input: {standardized}")
        return standardized, None

    load_dotenv()
    api_key = os.environ.get("TNG_API_KEY")
    if not api_key:
        raise RuntimeError(
            "The standardized input is missing and TNG_API_KEY is absent from .env."
        )
    raw_cutout = Path(prepare["raw_cutout"])
    raw_cutout.parent.mkdir(parents=True, exist_ok=True)
    standardized.parent.mkdir(parents=True, exist_ok=True)
    scope = prepare["cutout_scope"]
    snapshot, subhalo_id = int(prepare["snapshot"]), int(prepare["subhalo_id"])
    if not raw_cutout.is_file():
        download_tng_cutout_streaming(
            simulation=prepare["simulation"],
            snapshot=snapshot,
            subhalo_id=subhalo_id,
            scope=scope,
            api_key=api_key,
            output_path=raw_cutout,
        )
    redshift = get_snap_z(snapshot, sim=prepare["simulation"], api_key=api_key)
    print(f"Standardizing TNG input at snapshot redshift z={redshift:.6f}...")
    make_sim_file_from_tng_data(
        str(raw_cutout),
        redshift,
        cosmo_h=float(prepare["cosmo_h"]),
        XH=float(prepare["hydrogen_mass_fraction"]),
        output_hdf5=str(standardized),
    )
    return standardized, float(redshift)


def filter_paths(config: dict[str, Any]) -> dict[str, str]:
    directory = Path(config["filters"]["directory"])
    names = config["filters"]["names"]
    if not isinstance(names, list) or not names:
        raise ValueError("filters.names must be a non-empty list.")
    paths = {name: directory / f"{name}.txt" for name in names}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing and config["filters"].get("export_missing", False):
        export_filters(names, directory, Path(config["filters"]["pixedfit_root"]))
        missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing filter files:\n  " + "\n  ".join(missing))
    return {name: str(path) for name, path in paths.items()}


def export_filters(names: list[str], directory: Path, pixedfit_root: Path) -> None:
    """Export configured piXedfit filters in GALSYN's two-column text format."""
    wavelength_path = pixedfit_root / "data/filters/filters_w.hdf5"
    transmission_path = pixedfit_root / "data/filters/filters_t.hdf5"
    required_path(wavelength_path, "piXedfit wavelength database")
    required_path(transmission_path, "piXedfit transmission database")
    directory.mkdir(parents=True, exist_ok=True)
    with (
        h5py.File(wavelength_path, "r") as wavelengths,
        h5py.File(transmission_path, "r") as transmissions,
    ):
        unavailable = [
            name
            for name in names
            if name not in wavelengths or name not in transmissions
        ]
        if unavailable:
            raise KeyError(f"Filters unavailable in piXedfit: {', '.join(unavailable)}")
        for name in names:
            output = directory / f"{name}.txt"
            if not output.is_file():
                np.savetxt(
                    output,
                    np.column_stack([wavelengths[name][:], transmissions[name][:]]),
                    fmt="%.10e",
                    header="wavelength_angstrom transmission",
                )
                print(f"Exported filter: {output}")


def synthesize(config: dict[str, Any], sim_file: Path, output_fits: Path) -> None:
    synthesis, spectra = config["synthesis"], config["synthesis"]["spectra"]
    gs = GalaxySynthesizer(
        str(sim_file),
        z=float(synthesis["redshift"]),
        filters=config["filters"]["names"],
        filter_transmission_path=filter_paths(config),
    )
    gs.ssp_filepath = str(required_path(synthesis["ssp_grid"], "SSP grid"))
    gs.ssp_interpolation_method = synthesis["interpolation"]
    gs.dim_kpc = float(synthesis["dim_kpc"])
    gs.smoothing_length = float(synthesis["smoothing_length_kpc"])
    if "pix_arcsec" in synthesis:
        gs.pix_arcsec, gs.pix_kpc = float(synthesis["pix_arcsec"]), None
    else:
        gs.pix_arcsec, gs.pix_kpc = None, float(synthesis["pix_kpc"])
    gs.flux_unit = synthesis["flux_unit"]
    gs.polar_angle_deg, gs.azimuth_angle_deg = (
        float(synthesis["polar_angle_deg"]),
        float(synthesis["azimuth_angle_deg"]),
    )
    gs.dust_method, gs.dust_law, gs.ncpu = (
        synthesis["dust_method"],
        int(synthesis["dust_law"]),
        int(synthesis["ncpu"]),
    )
    if synthesis.get("tutorial_adaptive_dust", False):
        # GalSyn 0.1.5 asks for a lower-case filename here, while its wheel
        # ships ``Salim18_AV_dust_index.txt``. Load the packaged relation with
        # its actual filename so the tutorial setup also works on Linux.
        relation_path = importlib.resources.files("galsyn.data").joinpath(
            "Salim18_AV_dust_index.txt"
        )
        relation_data = np.loadtxt(relation_path)
        dust_index = {
            "AV": relation_data[:, 0],
            "dust_index": relation_data[:, 1],
        }
        gs.dust_index = dust_index
        gs.bump_amp = {
            "AV": dust_index["AV"],
            "bump_amp": bump_amp_from_dust_index(dust_index["dust_index"]),
        }
        relation = scale_dust_redshift_Vogelsberger20()
        gs.scale_dust_redshift = {
            "z": relation["z"],
            "tau_dust": relation["tau_dust"] * 1.6,
        }
        gs.dust_eta = 2.0
        gs.dust_index_bc = -0.7
        gs.bump_dwave = 0.035
    gs.output_pixel_spectra = True
    gs.rest_wave_min, gs.rest_wave_max = (
        float(spectra["rest_wave_min_angstrom"]),
        float(spectra["rest_wave_max_angstrom"]),
    )
    gs.rest_delta_wave, gs.name_out_img = (
        float(spectra["delta_wave_angstrom"]),
        str(output_fits),
    )
    gs.run_synthesis()


def clean(data: np.ndarray) -> np.ndarray:
    result = np.asarray(data, dtype=np.float64).copy()
    result[~np.isfinite(result)] = 0
    result[result < 0] = 0
    return result


def make_diagnostics(
    fits_path: Path, output_dir: Path, config: dict[str, Any]
) -> dict[str, str]:
    diagnostics, filters, rgb_filters = (
        config["diagnostics"],
        config["filters"]["names"],
        config["diagnostics"]["rgb_filters"],
    )
    if len(rgb_filters) != 3 or any(name not in filters for name in rgb_filters):
        raise ValueError(
            "diagnostics.rgb_filters must contain three configured filters."
        )
    rgb_path, stamps_path, spectrum_path = (
        output_dir / "rgb.png",
        output_dir / "filter_stamps.png",
        output_dir / "integrated_spectrum.png",
    )
    with fits.open(fits_path, memmap=True) as cube:
        red, green, blue = (
            clean(cube[f"DUST_{name.upper()}"].data) for name in reversed(rgb_filters)
        )
        positive = np.concatenate([item[item > 0] for item in (red, green, blue)])
        if positive.size == 0:
            raise ValueError("Selected RGB filters contain no positive flux.")
        scale = np.percentile(positive, float(diagnostics["map_percentile"]))
        rgb = make_lupton_rgb(
            red / scale,
            green / scale,
            blue / scale,
            stretch=float(diagnostics["rgb_stretch"]),
            Q=float(diagnostics["rgb_q"]),
        )
        Image.fromarray(np.flipud(np.nan_to_num(rgb, nan=0))).save(rgb_path)

        ncols, nrows = (
            min(4, len(filters) + 1),
            int(np.ceil((len(filters) + 1) / min(4, len(filters) + 1))),
        )
        figure, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
        axes = np.asarray(axes).ravel()
        axes[0].imshow(rgb, origin="lower")
        axes[0].set_title("RGB: " + "/".join(rgb_filters))
        axes[0].axis("off")
        for axis, name in zip(axes[1:], filters):
            image = clean(cube[f"DUST_{name.upper()}"].data)
            axis.imshow(
                image,
                origin="lower",
                cmap="gray",
                norm=simple_norm(
                    image, "sqrt", percent=float(diagnostics["map_percentile"])
                ),
            )
            axis.set_title(name)
            axis.axis("off")
        for axis in axes[len(filters) + 1 :]:
            axis.axis("off")
        figure.suptitle("Dust-attenuated GalSyn photometry", y=0.99)
        figure.tight_layout()
        figure.savefig(stamps_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

        wavelength = np.asarray(cube["WAVELENGTH_GRID"].data["WAVELENGTH"], dtype=float)
        dust, nodust = (
            np.asarray(cube["OBS_SPEC_DUST"].data, dtype=float),
            np.asarray(cube["OBS_SPEC_NODUST"].data, dtype=float),
        )
        pix_kpc, ny, nx = float(cube[0].header["PIX_KPC"]), *dust.shape[1:]
        yy, xx = np.ogrid[:ny, :nx]
        radius_pixels = float(diagnostics["aperture_radius_kpc"]) / pix_kpc
        aperture = (xx - (nx - 1) / 2) ** 2 + (
            yy - (ny - 1) / 2
        ) ** 2 <= radius_pixels**2
        figure, axis = plt.subplots(figsize=(11, 4.8))
        axis.plot(wavelength, np.nansum(dust[:, aperture], axis=1), lw=1, label="Dust")
        axis.plot(
            wavelength,
            np.nansum(nodust[:, aperture], axis=1),
            lw=0.8,
            alpha=0.65,
            label="No dust",
        )
        axis.set(
            title=f"Integrated spectrum within {diagnostics['aperture_radius_kpc']} kpc",
            xlabel="Observed wavelength [Å]",
            ylabel="Integrated flux",
        )
        axis.legend()
        figure.tight_layout()
        figure.savefig(spectrum_path, dpi=180)
        plt.close(figure)
    return {
        "rgb": str(rgb_path),
        "filter_stamps": str(stamps_path),
        "integrated_spectrum": str(spectrum_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML run configuration.")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing FITS product."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(config["run"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fits_path = output_dir / f"{config['run']['name']}_specphoto.fits"
    started = time.perf_counter()
    sim_file, source_redshift = prepare_simulation(config)
    if config["synthesis"]["redshift"] == "snapshot":
        if source_redshift is None:
            load_dotenv()
            api_key = os.environ.get("TNG_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "synthesis.redshift is 'snapshot', but TNG_API_KEY is absent "
                    "from .env and the standardized input has no stored redshift."
                )
            source_redshift = float(
                get_snap_z(
                    int(config["prepare"]["snapshot"]),
                    sim=config["prepare"]["simulation"],
                    api_key=api_key,
                )
            )
        config["synthesis"]["redshift"] = source_redshift
    if fits_path.exists() and not args.force:
        print(f"Using existing synthesis: {fits_path}")
    else:
        print(f"Synthesizing {fits_path}...")
        synthesize(config, sim_file, fits_path)
    if not fits_path.is_file():
        raise RuntimeError(f"GalSyn did not create expected FITS product: {fits_path}")
    products = make_diagnostics(fits_path, output_dir, config)
    copied_config = output_dir / "run_config.yaml"
    shutil.copyfile(args.config, copied_config)
    manifest = {
        "run_name": config["run"]["name"],
        "config": str(copied_config),
        "simulation_file": str(sim_file),
        "snapshot_redshift": source_redshift,
        "synthesis_redshift": config["synthesis"]["redshift"],
        "filters": config["filters"]["names"],
        "fits": str(fits_path),
        "products": products,
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Completed pipeline. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
