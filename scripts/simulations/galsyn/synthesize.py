from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from galsyn import GalaxySynthesizer


DEFAULT_SIM_FILE = Path(
    "data/standardized/"
    "TNG50-1_snap45_subhalo493093_parent_halo.hdf5"
)

DEFAULT_SSP_GRID = Path(
    "data/ssp/ssp_fsps_a50_z50_u10.hdf5"
)

DEFAULT_FILTER_DIR = Path("data/filters")

DEFAULT_FILTERS = [
    "jwst_nircam_f090w",
    "jwst_nircam_f150w",
    "jwst_nircam_f200w",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate broadband images, and optionally a spectral cube, "
            "from a GalSyn-standardized TNG file."
        )
    )

    parser.add_argument(
        "--sim-file",
        type=Path,
        default=DEFAULT_SIM_FILE,
        help="GalSyn-standardized HDF5 simulation file.",
    )

    parser.add_argument(
        "--ssp-grid",
        type=Path,
        default=DEFAULT_SSP_GRID,
        help="Precomputed GalSyn SSP-grid HDF5 file.",
    )

    parser.add_argument(
        "--filter-dir",
        type=Path,
        default=DEFAULT_FILTER_DIR,
        help="Directory containing two-column filter transmission files.",
    )

    parser.add_argument(
        "--filters",
        nargs="+",
        default=DEFAULT_FILTERS,
        help="Filter names. Each must have <filter-dir>/<name>.txt.",
    )

    parser.add_argument(
        "--redshift",
        type=float,
        default=1.206258,
        help="Observation redshift. Snapshot 45 has z approximately 1.206258.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output FITS file. A default is selected based on --spectra.",
    )

    parser.add_argument(
        "--dim-kpc",
        type=float,
        default=30.0,
        help="Physical side length of the output field in kpc.",
    )

    parser.add_argument(
        "--smoothing-length",
        type=float,
        default=0.20,
        help="Initial GalSyn gridding scale in kpc.",
    )

    parser.add_argument(
        "--pix-kpc",
        type=float,
        default=0.20,
        help="Output physical pixel scale in kpc/pixel.",
    )

    parser.add_argument(
        "--polar-angle",
        type=float,
        default=45.0,
        help="Viewing polar angle in degrees.",
    )

    parser.add_argument(
        "--azimuth-angle",
        type=float,
        default=45.0,
        help="Viewing azimuth angle in degrees.",
    )

    parser.add_argument(
        "--ncpu",
        type=int,
        default=max(1, min(24, (os.cpu_count() or 2) - 2)),
        help="Number of CPU cores.",
    )

    parser.add_argument(
        "--interpolation",
        choices=["nearest", "linear"],
        default="linear",
        help="SSP-grid interpolation method.",
    )

    parser.add_argument(
        "--spectra",
        action="store_true",
        help="Also generate wavelength-resolved pixel spectra.",
    )

    parser.add_argument(
        "--wave-min",
        type=float,
        default=1000.0,
        help="Minimum rest-frame wavelength in Angstrom.",
    )

    parser.add_argument(
        "--wave-max",
        type=float,
        default=30000.0,
        help="Maximum rest-frame wavelength in Angstrom.",
    )

    parser.add_argument(
        "--delta-wave",
        type=float,
        default=20.0,
        help=(
            "Rest-frame spectral spacing in Angstrom. "
            "Use 20 for a smoke test and 5 for higher resolution."
        ),
    )

    return parser


def build_filter_paths(
    filters: list[str],
    filter_dir: Path,
) -> dict[str, str]:
    paths = {
        name: filter_dir / f"{name}.txt"
        for name in filters
    }

    missing = [
        path
        for path in paths.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing filter transmission files:\n"
            + "\n".join(f"  {path}" for path in missing)
        )

    return {
        name: str(path)
        for name, path in paths.items()
    }


def validate_args(args: argparse.Namespace) -> None:
    if not args.sim_file.is_file():
        raise FileNotFoundError(
            f"Standardized simulation file not found: {args.sim_file}"
        )

    if not args.ssp_grid.is_file():
        raise FileNotFoundError(
            f"SSP grid not found: {args.ssp_grid}"
        )

    if args.redshift < 0:
        raise ValueError("--redshift must be non-negative.")

    if args.dim_kpc <= 0:
        raise ValueError("--dim-kpc must be positive.")

    if args.smoothing_length <= 0:
        raise ValueError("--smoothing-length must be positive.")

    if args.pix_kpc <= 0:
        raise ValueError("--pix-kpc must be positive.")

    if args.ncpu <= 0:
        raise ValueError("--ncpu must be positive.")

    if args.wave_max <= args.wave_min:
        raise ValueError("--wave-max must exceed --wave-min.")

    if args.delta_wave <= 0:
        raise ValueError("--delta-wave must be positive.")


def default_output_path(spectra: bool) -> Path:
    suffix = "specphoto" if spectra else "photo"

    return Path(
        "outputs/galsyn/"
        f"galsyn_tng50_phalo_45_493093_{suffix}.fits"
    )


def print_configuration(
    args: argparse.Namespace,
    output_path: Path,
    filter_paths: dict[str, str],
) -> None:
    image_pixels = round(args.dim_kpc / args.pix_kpc)

    print("GalSyn synthesis configuration")
    print("------------------------------")
    print(f"Simulation file:   {args.sim_file}")
    print(f"SSP grid:          {args.ssp_grid}")
    print(f"Redshift:          {args.redshift:.6f}")
    print(f"Filters:           {', '.join(args.filters)}")
    print(f"Field size:        {args.dim_kpc:.3f} kpc")
    print(f"Pixel size:        {args.pix_kpc:.3f} kpc/pixel")
    print(f"Approx. shape:     {image_pixels} x {image_pixels}")
    print(f"Smoothing length:  {args.smoothing_length:.3f} kpc")
    print(
        "Viewing direction: "
        f"polar={args.polar_angle:.1f}°, "
        f"azimuth={args.azimuth_angle:.1f}°"
    )
    print("Dust method:       line-of-sight")
    print("Dust law:          modified Calzetti, option 0")
    print(f"SSP interpolation: {args.interpolation}")
    print(f"CPU cores:         {args.ncpu}")
    print(f"Pixel spectra:     {args.spectra}")

    if args.spectra:
        n_wavelength = (
            int(
                (args.wave_max - args.wave_min)
                / args.delta_wave
            )
            + 1
        )

        print(
            "Rest wavelength:   "
            f"{args.wave_min:.1f}–{args.wave_max:.1f} Å"
        )
        print(f"Spectral spacing:  {args.delta_wave:.1f} Å")
        print(f"Approx. planes:    {n_wavelength:,}")

    print(f"Output:            {output_path}")

    print("\nFilter files")
    print("------------")
    for name, path in filter_paths.items():
        print(f"{name:28s} {path}")


def write_run_metadata(
    *,
    args: argparse.Namespace,
    output_path: Path,
    elapsed_seconds: float,
) -> None:
    metadata_path = output_path.with_suffix(".json")

    metadata = {
        "simulation": "TNG50-1",
        "snapshot": 45,
        "subhalo_id": 493093,
        "cutout_scope": "parent_halo",
        "redshift": args.redshift,
        "simulation_file": str(args.sim_file),
        "ssp_grid": str(args.ssp_grid),
        "filters": args.filters,
        "dim_kpc": args.dim_kpc,
        "smoothing_length_kpc": args.smoothing_length,
        "pix_kpc": args.pix_kpc,
        "polar_angle_deg": args.polar_angle,
        "azimuth_angle_deg": args.azimuth_angle,
        "dust_method": "los",
        "dust_law": 0,
        "ssp_interpolation": args.interpolation,
        "ncpu": args.ncpu,
        "output_pixel_spectra": args.spectra,
        "rest_wave_min_angstrom": (
            args.wave_min if args.spectra else None
        ),
        "rest_wave_max_angstrom": (
            args.wave_max if args.spectra else None
        ),
        "rest_delta_wave_angstrom": (
            args.delta_wave if args.spectra else None
        ),
        "output_fits": str(output_path),
        "elapsed_seconds": elapsed_seconds,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Metadata:          {metadata_path}")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    filter_paths = build_filter_paths(
        args.filters,
        args.filter_dir,
    )

    output_path = (
        args.output
        if args.output is not None
        else default_output_path(args.spectra)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_configuration(
        args,
        output_path,
        filter_paths,
    )

    gs = GalaxySynthesizer(
        str(args.sim_file),
        z=args.redshift,
        filters=args.filters,
        filter_transmission_path=filter_paths,
    )

    # Precomputed stellar-population spectra.
    gs.ssp_filepath = str(args.ssp_grid)
    gs.ssp_interpolation_method = args.interpolation

    # Physical field and sampling.
    gs.dim_kpc = args.dim_kpc
    gs.smoothing_length = args.smoothing_length

    # Keep the synthesis grid in physical units.
    # Survey angular sampling can be added later.
    gs.pix_arcsec = None
    gs.pix_kpc = args.pix_kpc

    gs.flux_unit = "MJy/sr"

    # Camera orientation.
    gs.polar_angle_deg = args.polar_angle
    gs.azimuth_angle_deg = args.azimuth_angle

    # Basic GalSyn line-of-sight attenuation.
    gs.dust_method = "los"
    gs.dust_law = 0

    gs.ncpu = args.ncpu

    # Images are always produced.
    # This controls whether the wavelength cube is also retained.
    gs.output_pixel_spectra = args.spectra

    if args.spectra:
        gs.rest_wave_min = args.wave_min
        gs.rest_wave_max = args.wave_max
        gs.rest_delta_wave = args.delta_wave

    gs.name_out_img = str(output_path)

    print("\nStarting synthesis...")
    started = time.perf_counter()

    gs.run_synthesis()

    elapsed = time.perf_counter() - started

    if not output_path.is_file():
        raise RuntimeError(
            "GalSyn returned without producing the expected output: "
            f"{output_path}"
        )

    print("\nSynthesis completed")
    print("-------------------")
    print(f"Elapsed:           {elapsed / 60:.2f} minutes")
    print(
        f"Output size:       "
        f"{output_path.stat().st_size / 1024**2:.2f} MiB"
    )
    print(f"FITS file:         {output_path}")

    write_run_metadata(
        args=args,
        output_path=output_path,
        elapsed_seconds=elapsed,
    )


if __name__ == "__main__":
    main()