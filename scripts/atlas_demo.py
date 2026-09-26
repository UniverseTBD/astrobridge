"""

    Based on:
        - TNG50 Cosmological hydrodynamical simulation (https://arxiv.org/pdf/1902.05553, https://arxiv.org/pdf/1902.05554):  magnetohydrodynamical simulation for galaxy formation 
        - TNG50-SKIRT Atlas (https://arxiv.org/pdf/2401.04224v1): sample of 1154 galaxies extracted from the TNG50 cosmological simulation at z = 0.
        
        - Alternative if we want to scale things up, Galsyn (https://arxiv.org/pdf/2603.23986)
    download:
        Calls TNG API endpoint for TNG50-1 | SKIRT Atlas and downloads the tar.
    extract: 
        Expands the archive into 240 FITS files.
    inspect:
        Lists filenames and verifies image shapes and units.
    build:
        Load u,g,z arrays
        Crops central region
        Maps z -> R, g -> G, u -> B
        Applies Lupton stretch
        Saves PNG
        Stores the arrays by band
        Attaches catalog parameters
        Writes LLaVA JSON record
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm


ATLAS_CATALOG_URL = (
    "https://www.tng-project.org/files/data/"
    "baes24_tng50_skirt_atlas.txt"
)

ATLAS_API_TEMPLATE = (
    "https://www.tng-project.org/api/TNG50-1/"
    "snapshots/99/subhalos/{subhalo_id}/"
    "skirt/skirt_atlas.tar.gz"
)

CATALOG_COLUMNS = [
    "subhalo_id",
    "stellar_mass_msun",
    "gas_mass_msun",
    "dark_matter_mass_msun",
    "sfr_msun_per_year",
    "stellar_half_mass_radius_kpc",
]

# Each Atlas archive contains 5 observers x 48 products per observer.
PRODUCTS_PER_OBSERVER = 48
OBSERVER_COUNT = 5
EXPECTED_PRODUCT_COUNT = PRODUCTS_PER_OBSERVER * OBSERVER_COUNT

REPO_ROOT = Path(__file__).resolve().parent.parent

# The TNG data portal is prone to slow transfers and gateway timeouts on
# large archives. Retry transient failures with backoff, and resume the
# partial ".part" file via HTTP Range instead of restarting from zero.
MAX_DOWNLOAD_ATTEMPTS = 8
RETRY_BASE_DELAY_SECONDS = 5.0
RETRY_MAX_DELAY_SECONDS = 120.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable_error(error: requests.exceptions.RequestException) -> bool:
    response = getattr(error, "response", None)
    if response is None:
        # Connection drop / read timeout with no HTTP response: transient.
        return True
    return response.status_code in RETRYABLE_STATUS_CODES


def _wait_before_retry(attempt: int) -> None:
    delay = min(
        RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
        RETRY_MAX_DELAY_SECONDS,
    )
    print(f"Retrying in {delay:.0f}s (attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS})...")
    time.sleep(delay)


def download_file(
    url: str,
    destination: Path,
    *,
    headers: dict[str, str] | None = None,
) -> None:
    """Download a file atomically, resuming and retrying on transient errors."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        resume_from = temporary.stat().st_size if temporary.exists() else 0
        request_headers = dict(headers or {})
        if resume_from:
            request_headers["Range"] = f"bytes={resume_from}-"

        try:
            with requests.get(
                url,
                headers=request_headers,
                stream=True,
                timeout=(30, 300),
            ) as response:
                if response.status_code == 416:
                    # Our partial file is already complete or corrupt from
                    # the server's point of view; drop it and start over.
                    temporary.unlink(missing_ok=True)
                    raise requests.exceptions.HTTPError(
                        "416 Range Not Satisfiable", response=response
                    )

                response.raise_for_status()

                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    raise RuntimeError(
                        f"Unexpected JSON response from {url}: {response.text}"
                    )

                resuming = bool(resume_from) and response.status_code == 206
                if resume_from and not resuming:
                    # Server ignored the Range request; restart from zero.
                    resume_from = 0

                remaining = int(response.headers.get("content-length", 0))
                total = (resume_from + remaining) if remaining else None

                with temporary.open("ab" if resuming else "wb") as output, tqdm(
                    total=total,
                    initial=resume_from,
                    unit="B",
                    unit_scale=True,
                    desc=destination.name,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue

                        output.write(chunk)
                        progress.update(len(chunk))

            temporary.replace(destination)
            return

        except requests.exceptions.RequestException as error:
            if attempt == MAX_DOWNLOAD_ATTEMPTS or not _is_retryable_error(error):
                raise

            print(f"Download error ({error}); will retry.")
            _wait_before_retry(attempt)


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    """Extract a tar archive while preventing path traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()

    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()

            if not target.is_relative_to(destination_resolved):
                raise RuntimeError(
                    f"Unsafe archive member encountered: {member.name}"
                )

        archive.extractall(destination)


def object_directory(subhalo_id: int) -> Path:
    return REPO_ROOT / "data" / "atlas" / f"TNG{subhalo_id:06d}"


def command_download(args: argparse.Namespace) -> None:
    load_dotenv()

    api_key = os.environ.get("TNG_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TNG_API_KEY is absent. Put it in the repository .env file."
        )

    object_dir = object_directory(args.subhalo_id)
    archive_path = (
        object_dir
        / f"TNG{args.subhalo_id:06d}_skirt_atlas.tar.gz"
    )
    catalog_path = object_dir / "atlas_catalog.txt"
    extraction_dir = object_dir / "files"

    if args.force:
        for path in (
            archive_path,
            archive_path.with_suffix(archive_path.suffix + ".part"),
            catalog_path,
            catalog_path.with_suffix(catalog_path.suffix + ".part"),
        ):
            if path.exists():
                path.unlink()
        if extraction_dir.exists():
            shutil.rmtree(extraction_dir)

    if not catalog_path.exists():
        print("Downloading Atlas sample catalog...")
        download_file(ATLAS_CATALOG_URL, catalog_path)
    else:
        print(f"Catalog already exists: {catalog_path}")

    if not archive_path.exists():
        print("Downloading Atlas galaxy archive...")

        url = ATLAS_API_TEMPLATE.format(
            subhalo_id=args.subhalo_id
        )

        download_file(
            url,
            archive_path,
            headers={"api-key": api_key},
        )
    else:
        print(f"Archive already exists: {archive_path}")

    if not extraction_dir.exists():
        print("Extracting archive...")
        safe_extract_tar(archive_path, extraction_dir)
    else:
        print(f"Extraction directory already exists: {extraction_dir}")

    fits_files = sorted(extraction_dir.rglob("*.fits"))

    print(f"\nFound {len(fits_files)} FITS files.")
    print(f"Object directory: {object_dir.resolve()}")

    if len(fits_files) != EXPECTED_PRODUCT_COUNT:
        print(
            f"Warning: expected {EXPECTED_PRODUCT_COUNT} products per "
            f"galaxy ({OBSERVER_COUNT} observers x "
            f"{PRODUCTS_PER_OBSERVER} products). "
            "Inspect the extracted archive before continuing."
        )


def first_2d_hdu(path: Path) -> tuple[np.ndarray, fits.Header, str]:
    """Load the first two-dimensional image from a FITS file."""
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue

            data = np.asarray(hdu.data)

            if data.ndim == 2:
                return (
                    data.astype(np.float32),
                    hdu.header.copy(),
                    hdu.name,
                )

    raise ValueError(f"No two-dimensional image found in {path}")


def command_inspect(args: argparse.Namespace) -> None:
    extraction_dir = object_directory(args.subhalo_id) / "files"

    if not extraction_dir.exists():
        raise FileNotFoundError(
            f"{extraction_dir} does not exist. Run download first."
        )

    paths = sorted(extraction_dir.rglob("*.fits"))

    search_terms = [term.lower() for term in args.contains]

    if search_terms:
        paths = [
            path
            for path in paths
            if all(term in str(path).lower() for term in search_terms)
        ]

    if not paths:
        print("No matching FITS files.")
        return

    for path in paths:
        try:
            data, header, hdu_name = first_2d_hdu(path)
            shape = f"{data.shape[0]}x{data.shape[1]}"
            unit = str(header.get("BUNIT", "unknown"))
        except Exception as error:
            shape = "unreadable"
            unit = f"ERROR: {error}"
            hdu_name = "unknown"

        relative = path.relative_to(extraction_dir)

        print(
            f"{relative}\n"
            f"    shape={shape}  hdu={hdu_name}  BUNIT={unit}"
        )


def center_crop(
    image: np.ndarray,
    size_pixels: int,
) -> np.ndarray:
    height, width = image.shape

    if size_pixels > min(height, width):
        raise ValueError(
            f"Requested crop {size_pixels}px exceeds image shape "
            f"{image.shape}."
        )

    y0 = (height - size_pixels) // 2
    x0 = (width - size_pixels) // 2

    return image[
        y0 : y0 + size_pixels,
        x0 : x0 + size_pixels,
    ]


def clean_for_display(image: np.ndarray) -> np.ndarray:
    result = np.asarray(image, dtype=np.float32).copy()
    result[~np.isfinite(result)] = 0
    result[result < 0] = 0
    return result


def load_catalog_row(
    catalog_path: Path,
    subhalo_id: int,
) -> dict[str, Any]:
    catalog = pd.read_csv(
        catalog_path,
        comment="#",
        sep=r"\s+",
        names=CATALOG_COLUMNS,
    )

    matches = catalog.loc[
        catalog["subhalo_id"] == subhalo_id
    ]

    if matches.empty:
        raise KeyError(
            f"Subhalo {subhalo_id} is absent from {catalog_path}."
        )

    row = matches.iloc[0]

    return {
        key: (
            int(value)
            if key == "subhalo_id"
            else float(value)
        )
        for key, value in row.to_dict().items()
    }


def expected_atlas_filename(
    subhalo_id: int,
    observer: str,
    band: str,
) -> str:
    return f"TNG{subhalo_id:06d}_{observer}_LSST_{band}.fits"


def validate_atlas_input(
    label: str,
    path: Path,
    *,
    subhalo_id: int,
    observer: str,
    band: str,
) -> None:
    """Guard against silently mislabeled captioning data.

    The build command trusts --subhalo-id/--observer to write the
    LLaVA record's metadata, independently of the --u/--g/--z paths.
    Without this check, a mismatched or dust-free (_nodust) file would
    still produce a confidently-labeled but wrong record.
    """
    expected = expected_atlas_filename(subhalo_id, observer, band)

    if path.name != expected:
        raise ValueError(
            f"{label} image {path.name!r} does not match the "
            f"requested subhalo/observer/band. Expected {expected!r}. "
            "Pass the dust-attenuated Atlas file for the same "
            "--subhalo-id/--observer, not a --nodust variant or a "
            "different object's product."
        )


def command_build(args: argparse.Namespace) -> None:
    object_dir = object_directory(args.subhalo_id)
    catalog_path = object_dir / "atlas_catalog.txt"

    paths = {
        "lsst_u": Path(args.u),
        "lsst_g": Path(args.g),
        "lsst_z": Path(args.z),
    }
    bands = {"lsst_u": "u", "lsst_g": "g", "lsst_z": "z"}

    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} image does not exist: {path}"
            )

        validate_atlas_input(
            label,
            path,
            subhalo_id=args.subhalo_id,
            observer=args.observer,
            band=bands[label],
        )

    images: dict[str, np.ndarray] = {}
    headers: dict[str, fits.Header] = {}
    hdu_names: dict[str, str] = {}

    for label, path in paths.items():
        image, header, hdu_name = first_2d_hdu(path)
        images[label] = image
        headers[label] = header
        hdu_names[label] = hdu_name

    shapes = {image.shape for image in images.values()}
    if len(shapes) != 1:
        raise ValueError(f"Band shapes do not agree: {shapes}")

    # Atlas sampling is 100 pc per pixel.
    crop_pixels = round(args.crop_kpc / 0.1)

    cropped = {
        label: clean_for_display(
            center_crop(image, crop_pixels)
        )
        for label, image in images.items()
    }

    # A single shared scale preserves the relative band intensities.
    stack = np.stack(
        [
            cropped["lsst_z"],
            cropped["lsst_g"],
            cropped["lsst_u"],
        ]
    )

    positive = stack[np.isfinite(stack) & (stack > 0)]

    if positive.size == 0:
        raise ValueError("All selected image pixels are non-positive.")

    scale = float(np.percentile(positive, args.percentile))

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid RGB scale: {scale}")

    red = cropped["lsst_z"] / scale
    green = cropped["lsst_g"] / scale
    blue = cropped["lsst_u"] / scale

    rgb = make_lupton_rgb(
        red,
        green,
        blue,
        stretch=args.stretch,
        Q=args.q,
    )

    output_dir = (
        REPO_ROOT
        / "data"
        / "atlas_outputs"
        / "demo"
        / f"TNG{args.subhalo_id:06d}_{args.observer}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "atlas_lsst_ugz.png"
    bands_path = output_dir / "atlas_lsst_ugz_bands.npz"
    metadata_path = output_dir / "metadata.json"
    llava_path = output_dir / "llava_record.json"

    # Flip vertically so the PNG agrees with the common FITS display
    # convention imshow(..., origin="lower").
    Image.fromarray(np.flipud(rgb)).save(png_path)

    np.savez_compressed(
        bands_path,
        lsst_u=cropped["lsst_u"],
        lsst_g=cropped["lsst_g"],
        lsst_z=cropped["lsst_z"],
    )

    properties = load_catalog_row(
        catalog_path,
        args.subhalo_id,
    )

    metadata = {
        "simulation": "TNG50-1",
        "snapshot": 99,
        "simulation_redshift": 0.0,
        "subhalo_id": args.subhalo_id,
        "observer": args.observer,
        "source": "TNG50-SKIRT Atlas",
        "dust_attenuated": True,
        "bands": ["LSST_u", "LSST_g", "LSST_z"],
        "rgb_mapping": {
            "red": "LSST_z",
            "green": "LSST_g",
            "blue": "LSST_u",
        },
        "atlas_pixel_scale_pc": 100.0,
        "crop_kpc": args.crop_kpc,
        "crop_pixels": crop_pixels,
        "rgb_parameters": {
            "percentile": args.percentile,
            "scale": scale,
            "stretch": args.stretch,
            "Q": args.q,
        },
        "input_files": {
            label: str(path)
            for label, path in paths.items()
        },
        "input_hdus": hdu_names,
        "input_units": {
            label: str(headers[label].get("BUNIT", "unknown"))
            for label in paths
        },
        "physical_properties": properties,
        "outputs": {
            "rgb": str(png_path),
            "bands": str(bands_path),
        },
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    simulation_description = (
        "In the simulation, the galaxy has a stellar mass of "
        f"{properties['stellar_mass_msun']:.3e} solar masses, "
        "a star-formation rate of "
        f"{properties['sfr_msun_per_year']:.3f} solar masses "
        "per year, and a stellar half-mass radius of "
        f"{properties['stellar_half_mass_radius_kpc']:.3f} kpc."
    )

    llava_record = {
        "id": (
            f"TNG50-1_snap099_subhalo"
            f"{args.subhalo_id}_{args.observer}"
        ),
        "image": str(png_path),
        "metadata": metadata,
        "conversations": [
            {
                "from": "human",
                "value": (
                    "<image>\nWhat does this image represent?"
                ),
            },
            {
                "from": "gpt",
                "value": (
                    "This is a dust-attenuated synthetic optical "
                    "image of a TNG50 galaxy at redshift zero. "
                    "It combines the LSST u, g, and z Atlas bands "
                    f"from observer {args.observer}."
                ),
            },
            {
                "from": "human",
                "value": (
                    "What intrinsic properties are supplied by "
                    "the simulation catalog?"
                ),
            },
            {
                "from": "gpt",
                "value": simulation_description,
            },
        ],
    }

    llava_path.write_text(
        json.dumps([llava_record], indent=2),
        encoding="utf-8",
    )

    print(f"RGB:      {png_path}")
    print(f"Bands:    {bands_path}")
    print(f"Metadata: {metadata_path}")
    print(f"LLaVA:    {llava_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one TNG50-SKIRT Atlas demo object."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    download_parser = subparsers.add_parser(
        "download",
        help="Download and extract one Atlas galaxy.",
    )
    download_parser.add_argument(
        "--subhalo-id",
        type=int,
        default=8,
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
    )
    download_parser.set_defaults(function=command_download)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="List extracted FITS products.",
    )
    inspect_parser.add_argument(
        "--subhalo-id",
        type=int,
        default=8,
    )
    inspect_parser.add_argument(
        "--contains",
        nargs="*",
        default=[],
        help=(
            "Only show paths containing all supplied substrings."
        ),
    )
    inspect_parser.set_defaults(function=command_inspect)

    build_parser_ = subparsers.add_parser(
        "build",
        help="Create RGB, metadata, and one LLaVA record.",
    )
    build_parser_.add_argument(
        "--subhalo-id",
        type=int,
        default=8,
    )
    build_parser_.add_argument(
        "--observer",
        default="O2",
    )
    build_parser_.add_argument("--u", required=True)
    build_parser_.add_argument("--g", required=True)
    build_parser_.add_argument("--z", required=True)
    build_parser_.add_argument(
        "--crop-kpc",
        type=float,
        default=30.0,
    )
    build_parser_.add_argument(
        "--percentile",
        type=float,
        default=99.7,
    )
    build_parser_.add_argument(
        "--stretch",
        type=float,
        default=0.15,
    )
    build_parser_.add_argument(
        "--q",
        type=float,
        default=8.0,
    )
    build_parser_.set_defaults(function=command_build)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()