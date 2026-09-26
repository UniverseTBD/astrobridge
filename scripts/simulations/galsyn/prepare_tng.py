from __future__ import annotations

import os
import time
from pathlib import Path

import h5py
from dotenv import load_dotenv
from galsyn.simutils_tng import (
    download_cutout_parent_halo_hdf5,
    download_cutout_subhalo_hdf5,
    get_snap_z,
    make_sim_file_from_tng_data,
)


SIMULATION = "TNG50-1"
SNAPSHOT = 45
SUBHALO_ID = 493093

# "parent_halo" reproduces the DR1 `phalo` product.
# "subhalo" produces a smaller isolated-galaxy cutout.
CUTOUT_SCOPE = "parent_halo"

RAW_DIR = Path("data/tng_cutouts")
STANDARDIZED_DIR = Path("data/standardized")


def describe_hdf5(path: Path) -> None:
    """Print basic particle counts and file structure."""
    with h5py.File(path, "r") as handle:
        print(f"\nHDF5 file: {path}")
        print(f"Size: {path.stat().st_size / 1024**2:.2f} MiB")
        print(f"Top-level groups: {list(handle.keys())}")

        if "PartType4" in handle:
            print(
                "Star particles:",
                len(handle["PartType4"]["Masses"]),
            )

        if "PartType0" in handle:
            print(
                "Gas cells:",
                len(handle["PartType0"]["Masses"]),
            )

        if "star" in handle:
            # GalSyn-standardized format.
            for key in handle["star"].keys():
                print(
                    f"star/{key}:",
                    handle["star"][key].shape,
                )
                break

        if "gas" in handle:
            for key in handle["gas"].keys():
                print(
                    f"gas/{key}:",
                    handle["gas"][key].shape,
                )
                break


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("TNG_API_KEY")
    if not api_key:
        raise RuntimeError("TNG_API_KEY is missing from .env")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    STANDARDIZED_DIR.mkdir(parents=True, exist_ok=True)

    scope_label = (
        "phalo"
        if CUTOUT_SCOPE == "parent_halo"
        else "shalo"
    )

    cutout_path = (
        RAW_DIR
        / f"cutout_{scope_label}_{SNAPSHOT}_{SUBHALO_ID}.hdf5"
    )

    sim_file = (
        STANDARDIZED_DIR
        / f"sim_file_tng_{scope_label}_{SNAPSHOT}_{SUBHALO_ID}.hdf5"
    )

    print("GalSyn TNG preparation")
    print("----------------------")
    print(f"Simulation:   {SIMULATION}")
    print(f"Snapshot:     {SNAPSHOT}")
    print(f"SubhaloID:    {SUBHALO_ID}")
    print(f"Cutout scope: {CUTOUT_SCOPE}")
    print(f"Raw cutout:   {cutout_path}")
    print(f"GalSyn input: {sim_file}")

    if not cutout_path.exists():
        print("\nDownloading TNG particle cutout...")
        started = time.perf_counter()

        if CUTOUT_SCOPE == "parent_halo":
            download_cutout_parent_halo_hdf5(
                SNAPSHOT,
                SUBHALO_ID,
                api_key=api_key,
                sim=SIMULATION,
                name=str(cutout_path),
            )

        elif CUTOUT_SCOPE == "subhalo":
            download_cutout_subhalo_hdf5(
                SNAPSHOT,
                SUBHALO_ID,
                api_key=api_key,
                sim=SIMULATION,
                name=str(cutout_path),
            )

        else:
            raise ValueError(
                "CUTOUT_SCOPE must be 'parent_halo' or 'subhalo'"
            )

        print(
            "Download completed in "
            f"{time.perf_counter() - started:.1f} seconds."
        )
    else:
        print("\nRaw cutout already exists; skipping download.")

    describe_hdf5(cutout_path)

    print("\nRetrieving snapshot redshift...")
    redshift = get_snap_z(
        SNAPSHOT,
        sim=SIMULATION,
        api_key=api_key,
    )

    print(f"Snapshot redshift: {redshift:.6f}")

    if not sim_file.exists():
        print("\nCreating GalSyn standardized input...")
        started = time.perf_counter()

        make_sim_file_from_tng_data(
            str(cutout_path),
            redshift,
            cosmo_h=0.6774,
            XH=0.76,
            output_hdf5=str(sim_file),
        )

        print(
            "Standardization completed in "
            f"{time.perf_counter() - started:.1f} seconds."
        )
    else:
        print("\nStandardized input already exists; skipping.")

    describe_hdf5(sim_file)

    print("\nPreparation complete.")
    print(f"Use this file with GalaxySynthesizer:\n{sim_file}")


if __name__ == "__main__":
    main()