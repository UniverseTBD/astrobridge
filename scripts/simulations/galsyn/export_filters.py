from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


PIXEDFIT_ROOT = Path("external/piXedfit")
WAVELENGTH_DATABASE = (
    PIXEDFIT_ROOT / "data/filters/filters_w.hdf5"
)
TRANSMISSION_DATABASE = (
    PIXEDFIT_ROOT / "data/filters/filters_t.hdf5"
)

OUTPUT_DIR = Path("data/filters")

LSST_FILTERS = [
    "lsst_u",
    "lsst_g",
    "lsst_r",
    "lsst_i",
    "lsst_z",
    "lsst_y",
]


def main() -> None:
    if not WAVELENGTH_DATABASE.is_file():
        raise FileNotFoundError(WAVELENGTH_DATABASE)

    if not TRANSMISSION_DATABASE.is_file():
        raise FileNotFoundError(TRANSMISSION_DATABASE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (
        h5py.File(WAVELENGTH_DATABASE, "r") as wavelength_db,
        h5py.File(TRANSMISSION_DATABASE, "r") as transmission_db,
    ):
        available = set(wavelength_db.keys())

        for filter_name in LSST_FILTERS:
            if filter_name not in available:
                matches = sorted(
                    name
                    for name in available
                    if filter_name.split("_")[-1] in name
                )

                raise KeyError(
                    f"{filter_name!r} is unavailable. "
                    f"Possible matches: {matches[:20]}"
                )

            wavelength = np.asarray(
                wavelength_db[filter_name][:],
                dtype=np.float64,
            )

            transmission = np.asarray(
                transmission_db[filter_name][:],
                dtype=np.float64,
            )

            if wavelength.shape != transmission.shape:
                raise ValueError(
                    f"Shape mismatch for {filter_name}: "
                    f"{wavelength.shape} versus "
                    f"{transmission.shape}"
                )

            output_path = OUTPUT_DIR / f"{filter_name}.txt"

            np.savetxt(
                output_path,
                np.column_stack(
                    [wavelength, transmission]
                ),
                fmt="%.10e",
                header="wavelength_angstrom transmission",
            )

            print(
                f"{filter_name}: {output_path} "
                f"({wavelength.min():.1f}–"
                f"{wavelength.max():.1f} Å)"
            )


if __name__ == "__main__":
    main()