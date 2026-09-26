from pathlib import Path

from astropy.io import fits


CUBE_PATH = Path("./data/TNG50-11-10_8sqarcmin/dust0-S18K13/angle1/individual_datacubes/galsyn_tng50_phalo_45_493093.fits.gz")


def main() -> None:
    if not CUBE_PATH.is_file():
        raise FileNotFoundError(CUBE_PATH)

    with fits.open(CUBE_PATH, memmap=True) as hdul:
        hdul.info()

        print("\nExtensions:")
        for index, hdu in enumerate(hdul):
            shape = None if hdu.data is None else hdu.data.shape
            unit = hdu.header.get("BUNIT", "unknown")

            print(
                f"{index:3d}  "
                f"{hdu.name:40s}  "
                f"shape={str(shape):18s}  "
                f"unit={unit}"
            )


if __name__ == "__main__":
    main()