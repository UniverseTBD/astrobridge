import argparse
import time
from pathlib import Path

import astropy.units as u
import lsdb
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from tqdm.auto import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Crossmatch an MMU dataset with SIMBAD."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The Hugging Face dataset path (e.g., UniverseTBD/mmu_desi_edr_sv3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The path to save the output CSV",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200_000,
        help="Number of coordinates to query per batch (default: 200,000)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=3.0,
        help="Search radius in arcseconds (default: 3.0)",
    )
    return parser.parse_args()


def load_and_prepare_catalog(dataset_path):
    print(f"Loading catalog from {dataset_path}...")
    catalog = lsdb.open_catalog(f"hf://datasets/{dataset_path}", columns=["object_id"])
    catalog_df = catalog.compute().to_pandas()

    og_number_of_rows = len(catalog_df)
    catalog_df["temp_location_id"] = (
        catalog_df["ra"].astype(str) + " " + catalog_df["dec"].astype(str)
    )
    catalog_df = catalog_df.drop_duplicates(subset="temp_location_id", keep="first")
    catalog_df = catalog_df.drop(columns=["temp_location_id"])
    print(
        f"{len(catalog_df)} unique coordinates found ({len(catalog_df) / og_number_of_rows * 100:.2f}% of original)."
    )

    catalog_df["ra"] = catalog_df["ra"].astype(float)
    catalog_df["dec"] = catalog_df["dec"].astype(float)

    return catalog_df


def merge_batch_results(batch_df, results_df, search_radius):
    if results_df is None or results_df.empty:
        return pd.DataFrame(
            columns=["main_id", "coo_bibcode", "object_id", "ra", "dec"]
        )

    batch_coords = SkyCoord(
        ra=batch_df["ra"].astype(float).to_numpy() * u.deg,
        dec=batch_df["dec"].astype(float).to_numpy() * u.deg,
        frame="icrs",
    )

    results_coords = SkyCoord(
        ra=results_df["ra"].astype(float).to_numpy() * u.deg,
        dec=results_df["dec"].astype(float).to_numpy() * u.deg,
        frame="icrs",
    )

    idx, d2d, _ = batch_coords.match_to_catalog_sky(results_coords)
    valid_matches = d2d <= search_radius

    matched_results = results_df.iloc[idx].reset_index(drop=True)
    merged_df = (
        batch_df.loc[valid_matches, ["object_id", "ra", "dec"]]
        .reset_index(drop=True)
        .copy()
    )
    merged_df["main_id"] = matched_results.loc[valid_matches, "main_id"].values
    merged_df["coo_bibcode"] = matched_results.loc[valid_matches, "coo_bibcode"].values

    return merged_df[["main_id", "coo_bibcode", "object_id", "ra", "dec"]]


def execute_crossmatch_pipeline(catalog_df, output_path, batch_size, radius):
    coords = SkyCoord(
        ra=catalog_df["ra"].to_numpy(dtype=float) * u.deg,
        dec=catalog_df["dec"].to_numpy(dtype=float) * u.deg,
        frame="icrs",
    )
    search_radius = radius * u.arcsec

    print(f"Starting crossmatch in batches of {batch_size}...")
    merged_results = pd.DataFrame(
        columns=["main_id", "coo_bibcode", "object_id", "ra", "dec"]
    )

    for i in tqdm(range(0, len(catalog_df), batch_size)):
        batch_coords = coords[i : i + batch_size]

        result_table = Simbad.query_region(batch_coords, radius=search_radius)

        batch_df = catalog_df.iloc[i : i + batch_size].reset_index(drop=True)
        batch_merged_results = merge_batch_results(
            batch_df,
            result_table.to_pandas() if result_table is not None else None,
            search_radius,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        merged_results = (
            merged_results
            if batch_merged_results.empty
            else batch_merged_results
            if merged_results.empty
            else pd.concat([merged_results, batch_merged_results], ignore_index=True)
        )
        merged_results.to_csv(output_path, index=False)
        time.sleep(30)

    print(f"Crossmatch complete! Final results saved to {output_path}")


def main():
    args = parse_arguments()
    catalog_df = load_and_prepare_catalog(args.dataset)
    execute_crossmatch_pipeline(catalog_df, args.output, args.batch_size, args.radius)


if __name__ == "__main__":
    main()
