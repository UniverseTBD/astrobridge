import argparse
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fetch paper metadata from NASA ADS using a mapping CSV."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input mapping CSV containing the 'bibcode' column",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output paper data CSV",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Your NASA ADS API token. If omitted, the script checks the ADS_API_TOKEN environment variable.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2000,
        help="Number of bibcodes to query per batch (default: 2000)",
    )
    return parser.parse_args()


def resolve_api_token(cli_token):
    if cli_token:
        return cli_token

    env_token = os.environ.get("ADS_API_TOKEN")
    if env_token:
        return env_token

    print("Error: NASA ADS API token not found.")
    sys.exit(1)


def load_unique_bibcodes(input_csv):
    simbad_results_df = pd.read_csv(input_csv)
    unique_bibcodes = simbad_results_df["bibcode"].dropna().unique()
    return unique_bibcodes


def fetch_ads_data(unique_bibcodes, output_csv, token, chunk_size):
    url = "https://api.adsabs.harvard.edu/v1/search/bigquery"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "big-query/csv"}

    ads_data = []

    chunks = [
        unique_bibcodes[i : i + chunk_size]
        for i in range(0, len(unique_bibcodes), chunk_size)
    ]

    for index, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        payload = "bibcode\n" + "\n".join(chunk)

        params = {"q": "*:*", "fl": "bibcode,title,abstract,doi", "rows": 2000}

        try:
            response = requests.post(url, headers=headers, params=params, data=payload)
            response.raise_for_status()

            docs = response.json().get("response", {}).get("docs", [])

            for paper in docs:
                title = paper.get("title", ["No Title Available"])[0]
                abstract = paper.get("abstract", "No Abstract Available")
                doi = paper.get("doi", "No DOI Available")
                bibcode = paper.get("bibcode")
                eprint_url = (
                    f"https://ui.adsabs.harvard.edu/link_gateway/{bibcode}/EPRINT_PDF"
                )

                ads_data.append(
                    {
                        "bibcode": bibcode,
                        "paper_title": title,
                        "abstract": abstract,
                        "doi": doi,
                        "preprint_url": eprint_url,
                    }
                )

        except Exception as e:
            print(f"Error processing chunk {index + 1}: {e}")

        time.sleep(5.0)

    ads_df = pd.DataFrame(ads_data)
    ads_df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(ads_df)} rows to {output_csv}")


def main():
    args = parse_arguments()

    api_token = resolve_api_token(args.token)

    print(f"Loading mapping data from {args.input}...")
    unique_bibcodes = load_unique_bibcodes(args.input)
    print(f"Found {len(unique_bibcodes)} unique papers to fetch.")

    fetch_ads_data(unique_bibcodes, args.output, api_token, args.chunk_size)


if __name__ == "__main__":
    main()
