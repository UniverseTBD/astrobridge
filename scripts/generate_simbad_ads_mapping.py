import argparse
import os
import time

import pandas as pd
import pyvo as vo
from astropy.table import Table
from tqdm.auto import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a mapping CSV from SIMBAD using crossmatch results.")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the input CSV containing main_ids"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Path to save the output mapping CSV"
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=20000, 
        help="Number of IDs to process per TAP upload chunk (default: 20000)"
    )
    return parser.parse_args()

def load_unique_ids(input_csv):
    print(f"Loading input dataset from {input_csv}...")
    full_df = pd.read_csv(input_csv)
    
    unique_ids = full_df['main_id'].dropna().drop_duplicates().reset_index(drop=True)
    print(f"Found {len(unique_ids)} unique objects to map.")
    
    return unique_ids

def generate_mapping(unique_ids, output_csv, chunk_size):
    SIMBAD_TAP_URL = "http://simbad.u-strasbg.fr/simbad/sim-tap"
    print("Connecting to SIMBAD TAP server...")
    service = vo.dal.TAPService(SIMBAD_TAP_URL)

    adql_query = """
    SELECT 
        u.main_id, 
        rf.bibcode
    FROM TAP_UPLOAD.input_table AS u, basic AS b, has_ref AS hr, ref AS rf
    WHERE u.main_id = b.main_id 
      AND b.oid = hr.oidref
      AND hr.oidbibref = rf.oidbib
    """

    chunks = [unique_ids[i:i + chunk_size] for i in range(0, len(unique_ids), chunk_size)]
    write_header = not os.path.exists(output_csv)

    for index, chunk in enumerate(tqdm(chunks, desc="Processing SIMBAD chunks")):
        
        df_upload = pd.DataFrame({'main_id': chunk})
        astropy_upload_table = Table.from_pandas(df_upload)
        
        try:
            job = service.run_sync(
                adql_query, 
                uploads={'input_table': astropy_upload_table},
                maxrec=3000000 
            )
            
            simbad_results_df = job.to_table().to_pandas()
            
            if not simbad_results_df.empty:
                simbad_results_df.to_csv(output_csv, mode='a', index=False, header=write_header)
                write_header = False             
                
        except Exception as e:
            print(f"Error processing SIMBAD chunk {index + 1}: {e}")
            
        time.sleep(10.0)
        
    print(f"\nMapping successfully compiled and saved to {output_csv}")

def main():
    args = parse_arguments()
    unique_ids = load_unique_ids(args.input)
    generate_mapping(unique_ids, args.output, args.chunk_size)

if __name__ == "__main__":
    main()