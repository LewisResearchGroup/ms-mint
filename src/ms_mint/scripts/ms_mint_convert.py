#!/usr/bin/env python

from ms_mint import io
from pathlib import Path as P
from tqdm import tqdm
import argparse
import os
import logging


def convert(fn, fmt='parquet', output_directory=None):
    
    if output_directory is None:
        output_directory = P(fn).parent
        
    fn_out = output_directory / P(fn).with_suffix(f".{fmt}")
    
    if fn_out.is_file():
        logging.error(f"File exists {fn_out}")
    else:
        os.makedirs(output_directory, exist_ok=True)
        logging.info(f"{fn} --> {fn_out}")
        
        if fmt == 'parquet':
            io.ms_file_to_df(fn).to_parquet(fn_out)
        elif fmt == 'feather':
            io.ms_file_to_df(fn).to_feather(fn_out)
         
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", nargs="+", required=True)
    parser.add_argument("-o", "--output-directory")
    parser.add_argument("-f", "--format", choices=['parquet', 'feather'], default='parquet')

    args = parser.parse_args()
    fns = args.input
    output_directory = args.output_directory
    fmt = args.format
    
    for fn in tqdm(fns):
        convert(fn, fmt, output_directory)


if __name__ == "__main__":
    main()
