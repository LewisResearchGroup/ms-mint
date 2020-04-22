#!/usr/bin/env python

import sys
import pandas as pd
from os.path import isfile
from glob import glob

import ms_mint
from ms_mint.dash_gui import app, mint
import multiprocessing
import os


if __name__ == '__main__':
    if os.name == 'nt':
        print('On windows')
        # https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Multiprocessing
        multiprocessing.freeze_support()
    
    args = sys.argv
    
    if '--version' in args:
        print('Mint version:', ms_mint.__version__)
        exit()
    
    if '--debug' in args:
        DEBUG = True
        mint.verbose = True
        mint.peaklist_files = ['/data/metabolomics_storage/MINT/MINT_peaklists/MINT-peaklist__MSMLSListCol001Neg.csv']
        mint.peaklist = mint.peaklist
        mint.files = glob('/data/metabolomics_storage/HILICneg15/**.mzXML', recursive=True)
        print('MINT files:', mint.files)
        print('MINTegration list:')
        print(mint.peaklist.to_string())

    else:
        DEBUG = False

    if '--verbose' in args:
        mint.verbose = True

    if '--data' in args:
        if isfile('/tmp/mint_results.csv'):
            mint._results = pd.read_csv('/tmp/mint_results.csv')
        for i in mint.files:
            assert isfile(i)
        for i in mint.peaklist_files:
            assert isfile(i)

    app.run_server(debug=DEBUG, port=9999)