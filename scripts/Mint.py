#!/usr/bin/env python

import os
import sys
import subprocess
import multiprocessing

import pandas as pd
from os.path import isfile
from glob import glob

import ms_mint
from ms_mint.app.App import app, mint


if __name__ == '__main__':
    
    args = sys.argv


    url = 'http://localhost:9999'
    
    if not '--no-browser' in args:
        if os.name == 'nt':
            # https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Multiprocessing
            multiprocessing.freeze_support()
            
        if sys.platform in ['win32', 'nt']:
            os.startfile(url)
            
        elif sys.platform=='darwin':
            subprocess.Popen(['open', url])
            
        else:
            try:
                subprocess.Popen(['xdg-open', url])
            except OSError:
                print('Please open a browser on: ', url)

    if '--version' in args:
        print('Mint version:', ms_mint.__version__)
        exit()
    
    if '--debug' in args:
        DEBUG = True
        mint.verbose = True

    else:
        DEBUG = False

    if '--verbose' in args:
        mint.verbose = True

    if '--data' in args:
        mint.files = glob('/data/metabolomics_storage/MINT/MINT_demofiles/*/**.mzXML', recursive=True)[:4]
        mint.peaklist_files = 'tests/data/example_peaklist.csv'

        print('MINT files:', mint.files)
        print('MINTegration list:')
        print(mint.peaklist.to_string())
        
    app.run_server(debug=DEBUG, port=9999)