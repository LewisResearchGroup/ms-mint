#!/usr/env python

import sys
import pandas as pd
from os.path import isfile
from glob import glob

from ms_mint.dash_gui import app, mint

if __name__ == '__main__':
    
    args = sys.argv
    
    if '--debug' in args:
        DEBUG = True
        mint.peaklist_files = ['tests/data/peaklist_v0.csv']
        mint.files = glob('**/*.mzXML', recursive=True)
        print('MINT files:', mint.files)
        print('MINTegration list:')
        print(mint.peaklist.to_string())
    else:
        DEBUG = False

    if '--data' in args:
        if isfile('/tmp/mint_results.csv'):
            mint._results = pd.read_csv('/tmp/mint_results.csv')
        for i in mint.files:
            assert isfile(i)
        for i in mint.peaklist_files:
            assert isfile(i)

    app.run_server(debug=DEBUG, port=9999)
