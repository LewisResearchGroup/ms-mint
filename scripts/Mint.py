#!/usr/bin/env python

import os
import sys
import subprocess
import multiprocessing
import argparse
from waitress import serve

import ms_mint



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='MINT frontend.')

    parser.add_argument('--no-browser', action='store_true', default=False)
    parser.add_argument('--version', default=False, action='store_true')
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--port', type=int, default=9999)

    args = parser.parse_args()

    if args.version:
        print('Mint version:', ms_mint.__version__)
        exit()

    url = f'http://localhost:{args.port}'
    
    if not args.no_browser:
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

    
    if  args.data_dir is not None: 
        os.environ["MINT_DATA_DIR"] = args.data_dir

    from ms_mint.app.app import app

    #app.run_server(debug=args.debug, port=args.port)
    serve(app.server, port=9999)