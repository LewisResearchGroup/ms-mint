#!/usr/bin/env python

import os
import sys
import subprocess
import multiprocessing
import argparse
from waitress import serve

import ms_mint

from os.path import expanduser
from pathlib import Path as P

HOME = expanduser("~")
DATADIR = str( P(HOME)/'MINT' )


"""
Simple module that monkey patches pkg_resources.get_distribution used by dash
to determine the version of Flask-Compress which is not available with a
flask_compress.__version__ attribute. Known to work with dash==1.16.3 and
PyInstaller==3.6.
"""

from collections import namedtuple

import pkg_resources

IS_FROZEN = hasattr(sys, '_MEIPASS')

# backup true function
_true_get_distribution = pkg_resources.get_distribution
# create small placeholder for the dash call
# _flask_compress_version = parse_version(get_distribution("flask-compress").version)
_Dist = namedtuple('_Dist', ['version'])

def _get_distribution(dist):
    if IS_FROZEN and dist == 'flask-compress':
        return _Dist('1.5.0')
    else:
        return _true_get_distribution(dist)

# monkey patch the function so it can work once frozen and pkg_resources is of
# no help
pkg_resources.get_distribution = _get_distribution



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='MINT frontend.')

    parser.add_argument('--no-browser', action='store_true', default=False, 
        help='do not start the browser')
    parser.add_argument('--version', default=False, action='store_true', 
        help='print current version')
    parser.add_argument('--data-dir', default=DATADIR, 
        help='target directory for MINT data')
    parser.add_argument('--debug', default=False, action='store_true', 
        help='start MINT server in debug mode')
    parser.add_argument('--port', type=int, default=9999, 
        help='change the port')
    parser.add_argument('--serve-path', default=None, type=str, 
        help="serve app at a different path e.g. '/mint/' to serve the app at 'localhost:9999/mint/'")

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
    
    if args.serve_path is not None:
        os.environ['MINT_SERVE_PATH'] = args.serve_path

    from ms_mint.app.app import app

    if args.debug:
        app.run_server(debug=args.debug, port=args.port)
    else:
        serve(app.server, port=9999)