#!/usr/bin/env python

import os
import sys
import subprocess
import multiprocessing
import argparse
import wget
import pkg_resources
import xlsxwriter

from waitress import serve
from os.path import expanduser
from pathlib import Path as P
from collections import namedtuple
from multiprocessing import freeze_support

from scipy.spatial.transform import _rotation_groups
import ms_mint


def fake_use_wget():
    "Need this here for pyinstaller/flake8"
    wget


"""
Simple module that monkey patches pkg_resources.get_distribution used by dash
to determine the version of Flask-Compress which is not available with a
flask_compress.__version__ attribute. Known to work with dash==1.16.3 and
PyInstaller==3.6.
"""

welcome = r"""
 __________________________________________________________________________________________________________
/___/\\\\____________/\\\\__/\\\\\\\\\\\__/\\\\\_____/\\\__/\\\\\\\\\\\\\\\_______________/\\\_____________\
|___\/\\\\\\________/\\\\\\_\/////\\\///__\/\\\\\\___\/\\\_\///////\\\/////______________/\\\\\\\__________|
|____\/\\\//\\\____/\\\//\\\_____\/\\\_____\/\\\/\\\__\/\\\_______\/\\\__________________/\\\\\\\\\________|
|_____\/\\\\///\\\/\\\/_\/\\\_____\/\\\_____\/\\\//\\\_\/\\\_______\/\\\_________________\//\\\\\\\________|
|______\/\\\__\///\\\/___\/\\\_____\/\\\_____\/\\\\//\\\\/\\\_______\/\\\__________________\//\\\\\________|
|_______\/\\\____\///_____\/\\\_____\/\\\_____\/\\\_\//\\\/\\\_______\/\\\___________________\//\\\________|
|________\/\\\_____________\/\\\_____\/\\\_____\/\\\__\//\\\\\\_______\/\\\____________________\///________|
|_________\/\\\_____________\/\\\__/\\\\\\\\\\\_\/\\\___\//\\\\\_______\/\\\_____________________/\\\______|
|__________\///______________\///__\///////////__\///_____\/////________\///_____________________\///______|
\__________________________________________________________________________________________________________/
       \
        \   ^__^
         \  (@@)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
"""


def _get_distribution(dist):
    if IS_FROZEN and dist == "flask-compress":
        return _Dist("1.5.0")
    else:
        return _true_get_distribution(dist)


if __name__ == "__main__":
    freeze_support()

    HOME = expanduser("~")
    DATADIR = str(P(HOME) / "MINT")
    IS_FROZEN = hasattr(sys, "_MEIPASS")

    # backup true function
    _true_get_distribution = pkg_resources.get_distribution
    # create small placeholder for the dash call
    # _flask_compress_version = parse_version(get_distribution("flask-compress").version)
    _Dist = namedtuple("_Dist", ["version"])

    # monkey patch the function so it can work once frozen and pkg_resources is of
    # no help
    pkg_resources.get_distribution = _get_distribution

    parser = argparse.ArgumentParser(description="MINT frontend.")

    parser.add_argument(
        "--no-browser",
        action="store_true",
        default=False,
        help="do not start the browser",
    )
    parser.add_argument(
        "--version", default=False, action="store_true", help="print current version"
    )
    parser.add_argument(
        "--data-dir", default=DATADIR, help="target directory for MINT data"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="start MINT server in debug mode",
    )
    parser.add_argument("--port", type=int, default=9999, help="Port to use")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host binding address"
    )
    parser.add_argument(
        "--serve-path",
        default=None,
        type=str,
        help="(deprecated) serve app at a different path e.g. '/mint/' to serve the app at 'localhost:9999/mint/'",
    )

    args = parser.parse_args()

    if args.version:
        print("Mint version:", ms_mint.__version__)
        exit()

    url = f"http://{args.host}:{args.port}"

    if not args.no_browser:
        if os.name == "nt":
            # https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Multiprocessing
            multiprocessing.freeze_support()

        # Open the browser
        if sys.platform in ["win32", "nt"]:
            os.startfile(url)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", url])
        else:
            try:
                subprocess.Popen(["xdg-open", url])
            except OSError:
                print("Please open a browser on: ", url)

    if args.data_dir is not None:
        os.environ["MINT_DATA_DIR"] = args.data_dir

    if args.serve_path is not None:
        os.environ["MINT_SERVE_PATH"] = args.serve_path

    print(welcome)

    print("Loading app...")

    from ms_mint.app.app import create_app, register_callbacks

    app, cache, fsc = create_app()
    register_callbacks(app, cache, fsc)

    app.css.config.serve_locally = True
    app.scripts.config.serve_locally = True

    print("Configuration done starting server...")

    if args.debug:

        app.run_server(
            debug=args.debug,
            port=args.port,
            host=args.host,
            dev_tools_hot_reload=False,
            dev_tools_hot_reload_interval=3000,
            dev_tools_hot_reload_max_retry=30,
        )
    else:
        serve(app.server, port=args.port, host=args.host)
