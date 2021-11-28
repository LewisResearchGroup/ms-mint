import os
import tempfile
import logging
import tempfile

import pandas as pd

from pathlib import Path as P

import matplotlib

matplotlib.use("Agg")

import dash
from dash import html, dcc

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_extensions import Download
from dash_extensions.enrich import FileSystemCache

import dash_bootstrap_components as dbc

from flask_caching import Cache
from flask_login import current_user


from . import tools as T

from . import workspaces
from . import ms_files
from . import metadata
from . import targets
from . import peak_optimization
from . import processing
from . import add_metab
from . import analysis
from . import messages

import dash_uploader as du


def make_dirs():
    tmpdir = tempfile.gettempdir()
    tmpdir = os.path.join(tmpdir, "MINT")
    tmpdir = os.getenv("MINT_DATA_DIR", default=tmpdir)
    cachedir = os.path.join(tmpdir, ".cache")
    os.makedirs(tmpdir, exist_ok=True)
    os.makedirs(cachedir, exist_ok=True)
    print("MAKEDIRS:", tmpdir, cachedir)
    return P(tmpdir), P(cachedir)


TMPDIR, CACHEDIR = make_dirs()

config = {
    "DEBUG": True,  # some Flask specific configs
    "CACHE_TYPE": "simple",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300,
}

pd.options.display.max_colwidth = 1000

_modules = [
    workspaces,
    ms_files,
    metadata,
    targets,
    add_metab,
    peak_optimization,
    processing,
    analysis,
]

modules = {module._label: module for module in _modules}

# Collect outputs:
_outputs = html.Div(
    id="outputs",
    children=[module._outputs for module in _modules if module._outputs is not None],
    style={"visibility": "hidden"},
)


logout_button = (
    dbc.Button(
        "Logout",
        id="logout-button",
        style={"marginRight": "10px", "visibility": "hidden"},
    ),
)
logout_button = html.A(href="/logout", children=logout_button)

_layout = html.Div(
    [
        html.Div(logout_button),
        dcc.Interval(
            id="progress-interval", n_intervals=0, interval=2000, disabled=False
        ),
        html.A(
            href="https://sorenwacker.github.io/ms-mint/gui/",
            children=[
                html.Button(
                    "Documentation",
                    id="B_help",
                    style={"float": "right", "color": "info"},
                )
            ],
            target="_blank",
        ),
        html.A(
            href=f"https://github.com/sorenwacker/ms-mint/issues/new?body={T.get_issue_text()}",
            children=[
                html.Button(
                    "Issues", id="B_issues", style={"float": "right", "color": "info"}
                )
            ],
            target="_blank",
        ),
        dbc.Progress(
            id="progress-bar",
            value=100,
            style={"marginBottom": "20px", "width": "100%", "marginTop": "20px"},
        ),
        messages.layout(),
        Download(id="res-download-data"),
        html.Div(id="tmpdir", children=str(TMPDIR), style={"visibility": "hidden"}),
        html.Div(
            [
                html.P(
                    "Current Workspace: ",
                    style={
                        "display": "inline-block",
                        "marginRight": "5px",
                        "marginTop": "5px",
                    },
                ),
                html.Div(id="active-workspace", style={"display": "inline-block"}),
                html.Div(
                    id="wdir",
                    children="",
                    style={
                        "display": "inline-block",
                        "visibility": "visible",
                        "float": "right",
                    },
                ),
            ],
            style={"marginBottom": "20px"},
        ),
        html.Div(id="pko-creating-chromatograms"),
        dcc.Tabs(
            id="tab",
            value=_modules[0]._label,
            children=[
                dcc.Tab(
                    id=modules[key]._label,
                    value=key,
                    label=modules[key]._label,
                )
                for key in modules.keys()
            ],
        ),
        html.Div(id="pko-image-store", style={"visibility": "hidden", "height": "0px"}),
        html.Div(id="tab-content"),
        html.Div(id="viewport-container", style={"visibility": "hidden"}),
        _outputs,
    ],
    style={"margin": "2%"},
)


def register_callbacks(app, cache, fsc):
    logging.warning("Register callbacks")
    upload_root = os.getenv("MINT_DATA_DIR", tempfile.gettempdir())
    upload_dir = str(P(upload_root) / "MINT-Uploads")
    UPLOAD_FOLDER_ROOT = upload_dir
    du.configure_upload(app, UPLOAD_FOLDER_ROOT)

    messages.callbacks(app=app, fsc=fsc, cache=cache)

    for module in _modules:
        func = module.callbacks
        if func is not None:
            func(app=app, fsc=fsc, cache=cache)

    # Updates the current viewport
    app.clientside_callback(
        """
        function(href) {
            var w = window.innerWidth;
            var h = window.innerHeight;
            return `${w},${h}` ;
        }
        """,
        Output("viewport-container", "children"),
        Input("progress-interval", "n_intervals"),
    )

    @app.callback(
        Output("tab-content", "children"),
        Input("tab", "value"),
        State("wdir", "children"),
    )
    def render_content(tab, wdir):
        func = modules[tab].layout
        if tab != "Workspaces" and wdir == "":
            return dbc.Alert(
                "Please, create and activate a workspace.", color="warning"
            )
        elif (
            tab in ["Metadata", "Peak Optimization", "Processing"]
            and len(T.get_ms_fns(wdir)) == 0
        ):
            return dbc.Alert("Please import MS files.", color="warning")
        elif tab in ["Processing"] and (len(T.get_targets(wdir)) == 0):
            return dbc.Alert("Please, define targets.", color="warning")
        elif tab in ["Analysis"] and not P(T.get_results_fn(wdir)).is_file():
            return dbc.Alert("Please, create results (Processing).", color="warning")
        if func is not None:
            return func()
        else:
            raise PreventUpdate

    @app.callback(
        Output("tmpdir", "children"),
        Output("logout-button", "style"),
        Input("progress-interval", "n_intervals"),
    )
    def upate_tmpdir(x):
        if hasattr(app.server, "login_manager"):
            username = current_user.username
            logging.warning(f"User: {username}")
            tmpdir = str(TMPDIR / "User" / username)
            logging.warning(tmpdir)
            return tmpdir, {"visibility": "visible"}
        logging.info("Hide login button")
        return str(TMPDIR / "Local"), {"visibility": "hidden"}


def create_app(**kwargs):

    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.MINTY,
            "https://codepen.io/chriddyp/pen/bWLwgP.css",
        ],
        # requests_pathname_prefix=os.getenv('MINT_SERVE_PATH', default='/'),
        # routes_pathname_prefix=os.getenv('MINT_SERVE_PATH', default='/'),
        **kwargs,
    )

    # app.css.config.serve_locally = True
    # app.scripts.config.serve_locally = True

    app.layout = _layout
    app.title = "MINT"
    app.config["suppress_callback_exceptions"] = True

    upload_root = os.getenv("MINT_DATA_DIR", tempfile.gettempdir())
    CACHE_DIR = str(P(upload_root) / "MINT-Cache")

    logging.info("Cache directory: {}".format(CACHE_DIR))

    cache = Cache(
        app.server, config={"CACHE_TYPE": "filesystem", "CACHE_DIR": CACHE_DIR}
    )

    fsc = FileSystemCache(str(CACHEDIR))

    return app, cache, fsc


if __name__ == "__main__":
    app, cache = create_app()
    register_callbacks(app, cache)
    app.run_server(
        debug=True,
        threaded=True,
        dev_tools_hot_reload_interval=5000,
        dev_tools_hot_reload_max_retry=30,
    )
