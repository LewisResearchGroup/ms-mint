import os
import shutil
import uuid
import logging
import numpy as np
import tempfile

from pathlib import Path as P

from glob import glob

import pandas as pd

from dash import html, dcc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

import dash_bootstrap_components as dbc


from ms_mint.io import convert_ms_file_to_feather

from dash_tabulator import DashTabulator
from tqdm import tqdm

import dash_uploader as du

from . import tools as T


upload_root = os.getenv("MINT_DATA_DIR", tempfile.gettempdir())
upload_dir = str(P(upload_root) / "MINT-Uploads")
UPLOAD_FOLDER_ROOT = upload_dir


options = {
    "selectable": True,
    "headerFilterLiveFilterDelay": 3000,
    "layout": "fitDataFill",
    "height": "900px",
}

clearFilterButtonType = {"css": "btn btn-outline-dark", "text": "Clear Filters"}


columns = [
    {
        "formatter": "rowSelection",
        "titleFormatter": "rowSelection",
        "titleFormatterParams": {
            "rowRange": "active"  # only toggle the values of the active filtered rows
        },
        "hozAlign": "center",
        "headerSort": False,
        "width": "1px",
        "frozen": True,
    },
    {
        "title": "MS-file",
        "field": "MS-file",
        "headerFilter": True,
        "headerSort": True,
        "editor": None,
        "width": "80%",
        "sorter": "string",
        "frozen": True,
    },
    {
        "title": "Size [MB]",
        "field": "file_size",
        "headerFilter": True,
        "headerSort": True,
        "editor": None,
        "width": "20%",
        "sorter": "string",
        "frozen": True,
    },
    {
        "title": "",
        "field": "",
        "headerFilter": False,
        "formatter": "color",
        "width": "3px",
        "headerSort": False,
    },
]


ms_table = html.Div(
    id="ms-table-container",
    style={"Height": 0, "marginTop": "10%"},
    children=[
        DashTabulator(
            id="ms-table",
            columns=columns,
            options=options,
            clearFilterButtonType=clearFilterButtonType,
        )
    ],
)

_label = "MS-Files"

_layout = html.Div(
    [
        html.H3("Upload MS-files"),
        html.Div(
            du.Upload(
                id="ms-uploader",
                filetypes=["tar", "zip", "mzxml", "mzml", "mzXML", "mzML"],
                upload_id=uuid.uuid1(),
                max_files=10000,
                pause_button=True,
                cancel_button=True,
                text="Upload mzXML/mzML files.",
            ),
            style={
                "textAlign": "center",
                "width": "100%",
                "padding": "0px",
                "margin-bottom": "20px",
                "display": "inline-block",
            },
        ),
        html.Button("Import from URL", id="ms-import-from-url"),
        dcc.Input(
            id="url", placeholder="Drop URL / path here", style={"width": "100%"}
        ),
        dcc.Markdown("---", style={"marginTop": "10px"}),
        dcc.Markdown("##### Actions"),
        html.Button("Convert to Feather", id="ms-convert"),
        html.Button("Delete selected files", id="ms-delete", style={"float": "right"}),
        html.Div(id="ms-n-files"),
        dcc.Loading(ms_table),
        html.Div(id="ms-uploader-fns", style={"visibility": "hidden"}),
    ]
)


_outputs = html.Div(
    id="ms-outputs",
    children=[
        html.Div(id={"index": "ms-convert-output", "type": "output"}),
        html.Div(id={"index": "ms-delete-output", "type": "output"}),
        html.Div(id={"index": "ms-save-output", "type": "output"}),
        html.Div(id={"index": "ms-import-from-url-output", "type": "output"}),
        html.Div(id={"index": "ms-uploader-output", "type": "output"}),
    ],
)


def layout():
    return _layout


def callbacks(app, fsc, cache):
    @app.callback(
        Output("ms-table", "data"),
        Output("ms-table", "downloadButtonType"),
        Input({"index": "ms-uploader-output", "type": "output"}, "children"),
        Input("wdir", "children"),
        Input({"index": "ms-delete-output", "type": "output"}, "children"),
        Input({"index": "ms-convert-output", "type": "output"}, "children"),
        State("active-workspace", "children"),
    )
    def ms_table(value, wdir, files_deleted, files_converted, workspace):

        ms_files = T.get_ms_fns(wdir)

        data = pd.DataFrame(
            {
                "MS-file": [os.path.basename(fn) for fn in ms_files],
                "file_size": [
                    np.round(os.path.getsize(fn) / 1024 / 1024, 2) for fn in ms_files
                ],
            }
        )

        downloadButtonType = {
            "css": "btn btn-primary",
            "text": "Export",
            "type": "csv",
            "filename": f"{T.today()}__MINT__{workspace}__ms-files",
        }

        return data.to_dict("records"), downloadButtonType

    @app.callback(
        Output({"index": "ms-convert-output", "type": "output"}, "children"),
        Input("ms-convert", "n_clicks"),
        State("ms-table", "multiRowsClicked"),
        State("wdir", "children"),
    )
    def ms_convert(n_clicks, rows, wdir):
        target_dir = os.path.join(wdir, "ms_files")
        if n_clicks is None:
            raise PreventUpdate
        fns = [row["MS-file"] for row in rows]
        fns = [fn for fn in fns if not fn.endswith(".feather")]
        fns = [os.path.join(target_dir, fn) for fn in fns]
        n_total = len(fns)
        for i, fn in enumerate(fns):
            fsc.set("progress", int(100 * (i + 1) / n_total))
            new_fn = convert_ms_file_to_feather(fn)
            if os.path.isfile(new_fn):
                os.remove(fn)
        return dbc.Alert("Files converted to feather format.", color="info")

    @app.callback(
        Output({"index": "ms-delete-output", "type": "output"}, "children"),
        Input("ms-delete", "n_clicks"),
        State("ms-table", "multiRowsClicked"),
        State("wdir", "children"),
    )
    def ms_delete(n_clicks, rows, wdir):
        if n_clicks is None:
            raise PreventUpdate
        target_dir = os.path.join(wdir, "ms_files")
        for row in rows:
            fn = row["MS-file"]
            fn = P(target_dir) / fn
            os.remove(fn)
        return dbc.Alert(f"{len(rows)} files deleted", color="info")

    @app.callback(
        Output("ms-uploader-fns", "children"),
        [Input("ms-uploader", "uploadedFiles")],
        [
            State("ms-uploader", "fileNames"),
            State("ms-uploader", "upload_id"),
            State("ms-uploader", "isCompleted"),
            State("ms-uploader", "newestUploadedFileName"),
        ],
    )
    def callback_on_completion(n_files, filenames, upload_id, iscompleted, latest_file):
        if n_files == 0:
            return  # no files uploaded yet.
        out = []
        if filenames is not None:
            if upload_id:
                root_folder = P(UPLOAD_FOLDER_ROOT) / upload_id
            else:
                root_folder = P(UPLOAD_FOLDER_ROOT)

            for filename in filenames:
                file = root_folder / filename
                out.append(file)
            return str(file)
        return []

    @app.callback(
        Output({"index": "ms-uploader-output", "type": "output"}, "children"),
        Input("ms-uploader-fns", "children"),
        State("wdir", "children"),
    )
    def get_a_list(fn, wdir):
        if fn is None:
            raise PreventUpdate
        ms_dir = T.get_ms_dirname(wdir)
        fn_new = P(ms_dir) / P(fn).name
        shutil.move(fn, fn_new)
        logging.info(f"Move {fn} to {fn_new}")
        return dbc.Alert("Upload finished", color="success")

    @app.callback(
        Output({"index": "ms-import-from-url-output", "type": "output"}, "children"),
        Input("ms-import-from-url", "n_clicks"),
        State("url", "value"),
        State("wdir", "children"),
    )
    def import_from_url_or_path(n_clicks, url, wdir):
        if n_clicks is None or url is None:
            raise PreventUpdate
        url = url.strip()
        ms_dir = T.get_ms_dirname(wdir)
        logging.warning(
            f"Local file not found, looking for URL ({url}) [{P(url).is_dir()}, {os.path.isdir(url)}]"
        )
        fns = T.import_from_url(url, ms_dir, fsc=fsc)
        if fns is None:
            return dbc.Alert(f"No MS files found at {url}", color="warning")
        return dbc.Alert(f"{len(fns)} files imported.", color="success")

    @app.callback(Output("ms-n-files", "children"), Input("ms-table", "data"))
    def n_files(data):
        n_files = len(data)
        return dbc.Alert(f"{n_files} files in current workspace.", color="success")
