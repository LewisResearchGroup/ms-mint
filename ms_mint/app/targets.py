import os
import logging

import pandas as pd

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash_tabulator import DashTabulator

import dash_bootstrap_components as dbc

from . import tools as T

from ms_mint.standards import TARGETS_COLUMNS


columns = [{"name": i, "id": i, "selectable": True} for i in TARGETS_COLUMNS]

tabulator_options = {
    "groupBy": "Label",
    "selectable": True,
    "headerFilterLiveFilterDelay": 3000,
    "layout": "fitDataFill",
    "height": "900px",
}

downloadButtonType = {
    "css": "btn btn-primary",
    "text": "Export",
    "type": "csv",
    "filename": "Targets",
}

clearFilterButtonType = {"css": "btn btn-outline-dark", "text": "Clear Filters"}

pkl_table = html.Div(
    id="pkl-table-container",
    style={"minHeight": 100, "margin": "50px 50px 0px 0px"},
    children=[
        DashTabulator(
            id="pkl-table",
            columns=T.gen_tabulator_columns(
                [
                    "peak_label",
                    "mz_mean",
                    "mz_width",
                    "rt",
                    "rt_min",
                    "rt_max",
                    "intensity_threshold",
                    "target_filename",
                ]
            ),
            options=tabulator_options,
            downloadButtonType=downloadButtonType,
            clearFilterButtonType=clearFilterButtonType,
        )
    ],
)

_label = "Targets"

_layout = html.Div(
    [
        html.H3(_label),
        dcc.Upload(
            id="pkl-upload",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            # Allow multiple files to be uploaded
            multiple=True,
        ),
        dcc.Dropdown(
            id="pkl-ms-mode",
            options=[
                {
                    "value": "positive",
                    "label": "Add proton mass to formula (positive mode)",
                },
                {
                    "value": "negative",
                    "label": "Subtract proton mass from formula (negative mode)",
                },
            ],
            value=None,
        ),
        html.Button("Save", id="pkl-save"),
        html.Button("Clear", id="pkl-clear", style={"float": "right"}),
        pkl_table,
    ]
)


_outputs = html.Div(
    id="pkl-outputs",
    children=[
        html.Div(id={"index": "pkl-upload-output", "type": "output"}),
        html.Div(id={"index": "pkl-save-output", "type": "output"}),
        html.Div(id={"index": "pkl-clear-output", "type": "output"}),
    ],
)


def layout():
    return _layout


def callbacks(app, fsc=None, cache=None):
    @app.callback(
        Output("pkl-table", "data"),
        Input("pkl-upload", "contents"),
        Input("pkl-ms-mode", "value"),
        Input("pkl-clear", "n_clicks"),
        State("pkl-upload", "filename"),
        State("pkl-upload", "last_modified"),
        State("wdir", "children"),
    )
    def pkl_upload(
        list_of_contents, ms_mode, clear, list_of_names, list_of_dates, wdir
    ):
        prop_id = dash.callback_context.triggered[0]["prop_id"]
        if prop_id.startswith("pkl-clear"):
            return pd.DataFrame(columns=TARGETS_COLUMNS).to_dict("records")
        target_dir = os.path.join(wdir, "target")
        fn = T.get_targets_fn(wdir)
        if list_of_contents is not None:
            # process data in drag n drop field
            dfs = [
                T.parse_pkl_files(c, n, d, target_dir, ms_mode=ms_mode)
                for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
            ]
            data = dfs[0].to_dict("records")
            return data
        elif os.path.isfile(fn):
            # read from harddrive
            return T.read_targets(fn).to_dict("records")
        else:
            logging.warning(f"Targets file not found: {fn}")

    @app.callback(
        Output({"index": "pkl-save-output", "type": "output"}, "children"),
        Input("pkl-save", "n_clicks"),
        Input("pkl-table", "data"),
        Input("pkl-table", "dataChanged"),
        State("wdir", "children"),
    )
    def plk_save(n_clicks, data, data_changed, wdir):
        df = pd.DataFrame(data)
        if len(df) == 0:
            df = pd.DataFrame(columns=TARGETS_COLUMNS)
        T.write_targets(df, wdir)
        return dbc.Alert("Peaklist saved.", color="info")

    @app.callback(
        Output("pkl-table", "downloadButtonType"),
        Input("tab", "value"),
        State("active-workspace", "children"),
    )
    def table_export_fn(tab, ws_name):
        fn = f"{T.today()}-{ws_name}_MINT-target"
        downloadButtonType = {
            "css": "btn btn-primary",
            "text": "Export",
            "type": "csv",
            "filename": fn,
        }
        return downloadButtonType
