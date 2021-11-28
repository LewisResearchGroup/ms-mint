import os
import pandas as pd

from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_tabulator import DashTabulator

import dash_bootstrap_components as dbc

from ms_mint.standards import M_PROTON
from . import tools as T

from pkg_resources import resource_filename

CHEBI_CHEM_FN = os.path.abspath(
    resource_filename("ms_mint.static", "ChEBI-Chem.parquet")
)
CHEBI_GROUPS_FN = os.path.abspath(
    resource_filename("ms_mint.static", "ChEBI-Groups.parquet")
)
CHEBI_CHEM = pd.read_parquet(CHEBI_CHEM_FN)
CHEBI_GROUPS = pd.read_parquet(CHEBI_GROUPS_FN)


options = {
    "selectable": True,
    "headerFilterLiveFilterDelay": 3000,
    "layout": "fitDataFill",
    "height": "600px",
}

clearFilterButtonType = {"css": "btn btn-outline-dark", "text": "Clear Filters"}


groups_options = [{"label": "All", "value": "all"}] + [
    {"label": x.capitalize(), "value": x} for x in CHEBI_GROUPS.index
]

columns = T.gen_tabulator_columns(CHEBI_CHEM.columns, editor=None, col_width="auto")

add_metab_table = html.Div(
    id="add-metab-table-container",
    style={"minHeight": 100, "marginTop": "10%"},
    children=[
        dbc.Alert(
            "This feature is experimental. Please, check values and report errors to the issue tracker.",
            color="danger",
        ),
        dcc.Dropdown(
            id="add-metab-ms-mode",
            placeholder="Select ionization mode",
            options=[
                {"label": "Positive", "value": "Positive"},
                {"label": "Negative", "value": "Negative"},
            ],
            value="Negative",
        ),
        dcc.Dropdown(id="add-metab-groups", options=groups_options, multi=True),
        dcc.Loading(
            DashTabulator(
                id="add-metab-table",
                columns=columns,
                options=options,
                clearFilterButtonType=clearFilterButtonType,
            ),
        ),
        html.Button("Add selected metabolites to targets", id="add-metab"),
    ],
)

_label = "Add Metabolites"

_layout = html.Div([html.H3("Add Metabolites"), add_metab_table])

_outputs = html.Div(
    id="add-metab-outputs",
    children=[html.Div(id={"index": "add-metab-output", "type": "output"})],
)


def layout():
    return _layout


def callbacks(app, fsc, cache):
    @app.callback(
        Output({"index": "add-metab-output", "type": "output"}, "children"),
        Input("add-metab", "n_clicks"),
        State("add-metab-table", "multiRowsClicked"),
        State("add-metab-ms-mode", "value"),
        State("wdir", "children"),
    )
    def add_metab(n_clicks, rows, ms_mode, wdir):
        if n_clicks is None:
            raise PreventUpdate
        if ms_mode is None:
            return dbc.Alert("Please select ionization mode.", color="warning")

        targets = T.get_targets(wdir)

        for row in rows:
            charge = int(row["Charge"])
            if (ms_mode == "Negative") and (charge > 0):
                continue
            if (ms_mode == "Positive") and (charge < 0):
                continue
            if charge == 0:
                if ms_mode == "Negative":
                    targets.loc[row["ChEBI Name"], "mz_mean"] = (
                        row["Monoisotopic Mass"] - M_PROTON
                    )
                elif ms_mode == "Positive":
                    targets.loc[row["ChEBI Name"], "mz_mean"] = row[
                        "Monoisotopic Mass"
                    ]  # + M_PROTON
            else:
                targets.loc[row["ChEBI Name"], "mz_mean"] = row["Monoisotopic Mass"]

            targets.loc[row["ChEBI Name"], "mz_width"] = 10
            targets.loc[row["ChEBI Name"], "rt"] = -1
            targets.loc[row["ChEBI Name"], "rt_min"] = 0
            targets.loc[row["ChEBI Name"], "rt_max"] = 15
            targets.loc[row["ChEBI Name"], "intensity_threshold"] = 0

        T.write_targets(targets, wdir)
        n_peaks = len(targets)
        n_new = len(rows)
        return dbc.Alert(
            f"{n_new} peaks added, now {n_peaks} peaks defined.", color="info"
        )

    @app.callback(
        Output("add-metab-table", "data"),
        Input("add-metab-groups", "value"),
        Input("add-metab-ms-mode", "value"),
    )
    def generate_table(groups, ms_mode):
        print("Groups to add", groups)

        data = CHEBI_CHEM

        if ms_mode == "Positive":
            data = data[data.Charge.str.startswith("+") | (data.Charge == "0")]
        elif ms_mode == "Negative":
            data = data[data.Charge.str.startswith("-") | (data.Charge == "0")]

        if groups is None or len(groups) == 0:
            raise PreventUpdate
        elif "all" in groups:
            return data.to_dict("records")

        ids = (
            CHEBI_GROUPS.loc[groups]
            .explode("ChEBI IDs")
            .drop_duplicates()["ChEBI IDs"]
            .values
        )

        return data[data["ChEBI ID"].isin(ids)].to_dict("records")
