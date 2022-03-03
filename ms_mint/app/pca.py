import numpy as np
import seaborn as sns
import plotly.express as px

from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc

from ..Mint import Mint

from . import tools as T

options = [{"value": i, "label": i} for i in ["Standard", "Corner"]]

_layout = html.Div(
    [
        html.H3("Principal Components Analysis"),
        html.Button("Run PCA", id="pca-update"),
        dcc.Dropdown(
            id="pca-options",
            options=options,
            value=["Standard"],
            multi=True,
            placeholder="Scaling used before PCA",
        ),
        html.Label("Number of PCA components"),
        dcc.Slider(
            id="pca-nvars",
            value=3,
            min=2,
            max=10,
            marks={i: f"{i}" for i in range(2, 11)},
        ),
        html.Label("Height of facets"),
        dcc.Slider(
            id="pca-facent-height",
            value=2.5,
            min=1,
            max=5,
            step=0.1,
            marks={i: f"{i}" for i in np.arange(1, 5.5, 0.5)},
        ),
        html.H4("Cumulative explained variance"),
        dcc.Loading(
            html.Div(
                id="pca-figure-explained-variance",
                style={"margin": "auto", "text-align": "center"},
            )
        ),
        html.H4("Scatter plot of principal components"),
        dcc.Loading(
            html.Div(
                id="pca-figure-pairplot",
                style={"margin": "auto", "text-align": "center"},
            )
        ),
        html.H4("Principal components compositions"),
        dcc.Loading(dcc.Graph(id="pca-figure-contrib")),
    ]
)

_label = "PCA"


def layout():
    return _layout


def callbacks(app, fsc, cache):
    @app.callback(
        Output("pca-figure-explained-variance", "children"),
        Output("pca-figure-pairplot", "children"),
        Output("pca-figure-contrib", "figure"),
        Input("pca-update", "n_clicks"),
        State("pca-nvars", "value"),
        State("pca-facent-height", "value"),
        State("ana-groupby", "value"),
        State("ana-peak-labels-include", "value"),
        State("ana-peak-labels-exclude", "value"),
        State("ana-normalization-cols", "value"),
        State("ana-file-types", "value"),
        State("pca-options", "value"),
        State("wdir", "children"),
    )
    def create_pca(
        n_clicks,
        n_components,
        facet_height,
        groupby,
        include_labels,
        exclude_labels,
        norm_cols,
        file_types,
        options,
        wdir,
    ):
        if n_clicks is None:
            raise PreventUpdate
        if norm_cols is None:
            norm_cols = []

        df = T.get_complete_results(
            wdir,
            include_labels=include_labels,
            exclude_labels=exclude_labels,
            file_types=file_types,
        )

        if file_types is not None and len(file_types) > 0:
            df = df[df["Type"].isin(file_types)]

        if groupby is not None and len(groupby) > 0:
            color_groups = (
                df[["ms_file", groupby]].drop_duplicates().set_index("ms_file")
            )
        else:
            color_groups = None
            groupby = None

        if len(norm_cols) != 0:
            if ("peak_label" in norm_cols) and ("ms_file" in norm_cols):
                return dbc.Alert(
                    "'peak_label' and 'ms_file' should not be used together for normalization!",
                    color="danger",
                )

            df = df[df.Batch.notna()]
            cols = ["peak_max"]
            df.loc[:, cols] = (
                (
                    df[cols]
                    - df[cols + norm_cols]
                    .groupby(norm_cols)
                    .transform("median")[cols]
                    .values
                )
                / df[cols + norm_cols].groupby(norm_cols).transform("std")[cols].values
            ).reset_index()

        figures = []
        mint = Mint()
        mint.results = df
        mint.pca(n_components=n_components)

        ndx = mint.decomposition_results["df_projected"].index.to_list()

        mint.pca_plot_cumulative_variance()

        src = T.fig_to_src()
        figures.append(html.Img(src=src))

        if color_groups is not None:
            color_groups = color_groups.loc[ndx].values

        with sns.plotting_context("paper"):
            mint.plot_pair_plot(
                group_name=groupby,
                color_groups=color_groups,
                n_vars=n_components,
                height=facet_height,
                corner="Corner" in options,
            )

        src = T.fig_to_src()
        figures.append(html.Img(src=src))

        contrib = mint.decomposition_results["feature_contributions"]

        fig_contrib = px.bar(
            data_frame=contrib,
            x="peak_label",
            y="Coefficient",
            facet_col="PC",
            facet_col_wrap=1,
            height=200 * n_components + 200,
        )

        fig_contrib.update_layout(autosize=True)

        return figures[0], figures[1], fig_contrib
