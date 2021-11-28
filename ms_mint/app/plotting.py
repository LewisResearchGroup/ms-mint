from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from matplotlib import pyplot as plt

import seaborn as sns

plt.rcParams["figure.autolayout"] = False
sns.set_context("paper")


from . import tools as T


_label = "Plotting"

_kinds = [
    "bar",
    "violin",
    "box",
    "count",
    "boxen",
    "scatter",
    "line",
    "strip",
    "swarm",
    "point",
]

_palettes = [
    "Accent",
    "Accent_r",
    "Blues",
    "Blues_r",
    "BrBG",
    "BrBG_r",
    "BuGn",
    "BuGn_r",
    "BuPu",
    "BuPu_r",
    "CMRmap",
    "CMRmap_r",
    "Dark2",
    "Dark2_r",
    "GnBu",
    "GnBu_r",
    "Greens",
    "Greens_r",
    "Greys",
    "Greys_r",
    "OrRd",
    "OrRd_r",
    "Oranges",
    "Oranges_r",
    "PRGn",
    "PRGn_r",
    "Paired",
    "Paired_r",
    "Pastel1",
    "Pastel1_r",
    "Pastel2",
    "Pastel2_r",
    "PiYG",
    "PiYG_r",
    "PuBu",
    "PuBuGn",
    "PuBuGn_r",
    "PuBu_r",
    "PuOr",
    "PuOr_r",
    "PuRd",
    "PuRd_r",
    "Purples",
    "Purples_r",
    "RdBu",
    "RdBu_r",
    "RdGy",
    "RdGy_r",
    "RdPu",
    "RdPu_r",
    "RdYlBu",
    "RdYlBu_r",
    "RdYlGn",
    "RdYlGn_r",
    "Reds",
    "Reds_r",
    "Set1",
    "Set1_r",
    "Set2",
    "Set2_r",
    "Set3",
    "Set3_r",
    "Spectral",
    "Spectral_r",
    "Wistia",
    "Wistia_r",
    "YlGn",
    "YlGnBu",
    "YlGnBu_r",
    "YlGn_r",
    "YlOrBr",
    "YlOrBr_r",
    "YlOrRd",
    "YlOrRd_r",
    "afmhot",
    "afmhot_r",
    "autumn",
    "autumn_r",
    "binary",
    "binary_r",
    "bone",
    "bone_r",
    "brg",
    "brg_r",
    "bwr",
    "bwr_r",
    "cividis",
    "cividis_r",
    "cool",
    "cool_r",
    "coolwarm",
    "coolwarm_r",
    "copper",
    "copper_r",
    "cubehelix",
    "cubehelix_r",
    "flag",
    "flag_r",
    "gist_earth",
    "gist_earth_r",
    "gist_gray",
    "gist_gray_r",
    "gist_heat",
    "gist_heat_r",
    "gist_ncar",
    "gist_ncar_r",
    "gist_rainbow",
    "gist_rainbow_r",
    "gist_stern",
    "gist_stern_r",
    "gist_yarg",
    "gist_yarg_r",
    "gnuplot",
    "gnuplot2",
    "gnuplot2_r",
    "gnuplot_r",
    "gray",
    "gray_r",
    "hot",
    "hot_r",
    "hsv",
    "hsv_r",
    "icefire",
    "icefire_r",
    "inferno",
    "inferno_r",
    "jet",
    "jet_r",
    "magma",
    "magma_r",
    "mako",
    "mako_r",
    "nipy_spectral",
    "nipy_spectral_r",
    "ocean",
    "ocean_r",
    "pink",
    "pink_r",
    "plasma",
    "plasma_r",
    "prism",
    "prism_r",
    "rainbow",
    "rainbow_r",
    "rocket",
    "rocket_r",
    "seismic",
    "seismic_r",
    "spring",
    "spring_r",
    "summer",
    "summer_r",
    "tab10",
    "tab10_r",
    "tab20",
    "tab20_r",
    "tab20b",
    "tab20b_r",
    "tab20c",
    "tab20c_r",
    "terrain",
    "terrain_r",
    "turbo",
    "turbo_r",
    "twilight",
    "twilight_r",
    "twilight_shifted",
    "twilight_shifted_r",
    "viridis",
    "viridis_r",
    "vlag",
    "vlag_r",
    "winter",
    "winter_r",
]

kind_options = [{"value": x, "label": x.capitalize()} for x in _kinds]

palette_options = [{"value": x, "label": x} for x in _palettes]

options = [
    {"label": "Rotate x-ticks", "value": "rot-x-ticks"},
    {"label": "Share x-axis", "value": "share-x"},
    {"label": "Share y-axes", "value": "share-y"},
    {"label": "Scientific notation", "value": "sci"},
    {"label": "Logarithmic x-scale", "value": "log-x"},
    {"label": "Logarithmic y-scale", "value": "log-y"},
    {"label": "High Quality", "value": "HQ"},
    {"label": "Don't dodge", "value": "no-dodge"},
]

_layout = html.Div(
    [
        html.H3(_label),
        dcc.Input(
            id="plot-fig-height",
            placeholder="Figure facet height x",
            value=2.5,
            type="number",
        ),
        dcc.Input(
            id="plot-fig-aspect",
            placeholder="Figure aspect ratio",
            value=1,
            type="number",
        ),
        dcc.Input(id="plot-title", placeholder="Title", value=None),
        dcc.Dropdown(id="plot-kind", options=kind_options, value="bar"),
        dcc.Dropdown(id="plot-x", options=[], value=None, placeholder="X"),
        dcc.Dropdown(id="plot-y", options=[], value="peak_max", placeholder="Y"),
        dcc.Dropdown(id="plot-hue", options=[], value=None, placeholder="Color"),
        dcc.Dropdown(id="plot-col", options=[], value=None, placeholder="Columns"),
        dcc.Dropdown(id="plot-row", options=[], value=None, placeholder="Rows"),
        dcc.Dropdown(id="plot-style", options=[], value=None, placeholder="Style"),
        dcc.Dropdown(id="plot-size", options=[], value=None, placeholder="Size"),
        dcc.Dropdown(
            id="plot-palette",
            options=palette_options,
            value=None,
            placeholder="Palette (Colors)",
        ),
        html.Label("Column wrap:"),
        dcc.Slider(id="plot-col-wrap", step=1, min=0, max=30, value=0),
        dcc.Dropdown(id="plot-options", value=["sci"], options=options, multi=True),
        html.Button("Update", id="plot-update"),
        dcc.Loading(
            html.Div(
                id="plot-figures",
                style={
                    "margin": "auto",
                    "marginTop": "10%",
                    "text-align": "center",
                    "maxWidth": "100%",
                    "minHeight": "300px",
                },
            )
        ),
    ]
)

_ouptuts = html.Div([])


def layout():
    return _layout


def callbacks(app, fsc, cache):
    @app.callback(
        Output("plot-x", "options"),
        Output("plot-y", "options"),
        Output("plot-col", "options"),
        Output("plot-row", "options"),
        Output("plot-hue", "options"),
        Output("plot-size", "options"),
        Output("plot-style", "options"),
        Input("ana-secondary-tab", "value"),
        State("wdir", "children"),
    )
    def fill_options(tab, wdir):
        if tab != _label:
            raise PreventUpdate
        results = T.get_complete_results(wdir)
        results = results.dropna(axis=1, how="all")
        cols = results.columns
        options = [{"value": x, "label": x} for x in cols]
        return [options] * 7

    @app.callback(
        Output("plot-figures", "children"),
        Input("plot-update", "n_clicks"),
        State("plot-kind", "value"),
        State("plot-fig-height", "value"),
        State("plot-fig-aspect", "value"),
        State("plot-x", "value"),
        State("plot-y", "value"),
        State("plot-hue", "value"),
        State("plot-col", "value"),
        State("plot-row", "value"),
        State("plot-col-wrap", "value"),
        State("plot-style", "value"),
        State("plot-size", "value"),
        State("plot-title", "value"),
        State("ana-file-types", "value"),
        State("ana-peak-labels-include", "value"),
        State("ana-peak-labels-exclude", "value"),
        State("ana-ms-order", "value"),
        State("plot-palette", "value"),
        State("plot-options", "value"),
        State("wdir", "children"),
    )
    def create_figure(
        n_clicks,
        kind,
        height,
        aspect,
        x,
        y,
        hue,
        col,
        row,
        col_wrap,
        style,
        size,
        title,
        file_types,
        include_labels,
        exclude_labels,
        ms_order,
        palette,
        options,
        wdir,
    ):

        if n_clicks is None:
            raise PreventUpdate
        if col_wrap == 0:
            col_wrap = None
        if col is None and row is None:
            col_wrap = None
        if height is None:
            height = 2.5
        if aspect is None:
            aspect = 1

        # With hue both x and y have to be set.
        if (hue is not None) and (None in [x, y]):
            if x is None:
                x = hue
            if y is None:
                y = hue

        height = min(float(height), 100)
        height = max(height, 1)
        aspect = max(0.01, float(aspect))
        aspect = min(aspect, 100)

        df = T.get_complete_results(
            wdir,
            include_labels=include_labels,
            exclude_labels=exclude_labels,
            file_types=file_types,
        )

        df = df[(df.peak_n_datapoints > 0) & (df.peak_max > 0)]

        if ms_order is None or len(ms_order) == 0:
            df = df.sort_values(
                [i for i in [col, row, hue, x, y, style, size] if i is not None]
            )
        else:
            df = df.sort_values(ms_order)

        if hue is not None:
            df = df[df[hue].notna()]

        n_c, n_r = 1, 1
        if col is not None:
            n_c = len(df[col].drop_duplicates())
        if row is not None:
            n_r = len(df[row].drop_duplicates())

        if (n_c is not None) and (col_wrap is not None):
            n_c = n_c // col_wrap
            n_r = n_c % col_wrap
        del n_r

        if hue is not None and ((x is None) and (y is None)):
            x = hue

        if kind in ["scatter", "line"]:
            plot_func = sns.relplot
            kwargs = dict(
                facet_kws=dict(
                    sharex="share-x" in options,
                    sharey="share-y" in options,
                    legend_out=True,
                ),
                style=style,
                size=size,
            )
        else:
            plot_func = sns.catplot
            kwargs = dict(
                sharex="share-x" in options,
                sharey="share-y" in options,
                dodge="no-dodge" not in options,
                facet_kws=dict(legend_out=True),
            )

        try:
            g = plot_func(
                x=x,
                y=y,
                hue=hue,
                col=col,
                row=row,
                col_wrap=col_wrap,
                data=df,
                kind=kind,
                height=height,
                aspect=aspect,
                palette=palette,
                **kwargs
            )
        except Exception as e:
            return dbc.Alert(str(e), color="danger")

        g.fig.subplots_adjust(top=0.9)
        g.set_titles(col_template="{col_name}", row_template="{row_name}", y=1.05)

        if "log-x" in options:
            g.set(xscale="log")
        if "log-y" in options:
            g.set(yscale="log")

        notation = "sci" if "sci" in options else "plain"
        for ax in g.axes.flatten():
            try:
                ax.ticklabel_format(style=notation, scilimits=(0, 0), axis="x")
            except:
                pass
            try:
                ax.ticklabel_format(style=notation, scilimits=(0, 0), axis="y")
            except:
                pass

        if "rot-x-ticks" in options:
            g.set_xticklabels(rotation=90)

        if title is not None:
            g.fig.suptitle(title, y=1.01)

        g.tight_layout(w_pad=0)

        # try:
        #    g.legend.set_bbox_to_anchor((1.2, 0.7))
        # except:
        #    pass

        src = T.fig_to_src(dpi=300 if "HQ" in options else None)

        return html.Img(src=src, style={"maxWidth": "80%"})
