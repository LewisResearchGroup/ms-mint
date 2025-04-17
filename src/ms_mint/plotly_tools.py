#src/ms_mint/plotly_tools.py

import logging
import numpy as np
import pandas as pd
import colorlover as cl
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
from pathlib import Path as P
from collections.abc import Iterable
from plotly.subplots import make_subplots
from typing import Optional, Union, List, Dict, Any, Tuple, Callable, Set
from plotly.graph_objs._figure import Figure as PlotlyFigure

from .tools import fn_to_label


def set_template() -> None:
    """Set a default template for plotly figures.

    Creates a "draft" template with smaller font size and sets it as the default
    template for all plotly figures.
    """
    pio.templates["draft"] = go.layout.Template(
        layout=dict(font={"size": 10}),
    )

    pio.templates.default = "draft"


set_template()


def plotly_heatmap(
    df: pd.DataFrame,
    normed_by_cols: bool = False,
    transposed: bool = False,
    clustered: bool = False,
    add_dendrogram: bool = False,
    name: str = "",
    x_tick_colors: Optional[str] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    correlation: bool = False,
    call_show: bool = False,
    verbose: bool = False,
) -> Optional[PlotlyFigure]:
    """Create an interactive heatmap from a dense-formatted dataframe.

    Args:
        df: Input data in DataFrame format.
        normed_by_cols: Whether to normalize column vectors.
        transposed: Whether to transpose the generated image.
        clustered: Whether to apply hierarchical clustering on rows.
        add_dendrogram: Whether to show a dendrogram (only when clustered=True).
        name: Name to use in figure title.
        x_tick_colors: Color of x-ticks.
        height: Image height in pixels.
        width: Image width in pixels.
        correlation: Whether to convert the table to a correlation matrix.
        call_show: Whether to display the figure immediately.
        verbose: Whether to print additional information.

    Returns:
        A Plotly Figure object, or None if call_show is True.
    """
    max_is_not_zero = df.max(axis=1) != 0
    non_zero_labels = max_is_not_zero[max_is_not_zero].index
    df = df.loc[non_zero_labels]

    colorscale = "Bluered"
    plot_attributes = []

    if normed_by_cols:
        df = df.divide(df.max()).fillna(0)
        plot_attributes.append("normalized")

    if transposed:
        df = df.T

    if correlation:
        plot_type = "Correlation"
        df = df.corr()
        colorscale = [
            [0.0, "rgb(165,0,38)"],
            [0.1111111111111111, "rgb(215,48,39)"],
            [0.2222222222222222, "rgb(244,109,67)"],
            [0.3333333333333333, "rgb(253,174,97)"],
            [0.4444444444444444, "rgb(254,224,144)"],
            [0.5555555555555556, "rgb(224,243,248)"],
            [0.6666666666666666, "rgb(171,217,233)"],
            [0.7777777777777778, "rgb(116,173,209)"],
            [0.8888888888888888, "rgb(69,117,180)"],
            [1.0, "rgb(49,54,149)"],
        ]
    else:
        plot_type = "Heatmap"

    if clustered:
        dendro_side = ff.create_dendrogram(
            df,
            orientation="right",
            labels=df.index.to_list(),
            color_threshold=0,
            colorscale=["black"] * 8,
        )
        dendro_leaves = dendro_side["layout"]["yaxis"]["ticktext"]
        df = df.loc[dendro_leaves, :]
        if correlation:
            df = df[df.index]

    x = df.columns
    if clustered:
        y = dendro_leaves
    else:
        y = df.index.to_list()
    z = df.values

    heatmap = go.Heatmap(x=x, y=y, z=z, colorscale=colorscale)

    if name == "":
        title = ""
    else:
        title = f"{plot_type} of {','.join(plot_attributes)} {name}"

    # Figure without side-dendrogram
    if (not add_dendrogram) or (not clustered):
        fig = go.Figure(heatmap)
        fig.update_layout(
            {"title_x": 0.5},
            title={"text": title},
            yaxis={"title": "", "tickmode": "array", "automargin": True},
        )

        fig.update_layout({"height": height, "width": width, "hovermode": "closest"})

    else:  # Figure with side-dendrogram
        fig = go.Figure()

        for i in range(len(dendro_side["data"])):
            dendro_side["data"][i]["xaxis"] = "x2"

        for data in dendro_side["data"]:
            fig.add_trace(data)

        y_labels = heatmap["y"]
        heatmap["y"] = dendro_side["layout"]["yaxis"]["tickvals"]

        fig.add_trace(heatmap)

        fig.update_layout(
            {
                "height": height,
                "width": width,
                "showlegend": False,
                "hovermode": "closest",
                "paper_bgcolor": "white",
                "plot_bgcolor": "white",
                "title_x": 0.5,
            },
            title={"text": title},
            # X-axis of main figure
            xaxis={
                "domain": [0.11, 1],
                "mirror": False,
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "showticklabels": True,
                "ticks": "",
            },
            # X-axis of side-dendrogram
            xaxis2={
                "domain": [0, 0.1],
                "mirror": False,
                "showgrid": True,
                "showline": False,
                "zeroline": False,
                "showticklabels": False,
                "ticks": "",
            },
            # Y-axis of main figure
            yaxis={
                "domain": [0, 1],
                "mirror": False,
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "showticklabels": False,
            },
        )

        fig["layout"]["yaxis"]["ticktext"] = np.asarray(y_labels)
        fig["layout"]["yaxis"]["tickvals"] = np.asarray(dendro_side["layout"]["yaxis"]["tickvals"])

    fig.update_layout(
        autosize=True,
        hovermode="closest",
    )

    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)

    if call_show:
        fig.show(config={"displaylogo": False})
        return None
    else:
        return fig


def plotly_peak_shapes(
    mint_results: pd.DataFrame,
    mint_metadata: Optional[pd.DataFrame] = None,
    color: str = "ms_file_label",
    fns: Optional[List[str]] = None,
    col_wrap: int = 1,
    peak_labels: Optional[Union[str, List[str]]] = None,
    legend: bool = True,
    verbose: bool = False,
    legend_orientation: str = "v",
    call_show: bool = False,
    palette: str = "Plasma",
) -> Optional[PlotlyFigure]:
    """Plot peak shapes from mint results as interactive Plotly figure.

    Args:
        mint_results: DataFrame in Mint results format.
        mint_metadata: DataFrame in Mint metadata format.
        color: Column name determining color-coding of plots.
        fns: Filenames to include. If None, all files are used.
        col_wrap: Maximum number of subplot columns.
        peak_labels: Peak-labels to include. If None, all peaks are used.
        legend: Whether to display legend.
        verbose: If True, prints additional details.
        legend_orientation: Legend orientation ('v' for vertical, 'h' for horizontal).
        call_show: If True, displays the plot immediately.
        palette: Color palette to use.

    Returns:
        A Plotly Figure object, or None if call_show is True.
    """
    mint_results = mint_results.copy()

    # Merge with metadata if provided
    if mint_metadata is not None:
        mint_results = pd.merge(
            mint_results, mint_metadata, left_on="ms_file_label", right_index=True
        )

    # Filter by filenames
    if fns is not None:
        fns = [fn_to_label(fn) for fn in fns]
        mint_results = mint_results[mint_results.ms_file_label.isin(fns)]
    else:
        fns = mint_results.ms_file_label.unique()

    # Filter by peak_labels
    if peak_labels is not None:
        if isinstance(peak_labels, str):
            peak_labels = [peak_labels]
        mint_results = mint_results[mint_results.peak_label.isin(peak_labels)]
    else:
        peak_labels = mint_results.peak_label.unique()

    # Handle colors based on metadata or fall back to default behavior
    colors = None
    if color:
        unique_hues = mint_results[color].unique()

        colors = get_palette_colors(palette, len(unique_hues))

        color_mapping = dict(zip(unique_hues, colors))

        if color == "ms_file_label":
            hue_column = [color_mapping[fn] for fn in fns]
        else:
            # Existing logic remains the same for the else part
            hue_column = (
                mint_results.drop_duplicates("ms_file_label")
                .set_index("ms_file_label")[color]
                .map(color_mapping)
                .reindex(fns)
                .tolist()
            )

    else:
        hue_column = colors

    # Rest of the plotting process
    res = mint_results[mint_results.peak_max > 0]
    labels = mint_results.peak_label.unique()
    res = res.set_index(["peak_label", "ms_file_label"]).sort_index()

    # Calculate necessary number of rows
    n_rows = max(1, len(labels) // col_wrap)
    if n_rows * col_wrap < len(labels):
        n_rows += 1

    fig = make_subplots(rows=max(1, n_rows), cols=max(1, col_wrap), subplot_titles=peak_labels)

    for label_i, label in enumerate(peak_labels):
        for file_i, fn in enumerate(fns):
            try:
                x, y = res.loc[(label, fn), ["peak_shape_rt", "peak_shape_int"]]
            except KeyError as e:
                logging.warning(e)
                continue

            if not isinstance(x, Iterable):
                continue
            if isinstance(x, str):
                x = x.split(",")
                y = y.split(",")

            ndx_r = (label_i // col_wrap) + 1
            ndx_c = label_i % col_wrap + 1

            trace_color = hue_column[file_i]

            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    name=P(fn).name,
                    mode="markers",
                    legendgroup=file_i,
                    showlegend=(label_i == 0),
                    marker_color=trace_color,
                    text=fn,
                    fill="tozeroy",
                    marker=dict(size=3),
                ),
                row=ndx_r,
                col=ndx_c,
            )

            fig.update_xaxes(title_text="Scan time [s]", row=ndx_r, col=ndx_c)
            fig.update_yaxes(title_text="Intensity", row=ndx_r, col=ndx_c)

    # Layout updates
    if legend:
        fig.update_layout(legend_orientation=legend_orientation)

    fig.update_layout(showlegend=legend)
    fig.update_layout(height=400 * n_rows, title_text="Peak Shapes")

    if call_show:
        fig.show(config={"displaylogo": False})
        return None
    else:
        return fig


def get_palette_colors(palette_name: str, num_colors: int) -> List[str]:
    """Get a list of colors from a specific colorlover palette.

    Args:
        palette_name: Name of the color palette.
        num_colors: Number of colors to extract.

    Returns:
        List of color strings in the requested palette.
    """
    # Categories in the colorlover package
    categories = ["qual", "seq", "div"]

    num_colors = max(num_colors, 3)
    # Check in which category our palette resides
    for category in categories:
        if palette_name in cl.scales[f"{num_colors}"][category]:
            return cl.scales[f"{num_colors}"][category][palette_name]

    # If palette not found in any category, return a default one
    return cl.scales[f"{num_colors}"]["qual"]["Paired"]
