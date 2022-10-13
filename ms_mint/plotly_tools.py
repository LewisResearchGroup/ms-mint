import numpy as np
import colorlover as cl

import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio

from pathlib import Path as P
from collections.abc import Iterable
from plotly.subplots import make_subplots


def set_template():
    """
    A function that sets a template for plotly figures.
    """
    pio.templates["draft"] = go.layout.Template(
        layout=dict(font={"size": 10}),
    )

    pio.templates.default = "draft"


set_template()


def plotly_heatmap(
    df,
    normed_by_cols=False,
    transposed=False,
    clustered=False,
    add_dendrogram=False,
    name="",
    x_tick_colors=None,
    height=None,
    width=None,
    correlation=False,
    call_show=False,
    verbose=False,
):
    """
    Creates an interactive heatmap from a dense-formated dataframe.

    :param df: Input data
    :type df: pandas.DataFrame
    :param normed_by_cols: Whether or not to normalize column vectors, defaults to False
    :type normed_by_cols: bool, optional
    :param transposed: Whether or not to transpose the generated image, defaults to False
    :type transposed: bool, optional
    :param clustered: Whether or not to apply hierarchical clustering or rows, defaults to False
    :type clustered: bool, optional
    :param add_dendrogram: Whether or not to show a dendrogram (only with `clustered=True`), defaults to False
    :type add_dendrogram: bool, optional
    :param title: Title for figure, defaults to ""
    :type title: str, optional
    :param x_tick_colors: Color of x-ticks, defaults to None
    :type x_tick_colors: str, optional
    :param height: Image height in pixels, defaults to None
    :type height: int, optional
    :param width: Image width in pixels, defaults to None
    :type width: int, optional
    :param correlation: Whether or not to convert the table to a correlation matrix, defaults to False
    :type correlation: bool, optional
    :param call_show: Whether or not to call fig.show() to show image in new browser tab, defaults to False
    :type call_show: bool, optional
    :param verbose: Whether or not to be loud, defaults to False
    :type verbose: bool, optional
    :return: Returns a plotly image object.
    :rtype: plotly.Figure
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
        title = f'{plot_type} of {",".join(plot_attributes)} {name}'

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
        fig["layout"]["yaxis"]["tickvals"] = np.asarray(
            dendro_side["layout"]["yaxis"]["tickvals"]
        )

    fig.update_layout(
        # margin=dict( l=50, r=10, b=200, t=50, pad=0 ),
        autosize=True,
        hovermode="closest",
    )

    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)


    if call_show:
        fig.show(config={"displaylogo": False})
    else:
        return fig



def plotly_peak_shapes(
    mint_results,
    col_wrap=1,
    peak_labels=None,
    legend=True,
    verbose=False,
    legend_orientation="v",
    call_show=False,
):
    """
    Returns a plotly multiplost of all peak_shapes in mint.results
    grouped by peak_label.
    """
    mint_results = mint_results.copy()
    mint_results.ms_file = [P(fn).name for fn in mint_results.ms_file]

    res = mint_results[mint_results.peak_area > 0]

    fns = list(res.ms_file.drop_duplicates())
    labels = list(mint_results.peak_label.drop_duplicates())

    res = res.set_index(["peak_label", "ms_file"])

    if peak_labels is None:
        peak_labels = []

    if isinstance(peak_labels, str):
        peak_labels = [peak_labels]

    # Calculate neccessary number of rows
    n_rows = len(labels) // col_wrap
    if n_rows * col_wrap < len(labels):
        n_rows += 1

    if verbose:
        print(n_rows, col_wrap)
        print("ms_files:", fns)
        print("peak_labels:", peak_labels)
        print("Data:", res)

    fig = make_subplots(rows=max(1, n_rows), cols=max(1, col_wrap), subplot_titles=peak_labels)
    if len(fns) < 13:
        colors = cl.scales["12"]["qual"]["Paired"]
    else:
        colors = cl.interp(cl.scales["12"]["qual"]["Paired"], len(fns))

    # Create sub-plots
    for label_i, label in enumerate(peak_labels):
        for file_i, fn in enumerate(fns):
            #try:
            x, y = res.loc[(label, fn), ["peak_shape_rt", "peak_shape_int"]]
            #except:
            #    continue
            if not isinstance(x, Iterable):
                continue

            if isinstance(x, str):
                x = x.split(",")
                y = y.split(",")

            ndx_r = (label_i // col_wrap) + 1
            ndx_c = label_i % col_wrap + 1

            if len(x) == 1:
                mode = "markers"
            else:
                mode = "lines"

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=P(fn).name,
                    mode=mode,
                    legendgroup=file_i,
                    showlegend=(label_i == 0),
                    marker_color=colors[file_i],
                    text=fn,
                ),
                row=ndx_r,
                col=ndx_c,
            )

            fig.update_xaxes(title_text="Scan Time", row=ndx_r, col=ndx_c)
            fig.update_yaxes(title_text="Intensity", row=ndx_r, col=ndx_c)

    # Layout
    if legend:
        fig.update_layout(legend_orientation=legend_orientation)

    fig.update_layout(showlegend=legend)
    fig.update_layout(height=400 * n_rows, title_text="Peak Shapes")

    if call_show:
        fig.show(config={"displaylogo": False})
    else:
        return fig  
