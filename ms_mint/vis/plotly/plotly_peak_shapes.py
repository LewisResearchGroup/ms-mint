import colorlover as cl

import plotly.graph_objects as go

from collections.abc import Iterable
from os.path import basename
from plotly.subplots import make_subplots


def plotly_peak_shapes(
    mint_results,
    n_cols=1,
    biomarkers=None,
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
    mint_results.ms_file = [basename(i) for i in mint_results.ms_file]

    res = mint_results[mint_results.peak_area > 0]

    files = list(res.ms_file.drop_duplicates())
    labels = list(mint_results.peak_label.drop_duplicates())

    res = res.set_index(["peak_label", "ms_file"])

    if peak_labels is None:
        peak_labels = []
        # peak_labels = mint_results.groupby('peak_label').mean().peak_max.sort_values(ascending=False).index.astype(str)

    if biomarkers is None:
        biomarkers = peak_labels

    if biomarkers is None:
        biomarkers = []
        # biomarkers = mint_results.groupby('peak_label').mean().peak_max.sort_values(ascending=False).index.astype(str)

    if isinstance(biomarkers, str):
        biomarkers = [biomarkers]

    if len(biomarkers) != 0:
        labels = [str(i) for i in biomarkers]

    # Calculate neccessary number of rows
    n_rows = len(labels) // n_cols
    if n_rows * n_cols < len(labels):
        n_rows += 1

    if verbose:
        print(n_rows, n_cols)
        print("ms_files:", files)
        print("peak_labels:", labels)
        print("Data:", res)

    fig = make_subplots(rows=max(1, n_rows), cols=max(1, n_cols), subplot_titles=labels)
    if len(files) < 13:
        colors = cl.scales["12"]["qual"]["Paired"]
    else:
        colors = cl.interp(cl.scales["12"]["qual"]["Paired"], len(files))

    # Create sub-plots
    for label_i, label in enumerate(labels):
        for file_i, file in enumerate(files):
            try:
                x, y = res.loc[(label, file), ["peak_shape_rt", "peak_shape_int"]]
            except:
                continue
            if not isinstance(x, Iterable):
                continue

            if isinstance(x, str):
                x = x.split(",")
                y = y.split(",")

            ndx_r = (label_i // n_cols) + 1
            ndx_c = label_i % n_cols + 1

            if len(x) == 1:
                mode = "markers"
            else:
                mode = "lines"

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=basename(file),
                    mode=mode,
                    legendgroup=file_i,
                    showlegend=(label_i == 0),
                    marker_color=colors[file_i],
                    text=file,
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
