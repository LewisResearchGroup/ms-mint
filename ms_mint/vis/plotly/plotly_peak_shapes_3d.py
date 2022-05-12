import pandas as pd
import plotly.express as px

from os.path import basename


def plotly_peak_shapes_3d(
    mint_results,
    peak_label=None,
    legend=True,
    legend_orientation="v",
    call_show=False,
    verbose=False,
):
    """
    Returns a plotly 3D plot of all peak_shapes in mint.results
    where mint.results.peak_label == peak_label.
    """

    mint_results = mint_results.copy()
    mint_results.ms_file = [basename(i) for i in mint_results.ms_file]

    data = mint_results[mint_results.peak_label == peak_label]
    files = list(data.ms_file.drop_duplicates())

    grps = data.groupby("ms_file")

    # Peak labels are supposed to be strings
    # Sometimes they are converted to int though

    samples = []
    for i, fn in enumerate(files):
        grp = grps.get_group(fn)
        try:
            x, y, peak_max = grp[
                ["peak_shape_rt", "peak_shape_int", "peak_max"]
            ].values[0]
        except:
            continue

        if isinstance(x, str):
            x = x.split(",")
            y = y.split(",")
        sample = pd.DataFrame({"Scan Time": x, "Intensity": y})
        sample["peak_max"] = peak_max
        sample["ms_file"] = basename(fn)
        samples.append(sample)

    if len(samples) == 0:
        return None

    samples = pd.concat(samples)

    fig = px.line_3d(
        samples, x="Scan Time", y="peak_max", z="Intensity", color="ms_file"
    )
    fig.update_layout({"height": 1000, "width": 1000})

    # Layout
    if legend:
        fig.update_layout(legend_orientation=legend_orientation)

    fig.update_layout(showlegend=legend)

    fig.update_layout({"title": peak_label, "title_x": 0.5})

    if call_show:
        fig.show(config={"displaylogo": False})
    else:
        return fig
