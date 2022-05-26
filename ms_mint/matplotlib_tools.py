import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform



def hierarchical_clustering(
    df,
    vmin=None,
    vmax=None,
    figsize=(8, 8),
    top_height=2,
    left_width=2,
    xmaxticks=None,
    ymaxticks=None,
    metric="euclidean",
    cmap=None,
):
    """_summary_

    :param df: _description_
    :type df: _type_
    :param vmin: _description_, defaults to None
    :type vmin: _type_, optional
    :param vmax: _description_, defaults to None
    :type vmax: _type_, optional
    :param figsize: _description_, defaults to (8, 8)
    :type figsize: tuple, optional
    :param top_height: _description_, defaults to 2
    :type top_height: int, optional
    :param left_width: _description_, defaults to 2
    :type left_width: int, optional
    :param xmaxticks: _description_, defaults to None
    :type xmaxticks: _type_, optional
    :param ymaxticks: _description_, defaults to None
    :type ymaxticks: _type_, optional
    :param metric: _description_, defaults to "euclidean"
    :type metric: str, optional
    :param cmap: _description_, defaults to None
    :type cmap: _type_, optional
    :return: _description_
    :rtype: _type_
    """

    if isinstance(metric, str):
        metric_x, metric_y = metric, metric
    elif len(metric) == 2 and isinstance(metric[0], str) and isinstance(metric[1], str):
        metric_x, metric_y = metric
    elif metric is None:
        metric_x, metric_y = None, None

    df = df.copy()

    # Subplot sizes
    total_width, total_height = figsize

    main_h = 1 - (top_height / total_height)
    main_w = 1 - (left_width / total_width)

    gap_x = 0.1 / total_width
    gap_y = 0.1 / total_height

    left_h = main_h
    left_w = 1 - main_w

    top_h = 1 - main_h
    top_w = main_w

    if xmaxticks is None:
        xmaxticks = int(5 * main_w * total_width)
    if ymaxticks is None:
        ymaxticks = int(5 * main_h * total_height)

    dm = df.fillna(0).values
    D1 = squareform(pdist(dm, metric=metric_y))
    D2 = squareform(pdist(dm.T, metric=metric_x))

    fig = plt.figure(figsize=figsize)
    fig.set_tight_layout(False)

    # add left dendrogram
    ax1 = fig.add_axes([0, 0, left_w - gap_x, left_h], frameon=False)
    Y = linkage(D1, method="complete")
    Z1 = dendrogram(Y, orientation="left", color_threshold=0, above_threshold_color="k")
    ax1.set_xticks([])
    ax1.set_yticks([])
    # add top dendrogram
    ax2 = fig.add_axes([left_w, main_h + gap_y, top_w, top_h - gap_y], frameon=False)
    Y = linkage(D2, method="complete")
    Z2 = dendrogram(Y, color_threshold=0, above_threshold_color="k")
    ax2.set_xticks([])
    ax2.set_yticks([])
    # add matrix plot
    axmatrix = fig.add_axes([left_w, 0, main_w, main_h])
    idx1 = Z1["leaves"]
    idx2 = Z2["leaves"]
    D = dm[idx1, :]
    D = D[:, idx2]

    if cmap is None:
        cmap = "coolwarm"
    fig = axmatrix.matshow(D[::-1], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.xaxis.tick_bottom()

    clustered = df.iloc[Z1["leaves"][::-1], Z2["leaves"]]

    ndx_y = np.linspace(0, len(clustered.index) - 1, ymaxticks)
    ndx_x = np.linspace(0, len(clustered.columns) - 1, xmaxticks)
    ndx_y = [int(i) for i in ndx_y]
    ndx_x = [int(i) for i in ndx_x]

    _ = plt.yticks(ndx_y, clustered.iloc[ndx_y].index)
    _ = plt.xticks(ndx_x, clustered.columns[ndx_x], rotation=90)

    ndx_leaves = Z1["leaves"][::-1]
    col_leaves = Z2["leaves"]

    return clustered, fig, ndx_leaves, col_leaves


def plot_peak_shapes(
    mint_results,
    ms_files=None,
    peak_labels=None,
    height=4,
    aspect=1,
    legend=False,
    n_cols=None,
    col_wrap=4,
    hue="ms_file",
    top=None,
    title=None,
    dpi=None,
    sharex=False,
    sharey=False,
    **kwargs
):
    """_summary_

    :param mint_results: _description_
    :type mint_results: _type_
    :param ms_files: _description_, defaults to None
    :type ms_files: _type_, optional
    :param peak_labels: _description_, defaults to None
    :type peak_labels: _type_, optional
    :param height: _description_, defaults to 4
    :type height: int, optional
    :param aspect: _description_, defaults to 1
    :type aspect: int, optional
    :param legend: _description_, defaults to False
    :type legend: bool, optional
    :param n_cols: _description_, defaults to None
    :type n_cols: _type_, optional
    :param col_wrap: _description_, defaults to 4
    :type col_wrap: int, optional
    :param hue: _description_, defaults to "ms_file"
    :type hue: str, optional
    :param top: _description_, defaults to None
    :type top: _type_, optional
    :param title: _description_, defaults to None
    :type title: _type_, optional
    :param dpi: _description_, defaults to None
    :type dpi: _type_, optional
    :param sharex: _description_, defaults to False
    :type sharex: bool, optional
    :param sharey: _description_, defaults to False
    :type sharey: bool, optional
    :return: _description_
    :rtype: _type_
    """

    # fig = plt.figure(dpi=dpi)

    R = mint_results.copy()

    if peak_labels is not None:
        if isinstance(peak_labels, str):
            peak_labels = [peak_labels]
        R = R[R.peak_label.isin(peak_labels)]
    else:
        peak_labels = R.peak_label.drop_duplicates().values

    if ms_files is not None:
        R = R[R.ms_file.isin(ms_files)]

    dfs = []
    for peak_label in peak_labels:
        for ndx, row in R[
            (R.peak_label == peak_label) & (R.peak_n_datapoints > 1)
        ].iterrows():
            peak_rt = [float(i) for i in row.peak_shape_rt.split(",")]
            peak_int = [float(i) for i in row.peak_shape_int.split(",")]
            ms_file = row.ms_file
            df = pd.DataFrame(
                {
                    "Retention Time [min]": peak_rt,
                    "MS-Intensity": peak_int,
                    "ms_file": ms_file,
                    "peak_label": peak_label,
                }
            )
            dfs.append(df)

    df = pd.concat(dfs)

    if n_cols is not None:
        col_wrap = n_cols

    g = sns.relplot(
        data=df,
        x="Retention Time [min]",
        y="MS-Intensity",
        hue=hue,
        col="peak_label",
        kind="line",
        col_wrap=col_wrap,
        height=height,
        aspect=aspect,
        facet_kws=dict(sharex=sharex, sharey=sharey),
        legend=legend,
        **kwargs
    )

    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    for ax in g.axes.flatten():
        ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")

    if title is not None:
        g.fig.suptitle(title, y=1.01)

    return g
