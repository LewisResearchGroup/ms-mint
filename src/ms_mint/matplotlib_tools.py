#src/ms_mint/matplotlib_tools.py

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from typing import Union, List, Tuple, Optional, Dict, Any, Literal, Set
from matplotlib.figure import Figure
import seaborn.objects as so


def hierarchical_clustering(
    df: pd.DataFrame,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (8, 8),
    top_height: int = 2,
    left_width: int = 2,
    xmaxticks: Optional[int] = None,
    ymaxticks: Optional[int] = None,
    metric: Union[str, Tuple[str, str]] = "cosine",
    cmap: Optional[str] = None,
) -> Tuple[pd.DataFrame, Figure, List[int], List[int]]:
    """Perform and plot hierarchical clustering on a dataframe.

    Args:
        df: Input data in DataFrame format.
        vmin: Minimum value to anchor the colormap. If None, inferred from data.
        vmax: Maximum value to anchor the colormap. If None, inferred from data.
        figsize: Size of the main figure in inches.
        top_height: Height of the top dendrogram.
        left_width: Width of the left dendrogram.
        xmaxticks: Maximum number of x-ticks to display.
        ymaxticks: Maximum number of y-ticks to display.
        metric: Distance metric to use. Either a string to use the same metric for
            both axes, or a tuple of two strings for different metrics for each axis.
        cmap: Matplotlib colormap name. If None, uses "coolwarm".

    Returns:
        A tuple containing:
            - The clustered DataFrame (reordered according to clustering)
            - The matplotlib Figure object
            - The indices of rows in their clustered order
            - The indices of columns in their clustered order
    """
    if isinstance(metric, str):
        metric_x, metric_y = metric, metric
    elif (
        isinstance(metric, tuple)
        and len(metric) == 2
        and isinstance(metric[0], str)
        and isinstance(metric[1], str)
    ):
        metric_x, metric_y = metric
    elif metric is None:
        metric_x, metric_y = None, None
    else:
        raise ValueError("Metric must be a string or a tuple of two strings")

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
    fig.set_layout_engine('tight')

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
    im = axmatrix.matshow(D[::-1], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

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
    mint_results: pd.DataFrame,
    mint_metadata: Optional[pd.DataFrame] = None,
    fns: Optional[List[str]] = None,
    peak_labels: Optional[Union[str, List[str]]] = None,
    height: int = 3,
    aspect: float = 1.5,
    legend: bool = False,
    col_wrap: int = 4,
    hue: str = "ms_file_label",
    title: Optional[str] = None,
    dpi: Optional[int] = None,
    sharex: bool = False,
    sharey: bool = False,
    kind: str = "line",
    **kwargs,
) -> sns.FacetGrid:
    """Plot peak shapes from MS-MINT results.

    Args:
        mint_results: DataFrame in Mint results format.
        mint_metadata: DataFrame in Mint metadata format for additional sample information.
        fns: Filenames to include. If None, includes all files.
        peak_labels: Peak label(s) to include. If None, includes all peak labels.
        height: Height of each figure facet in inches.
        aspect: Aspect ratio (width/height) of each figure facet.
        legend: Whether to display a legend.
        col_wrap: Number of columns for subplots.
        hue: Column name to use for color grouping.
        title: Title to add to the figure.
        dpi: Resolution of generated image.
        sharex: Whether to share x-axis range between subplots.
        sharey: Whether to share y-axis range between subplots.
        kind: Type of seaborn relplot ('line', 'scatter', etc.).
        **kwargs: Additional keyword arguments passed to seaborn's relplot.

    Returns:
        A seaborn FacetGrid object containing the plot.
    """
    R = mint_results.copy()
    R = R[R.peak_area > 0]
    R["peak_label"] = R["peak_label"]

    if peak_labels is not None:
        if isinstance(peak_labels, str):
            peak_labels = [peak_labels]
        R = R[R.peak_label.isin(peak_labels)]
    else:
        peak_labels = R.peak_label.drop_duplicates().values

    if fns is not None:
        R = R[R.ms_file.isin(fns)]

    dfs = []
    for peak_label in peak_labels:
        for _, row in R[(R.peak_label == peak_label) & (R.peak_n_datapoints > 1)].iterrows():
            peak_rt = [float(i) for i in row.peak_shape_rt.split(",")]
            peak_int = [float(i) for i in row.peak_shape_int.split(",")]
            ms_file_label = row.ms_file_label
            mz = row.mz_mean
            rt = row.rt

            df = pd.DataFrame(
                {
                    "Scan time [s]": peak_rt,
                    "Intensity": peak_int,
                    "ms_file_label": ms_file_label,
                    "peak_label": peak_label,
                    "Expected Scan time [s]": rt,
                }
            )
            dfs.append(df)

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)

    # Add metadata
    if mint_metadata is not None:
        df = pd.merge(df, mint_metadata, left_on="ms_file_label", right_index=True, how="left")

    _facet_kws = dict(sharex=sharex, sharey=sharey)
    if "facet_kws" in kwargs.keys():
        _facet_kws.update(kwargs.pop("facet_kws"))

    g = sns.relplot(
        data=df,
        x="Scan time [s]",
        y="Intensity",
        hue=hue,
        col="peak_label",
        col_order=peak_labels,
        kind=kind,
        col_wrap=col_wrap,
        height=height,
        aspect=aspect,
        facet_kws=_facet_kws,
        legend=legend,
        **kwargs,
    )

    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    for ax in g.axes.flatten():
        ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")

    if title is not None:
        g.fig.suptitle(title, y=1.01)

    return g


def plot_peaks(
    series: pd.Series,
    peaks: Optional[pd.DataFrame] = None,
    highlight: Optional[List[int]] = None,
    expected_rt: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
    legend: bool = True,
    label: Optional[str] = None,
    **kwargs,
) -> Figure:
    """Plot time series data with peak annotations.

    Args:
        series: Time series data with time as index and intensity as values.
        peaks: DataFrame containing peak information.
        highlight: List of peak indices to highlight.
        expected_rt: Expected retention time to mark on the plot.
        weights: Array of weight values (e.g., for Gaussian weighting).
        legend: Whether to display the legend.
        label: Label for the time series data.
        **kwargs: Additional keyword arguments passed to the plot function.

    Returns:
        Matplotlib Figure containing the plot.
    """
    if highlight is None:
        highlight = []
    ax = plt.gca()
    ax.plot(
        series.index,
        series.values,
        label=label if label is not None else "Intensity",
        **kwargs,
    )
    if peaks is not None:
        series.iloc[peaks.ndxs].plot(label="Peaks", marker="x", y="intensity", lw=0, ax=ax)
        for i, (
            ndx,
            (_, _, _, peak_base_height, _, rt_min, rt_max),
        ) in enumerate(peaks.iterrows()):
            if ndx in highlight:
                plt.axvspan(rt_min, rt_max, color="green", alpha=0.25, label="Selected")
            plt.hlines(
                peak_base_height,
                rt_min,
                rt_max,
                color="orange",
                label="Peak width" if i == 0 else None,
            )
    if expected_rt is not None:
        plt.axvspan(expected_rt, expected_rt + 1, color="blue", alpha=1, label="Expected Rt")
    if weights is not None:
        plt.plot(weights, linestyle="--", label="Gaussian weight")
    plt.ylabel("Intensity")
    plt.xlabel("Scan time [s]")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if not legend:
        ax.get_legend().remove()
    return plt.gcf()


def plot_metabolomics_hist2d(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (4, 2.5),
    dpi: int = 300,
    set_dim: bool = True,
    cmap: str = "jet",
    rt_range: Optional[Tuple[float, float]] = None,
    mz_range: Optional[Tuple[float, float]] = None,
    mz_bins: int = 100,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Create a 2D histogram of metabolomics data.

    Args:
        df: DataFrame containing metabolomics data with scan_time, mz, and intensity columns.
        figsize: Size of the figure in inches (width, height).
        dpi: Resolution of the figure in dots per inch.
        set_dim: Whether to set figure dimensions.
        cmap: Colormap name to use for the plot.
        rt_range: Retention time range (min, max) to display. If None, uses data range.
        mz_range: M/Z range (min, max) to display. If None, uses data range.
        mz_bins: Number of bins to use for the m/z axis.
        **kwargs: Additional keyword arguments passed to plt.hist2d.

    Returns:
        The result of plt.hist2d, which is a tuple containing:
            - The histogram array
            - The edges of the bins along the x-axis
            - The edges of the bins along the y-axis
            - The Axes object
    """
    if set_dim:
        plt.figure(figsize=figsize, dpi=dpi)

    if mz_range is None:
        mz_range = (df.mz.min(), df.mz.max())

    if rt_range is None:
        rt_range = (df.scan_time.min(), df.scan_time.max())

    rt_bins = int((rt_range[1] - rt_range[0]) / 2)

    params = dict(vmin=1, vmax=1e3, cmap=cmap, range=(rt_range, mz_range))
    params.update(kwargs)

    fig = plt.hist2d(
        df["scan_time"],
        df["mz"],
        weights=df["intensity"].apply(np.log1p),
        bins=[rt_bins, mz_bins],
        **params,
    )

    plt.xlabel("Scan time [s]")
    plt.ylabel("m/z")
    plt.gca().ticklabel_format(useOffset=False, style="plain")
    return fig
