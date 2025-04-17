# src/ms_mint/MintPlotter.py
from __future__ import annotations

import numpy as np
import seaborn as sns
import warnings
from scipy.cluster.hierarchy import ClusterWarning
from pathlib import Path as P
import matplotlib
from matplotlib import pyplot as plt
from typing import Optional, List, Tuple, Union, Dict, Any, Callable
import pandas as pd
import plotly.express as px
from plotly.graph_objs._figure import Figure as PlotlyFigure

from .plotly_tools import plotly_heatmap, plotly_peak_shapes
from .matplotlib_tools import (
    plot_peak_shapes,
    hierarchical_clustering,
    plot_metabolomics_hist2d,
)
from .tools import scale_dataframe, mz_mean_width_to_min_max
from .io import ms_file_to_df


class MintPlotter:
    """Plot generator for visualizing MS-MINT analysis results.

    This class provides various visualization methods for metabolomics data processed
    by MS-MINT, including heatmaps, chromatograms, peak shapes, and 2D histograms.

    Attributes:
        mint: The Mint instance containing data to be visualized.
    """

    def __init__(self, mint: "ms_mint.Mint.Mint") -> None:
        """Initialize the MintPlotter with a Mint instance.

        Args:
            mint: Mint instance containing the data to visualize.
        """
        self.mint = mint

    def hierarchical_clustering(
        self,
        data: Optional[pd.DataFrame] = None,
        peak_labels: Optional[List[str]] = None,
        ms_files: Optional[List[str]] = None,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 8),
        targets_var: Optional[str] = None,
        var_name: str = "peak_max",
        vmin: int = -3,
        vmax: int = 3,
        xmaxticks: Optional[int] = None,
        ymaxticks: Optional[int] = None,
        apply: str = "log2p1",
        metric: str = "cosine",
        scaler: str = "standard",
        groupby: Optional[str] = None,
        transposed: bool = False,
        **kwargs,
    ) -> matplotlib.figure.Figure:
        """Perform hierarchical clustering and plot a heatmap.

        If no data is provided, data is taken from self.mint.crosstab(var_name).
        The clustered non-transformed non-scaled data is stored in `self.mint.clustered`.

        Args:
            data: DataFrame with data to be used for clustering. If None, crosstab of
                mint instance is used.
            peak_labels: List of peak labels to include in the analysis.
            ms_files: List of MS files to include in the analysis.
            title: Title for the plot.
            figsize: Tuple of (width, height) in inches for the figure.
            targets_var: Deprecated, use var_name instead.
            var_name: Name of the column from data to be used for cell values in the heatmap.
            vmin: Minimum value for color scaling.
            vmax: Maximum value for color scaling.
            xmaxticks: Maximum number of ticks on x-axis.
            ymaxticks: Maximum number of ticks on y-axis.
            apply: Transformation to be applied on the data. Can be "log1p", "log2p1",
                "log10p1" or None.
            metric: The distance metric to use for the tree. Can be any metric supported
                by scipy.spatial.distance.pdist.
            scaler: Method to scale data along both axes. Can be "standard", "robust" or None.
            groupby: Name of the column to group data before scaling. If None, scaling is
                applied to the whole data, not group-wise.
            transposed: Whether to transpose the figure or not.
            **kwargs: Additional keyword arguments passed to hierarchical_clustering.

        Returns:
            Matplotlib figure representing the clustered heatmap.
        """
        if targets_var is not None:
            warnings.warn("targets_var is deprecated, use var_name instead", DeprecationWarning)
            var_name = targets_var

        warnings.simplefilter("ignore", ClusterWarning)
        if data is None:
            data = self.mint.crosstab(
                var_name=var_name, apply=apply, scaler=scaler, groupby=groupby
            )

        if transposed:
            data = data.T

        _, fig, ndx_x, ndx_y = hierarchical_clustering(
            data,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
            xmaxticks=xmaxticks,
            ymaxticks=ymaxticks,
            metric=metric,
            **kwargs,
        )

        self.mint.clustered = data.iloc[ndx_x, ndx_y]

        return fig

    def peak_shapes(
        self,
        fns: Optional[Union[str, List[str]]] = None,
        peak_labels: Optional[Union[str, List[str]]] = None,
        interactive: bool = False,
        **kwargs,
    ) -> Union[sns.axisgrid.FacetGrid, PlotlyFigure]:
        """Plot peak shapes extracted from MS-MINT results.

        Args:
            fns: Filename(s) to include in the plot. If None, all files in results are used.
            peak_labels: Peak label(s) to include in the plot. If None, all peaks are used.
            interactive: If True, returns an interactive Plotly figure instead of a static
                Matplotlib figure.
            **kwargs: Additional keyword arguments passed to the underlying plotting functions.

        Returns:
            Either a seaborn FacetGrid or a Plotly figure depending on the 'interactive' parameter.
        """
        if peak_labels is None:
            peak_labels = self.mint.peak_labels

        if len(self.mint.results) > 0:
            if not interactive:
                return plot_peak_shapes(
                    self.mint.results,
                    mint_metadata=self.mint.meta,
                    fns=fns,
                    peak_labels=peak_labels,
                    **kwargs,
                )
            else:
                return plotly_peak_shapes(
                    self.mint.results,
                    mint_metadata=self.mint.meta,
                    fns=fns,
                    peak_labels=peak_labels,
                    **kwargs,
                )

    def heatmap(
        self,
        col_name: str = "peak_max",
        normed_by_cols: bool = True,
        transposed: bool = False,
        clustered: bool = False,
        add_dendrogram: bool = False,
        name: str = "",
        correlation: bool = False,
        **kwargs,
    ) -> Optional[PlotlyFigure]:
        """Create an interactive heatmap to explore the data.

        Calls mint.crosstab() and then visualizes the result using plotly_heatmap.

        Args:
            col_name: Name of the column in mint.results to be analyzed.
            normed_by_cols: Whether or not to normalize the columns in the crosstab.
            transposed: If True, transpose matrix before plotting.
            clustered: Whether or not to cluster the rows.
            add_dendrogram: Whether or not to replace row labels with a dendrogram.
            name: Label to use for the colorbar.
            correlation: If True, convert data to correlation matrix before plotting.
            **kwargs: Additional keyword arguments passed to plotly_heatmap.

        Returns:
            Interactive Plotly heatmap figure, or None if no results are available.
        """
        data = self.mint.crosstab(col_name)

        # Remove path and suffix from file name.
        transform_filenames_func = lambda x: P(x).with_suffix("").name
        data.index = [transform_filenames_func(i) for i in data.index]

        if len(self.mint.results) > 0:
            return plotly_heatmap(
                data,
                normed_by_cols=normed_by_cols,
                transposed=transposed,
                clustered=clustered,
                add_dendrogram=add_dendrogram,
                name=col_name if not name else name,
                correlation=correlation,
                **kwargs,
            )
        return None

    def histogram_2d(
        self,
        fn: str,
        peak_label: Optional[str] = None,
        rt_margin: float = 0,
        mz_margin: float = 0,
        **kwargs,
    ) -> matplotlib.figure.Figure:
        """Create a 2D histogram of an MS file.

        Args:
            fn: File name of the MS file to visualize.
            peak_label: Target to focus. If provided, the plot will highlight the region
                defined by the target parameters.
            rt_margin: Margin in retention time dimension to add around the target region.
            mz_margin: Margin in m/z dimension to add around the target region.
            **kwargs: Additional keyword arguments passed to plot_metabolomics_hist2d.

        Returns:
            Matplotlib Figure containing the 2D histogram.
        """
        df = ms_file_to_df(fn)
        mz_range, rt_range, rt_min, rt_max = None, None, None, None
        mz_min, mz_max = None, None

        if peak_label is not None:
            target_data = self.mint.targets.loc[peak_label]
            mz_mean, mz_width, rt_min, rt_max = target_data[
                ["mz_mean", "mz_width", "rt_min", "rt_max"]
            ]
            mz_min, mz_max = mz_mean_width_to_min_max(mz_mean, mz_width)
            mz_range = (mz_min - mz_margin, mz_max + mz_margin)
            rt_range = (rt_min - rt_margin, rt_max + rt_margin)

        fig = plot_metabolomics_hist2d(df, mz_range=mz_range, rt_range=rt_range, **kwargs)

        if rt_min is not None and mz_min is not None:
            plt.plot(
                [rt_min, rt_max, rt_max, rt_min, rt_min],
                [mz_min, mz_min, mz_max, mz_max, mz_min],
                color="w",
                ls="--",
                lw=0.5,
            )
        if peak_label is None:
            plt.title(f"{P(fn).with_suffix('').name}")
        else:
            plt.title(f"{P(fn).with_suffix('').name}\n{peak_label}")
        return fig

    def chromatogram(
        self,
        fns: Optional[Union[str, List[str]]] = None,
        peak_labels: Optional[Union[str, List[str]]] = None,
        interactive: bool = False,
        filters: Optional[List[Any]] = None,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> Union[sns.axisgrid.FacetGrid, sns.axes._base.AxesBase, PlotlyFigure]:
        """Plot chromatograms extracted from one or more files.

        Args:
            fns: File name(s) to extract chromatograms from. If None, all files are used.
            peak_labels: Target(s) from Mint.targets.peak_label to use for extraction parameters.
                If None, all targets are used.
            interactive: If True, returns an interactive Plotly figure instead of a static Matplotlib figure.
            filters: List of filters to apply to the chromatograms before plotting.
            ax: Matplotlib axes to plot on. If None, a new figure is created.
            **kwargs: Additional keyword arguments passed to the underlying plotting functions.

        Returns:
            Either a seaborn FacetGrid, a single Axes, or a Plotly figure depending on
            the 'interactive' parameter and whether an 'ax' is provided.
        """
        if isinstance(fns, str):
            fns = [fns]

        if fns is not None:
            fns = tuple(fns)

        if isinstance(peak_labels, str):
            peak_labels = [peak_labels]

        if peak_labels is None:
            peak_labels = self.mint.peak_labels

        if peak_labels is not None:
            peak_labels = tuple(peak_labels)

        data = self.mint.get_chromatograms(fns=fns, peak_labels=peak_labels, filters=filters)

        if not interactive:
            params = dict(
                x="scan_time",
                y="intensity",
                col="peak_label",
                col_wrap=1,
                col_order=peak_labels,
                height=1.5,
                aspect=5,
                hue="ms_file_label",
                facet_kws=dict(sharey=False),
                marker=".",
                linewidth=0,
            )
            params.update(kwargs)

            if ax is None:
                g = sns.relplot(data=data, **params)

                for peak_label, ax in zip(peak_labels, g.axes.flatten()):
                    _, _, rt_min, rt_max = self.mint.get_target_params(peak_label)
                    if rt_min is not None and rt_max is not None:
                        ax.axvspan(rt_min, rt_max, color="lightgreen", alpha=0.5, zorder=-1)
                    ax.ticklabel_format(style="sci", axis="y", useOffset=False, scilimits=(0, 0))
                g.set_titles(template="{col_name}")

            else:
                g = sns.lineplot(
                    data=data, x="scan_time", y="intensity", hue="ms_file_label", ax=ax, **kwargs
                )
            return g

        else:
            g = px.line(
                data_frame=data,
                x="scan_time",
                y="intensity",
                facet_col="peak_label",
                color="ms_file_label",
                height=700,
                facet_col_wrap=1,
            )
            g.update_xaxes(matches=None)
            g.update_yaxes(matches=None)
            return g
