import numpy as np
import seaborn as sns
import warnings

from scipy.cluster.hierarchy import ClusterWarning

from pathlib import Path as P

import matplotlib
from matplotlib import pyplot as plt

from typing import Optional, List, Tuple

from .plotly_tools import plotly_heatmap, plotly_peak_shapes
from .matplotlib_tools import (
    plot_peak_shapes,
    hierarchical_clustering,
    plot_metabolomics_hist2d,
)

import pandas as pd
import plotly.express as px

from .tools import scale_dataframe, mz_mean_width_to_min_max
from .io import ms_file_to_df


class MintPlotter:
    """
    Plot generator for mint.results.

    :param mint: Mint instance
    :type mint: ms_mint.Mint.Mint
    """

    def __init__(self, mint):
        """
        Plot generator for mint.results.

        :param mint: Mint instance
        :type mint: ms_mint.Mint.Mint
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
        """
        Performs a cluster analysis and plots a heatmap. If no data is provided,
        data is taken from self.mint.crosstab(var_name).
        The clustered non-transformed non-scaled data is stored in `self.mint.clustered`.

        :param data: DataFrame with data to be used for clustering. If None, crosstab of mint instance is used.
        :type data: pandas.DataFrame, optional

        :param var_name: Name of the column from data to be used for cell values in the heatmap. Defaults to "peak_max".
        :type var_name: str

        :param apply: Transformation to be applied on the data. Can be "log1p", "log2p1", "log10p1" or None. Defaults to "log2p1".
        :type apply: str, optional

        :param scaler: Method to scale data along both axes. Can be "standard", "robust" or None. Defaults to "standard".
        :type scaler: str, optional

        :param groupby: Name of the column to group data before scaling. If None, scaling is applied to the whole data, not group-wise.
        :type groupby: str, optional

        :param metric: The distance metric to use for the tree. Can be any metric supported by scipy.spatial.distance.pdist.
        :type metric: str, optional

        :param transposed: Whether to transpose the figure or not. Defaults to False.
        :type transposed: bool, optional

        :return: Matplotlib figure representing the clustered heatmap.
        :rtype: matplotlib.figure.Figure
        """

        if targets_var is not None:
            warnings.warn("targets_var is depricated use var_name instead", DeprecationWarning)
            var_name = targets_var

        warnings.simplefilter("ignore", ClusterWarning)
        if data is None:
            data = self.mint.crosstab(var_name=var_name, apply=apply, scaler=scaler, groupby=groupby)

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
    

    def peak_shapes(self, fns=None, peak_labels=None, interactive=False, **kwargs):
        """Plot peak shapes.

        :return: Figure with peak shapes.
        :rtype: seaborn.axisgrid.FacetGrid
        """

        if peak_labels is None:
            peak_labels = self.mint.peak_labels

        if len(self.mint.results) > 0:
            if not interactive:
                return plot_peak_shapes(
                    self.mint.results, mint_metadata=self.mint.meta, fns=fns, peak_labels=peak_labels, **kwargs
                )
            else:
                return plotly_peak_shapes(
                    self.mint.results, mint_metadata=self.mint.meta, fns=fns, peak_labels=peak_labels, **kwargs
                )

    def heatmap(
        self,
        col_name="peak_max",
        normed_by_cols=True,
        transposed=False,
        clustered=False,
        add_dendrogram=False,
        name="",
        correlation=False,
        **kwargs,
    ):
        """
        Creates an interactive heatmap
        that can be used to explore the data interactively.
        `mint.crosstab()` is called and then subjected to
        the `mint.vis.plotly.plotly_tools.plot_heatmap()`.

        :param col_name: str, default='peak_max'
            Name of the column in `mint.results` to be analysed.
        :param normed_by_cols: bool, default=True
            Whether or not to normalize the columns in the crosstab.
        :param clustered: bool, default=False
            Whether or not to cluster the rows.
        :param add_dendrogram: bool, default=False
            Whether or not to replace row labels with a dendrogram.
        :param transposed: bool, default=False
            If True transpose matrix before plotting.
        :param correlation: bool, default=False
            If True convert data to correlation matrix before plotting.

        :return: Interactive heatmap.
        :rtype: plotly.graph_objs._figure.Figure
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
                name=col_name,
                correlation=correlation,
                **kwargs,
            )

    def histogram_2d(self, fn, peak_label=None, rt_margin=0, mz_margin=0, **kwargs):
        """2D histogram of a ms file.

        :param fn: File name
        :type fn: str, optional
        :param peak_label: Target to focus, defaults to None
        :type peak_label: str, optional
        :param rt_margin: Margin in Rt dimension, defaults to 0
        :type rt_margin: int, optional
        :param mz_margin: Margin in mz dimension, defaults to 0
        :type mz_margin: int, optional
        :return: Figure object
        :rtype: matplotlib.Figure
        """
        df = ms_file_to_df(fn)
        mz_range, rt_range, rt_min, rt_max = None, None, None, None
        if peak_label is not None:
            target_data = self.mint.targets.loc[peak_label]
            mz_mean, mz_width, rt_min, rt_max = target_data[
                ["mz_mean", "mz_width", "rt_min", "rt_max"]
            ]
            mz_min, mz_max = mz_mean_width_to_min_max(mz_mean, mz_width)
            mz_range = (mz_min - mz_margin, mz_max + mz_margin)
            rt_range = (rt_min - rt_margin, rt_max + rt_margin)

        fig = plot_metabolomics_hist2d(
            df, mz_range=mz_range, rt_range=rt_range, **kwargs
        )

        if rt_min is not None:
            plt.plot(
                [rt_min, rt_max, rt_max, rt_min, rt_min],
                [mz_min, mz_min, mz_max, mz_max, mz_min],
                color="w",
                ls="--",
                lw=0.5,
            )
        if peak_label is None:
            plt.title(f'{P(fn).with_suffix("").name}')
        else:
            plt.title(f'{P(fn).with_suffix("").name}\n{peak_label}')
        return fig

    def chromatogram(self, fns=None, peak_labels=None, interactive=False, filters=None, ax=None, **kwargs):
        """Plot the chromatogram extracted from one or more files.

        :param fns: File names to extract chromatogram from.
        :type fns: str or List[str]
        :param peak_label: Target from Mint.targets.peak_label to take mz
        parameters, defaults to None. If None `mz_mean and mz_width are used`.
        :type peak_label: str, optional
        :param mz_mean: m/z value for chromatogram, defaults to None
        :type mz_mean: float, optional
        :param mz_width: mz_width, defaults to None
        :type mz_width: int>0, optional
        :return: Plot of chromatogram(s)
        :rtype: matplotlib.Figure
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
                    _, _, rt_min, rt_max = self.mint.get_target_params(
                        peak_label
                    )
                    if rt_min is not None and rt_max is not None:
                        ax.axvspan(rt_min, rt_max, color="lightgreen", alpha=0.5, zorder=-1)
                    ax.ticklabel_format(
                        style="sci", axis="y", useOffset=False, scilimits=(0, 0)
                    )
                g.set_titles(template="{col_name}")
            
            else:
                g = sns.lineplot(data=data, 
                                 x="scan_time",
                                 y="intensity",
                                 hue='ms_file_label',
                                 ax=ax,
                                 **kwargs)
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
