import numpy as np
import seaborn as sns

from warnings import simplefilter
from scipy.cluster.hierarchy import ClusterWarning

from pathlib import Path as P
from matplotlib import pyplot as plt

from .plotly_tools import plotly_heatmap, plotly_peak_shapes
from .matplotlib_tools import (
    plot_peak_shapes,
    hierarchical_clustering,
    plot_metabolomics_hist2d,
)

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
        data=None,
        peak_labels=None,
        ms_files=None,
        title=None,
        figsize=(8, 8),
        targets_var="peak_max",
        vmin=-3,
        vmax=3,
        xmaxticks=None,
        ymaxticks=None,
        transform_func="log2p1",
        scaler_ms_file=None,
        scaler_peak_label="standard",
        metric="cosine",
        transform_filenames_func="basename",
        transposed=False,
        **kwargs,
    ):
        """
        Performs a cluster analysis and plots a heatmap. If no data is provided,
        data is taken form self.mint.crosstab(targets_var).
        The clustered non-transformed non-scaled data is stored in `self.mint.clustered`.

        :param transform_func: default 'log2p1', values: [None, 'log1p', 'log2p1', 'log10p1']
            - None: no transformation
            - log1p: tranform data with lambda x: np.log1p(x)
            - log2p1: transform data with lambda x: log2(x+1)
            - log10p1: transform data with lambda x: log10(x+1)

        :param scaler_ms_file: default None, values: [None, 'standard', 'robust']
            - scaler used to scale along ms_file axis
            - if None no scaling is applied
            - if 'standard' use scikit learn StandardScaler()
            - if 'robust' use scikit learn RobustScaler()

        :param scaler_peak_label: default 'standard'
            - like scaler_ms_file, but scaling along peak_label axis

        :param metric: default 'cosine', can be string or a list of two values:
            if two values are provided e.g. ('cosine', 'euclidean') the first
            will be used to cluster the x-axis and the second for the y-axis.

            'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
            'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis',
            'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
            'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.
            More information:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

        :param transpose: bool, default False
            - True: transpose the figure
        """
        if len(self.mint.results) == 0:
            return None

        simplefilter("ignore", ClusterWarning)
        if data is None:
            data = self.mint.crosstab(targets_var).copy()

        if peak_labels is not None:
            data = data[peak_labels]

        if ms_files is not None:
            data = data.loc[ms_files]

        tmp_data = data.copy()

        if transform_func == "log1p":
            transform_func = np.log1p
        if transform_func == "log2p1":
            transform_func = lambda x: np.log2(x + 1)
        if transform_func == "log10p1":
            transform_func = lambda x: np.log10(x + 1)
        if transform_func is not None:
            tmp_data = tmp_data.apply(transform_func)

        if transform_filenames_func == "basename":
            transform_filenames_func = lambda x: P(x).with_suffix("").name
        if transform_filenames_func is not None:
            tmp_data.index = [transform_filenames_func(i) for i in tmp_data.index]

        # Scale along ms-files
        if scaler_ms_file is not None:
            tmp_data = scale_dataframe(tmp_data.T, scaler_ms_file).T

        # Scale along peak_labels
        if scaler_peak_label is not None:
            tmp_data = scale_dataframe(tmp_data, scaler_peak_label)

        if transposed:
            tmp_data = tmp_data.T

        _, fig, ndx_x, ndx_y = hierarchical_clustering(
            tmp_data,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
            xmaxticks=xmaxticks,
            ymaxticks=ymaxticks,
            metric=metric,
            **kwargs,
        )

        if not transposed:
            self.mint.clustered = data.iloc[ndx_x, ndx_y]
        else:
            self.mint.clustered = data.iloc[ndx_y, ndx_x]
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
                    self.mint.results, fns=fns, peak_labels=peak_labels, **kwargs
                )
            else:
                return plotly_peak_shapes(
                    self.mint.results, fns=fns, peak_labels=peak_labels, **kwargs
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

    def chromatogram(self, fns=None, peak_labels=None, interactive=False, filters=None, **kwargs):
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
