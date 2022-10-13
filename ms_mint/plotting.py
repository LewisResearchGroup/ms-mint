import numpy as np

from warnings import simplefilter
from scipy.cluster.hierarchy import ClusterWarning

from pathlib import Path as P

from .plotly_tools import plotly_heatmap, plotly_peak_shapes
from .matplotlib_tools import plot_peak_shapes, hierarchical_clustering
from .tools import scale_dataframe


class MintResultsPlotter:
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
        metric="euclidean",
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

        :param metric: default 'euclidean', can be string or a list of two values:
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
        elif transform_filenames_func is not None:
            tmp_data.columns = [transform_filenames_func(i) for i in tmp_data.columns]

        # Scale along ms-files
        if scaler_ms_file is not None:
            tmp_data = scale_dataframe(tmp_data.T, scaler_ms_file).T

        # Scale along peak_labels
        if scaler_peak_label is not None:
            tmp_data = scale_dataframe(tmp_data, scaler_peak_label)

        if transposed:
            tmp_data = tmp_data.T

        clustered, fig, ndx_x, ndx_y = hierarchical_clustering(
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

    def peak_shapes(self, interactive=False, **kwargs):
        """Plot peak shapes.

        :return: Figure with peak shapes.
        :rtype: seaborn.axisgrid.FacetGrid
        """
        if len(self.mint.results) > 0:
            if not interactive:
                return plot_peak_shapes(self.mint.results, **kwargs)
            else:
                return plotly_peak_shapes(self.mint.results, **kwargs)

    def heatmap(
        self,
        col_name="peak_max",
        normed_by_cols=True,
        transposed=False,
        clustered=False,
        add_dendrogram=False,
        name="",
        correlation=False,
        **kwargs
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
        if len(self.mint.results) > 0:
            return plotly_heatmap(
                self.mint.crosstab(col_name),
                normed_by_cols=normed_by_cols,
                transposed=transposed,
                clustered=clustered,
                add_dendrogram=add_dendrogram,
                name=col_name,
                correlation=correlation,
                **kwargs
            )

   
