import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Optional, Union, List, Dict, Any, Tuple, Literal
from matplotlib.figure import Figure
from plotly.graph_objs._figure import Figure as PlotlyFigure

import plotly.figure_factory as ff
from plotly import express as px
from __future__ import annotations

from sklearn.decomposition import PCA
from .tools import scale_dataframe


class PrincipalComponentsAnalyser:
    """Class for applying PCA to MS-MINT analysis results.

    This class provides functionality to perform Principal Component Analysis on
    MS-MINT metabolomics data and store the results for visualization.

    Attributes:
        mint: The Mint instance containing the data to analyze.
        results: Dictionary containing PCA results after running the analysis.
        plot: PCA_Plotter instance for visualizing the PCA results.
    """

    def __init__(self, mint: Optional["ms_mint.Mint.Mint"] = None) -> None:
        """Initialize a PrincipalComponentsAnalyser instance.

        Args:
            mint: Mint instance containing the data to analyze.
        """
        self.mint = mint
        self.results: Optional[Dict[str, Any]] = None
        self.plot = PCA_Plotter(self)

    def run(
        self,
        n_components: int = 3,
        on: Optional[str] = None,
        var_name: str = "peak_max",
        fillna: Union[str, float] = "median",
        apply: Optional[str] = None,
        groupby: Optional[Union[str, List[str]]] = None,
        scaler: str = "standard",
    ) -> None:
        """Run Principal Component Analysis on the current results.

        Performs PCA on the data and stores results in self.results.

        Args:
            n_components: Number of PCA components to calculate.
            on: Deprecated, use var_name instead.
            var_name: Column name from results to use for PCA.
            fillna: Method to fill missing values. One of "median", "mean", "zero",
                or a numeric value.
            apply: Transformation to apply to the data before PCA.
            groupby: Column(s) to group by before analysis.
            scaler: Method to scale the data. One of "standard", "robust", "minmax".

        Raises:
            DeprecationWarning: If the deprecated 'on' parameter is used.
        """
        if on is not None:
            warnings.warn("on is deprecated, use var_name instead", DeprecationWarning)
            var_name = on

        df = self.mint.crosstab(var_name=var_name, apply=apply, scaler=scaler, groupby=groupby)

        if fillna == "median":
            fillna = df.median()
        elif fillna == "mean":
            fillna = df.mean()
        elif fillna == "zero":
            fillna = 0

        df = df.fillna(fillna)

        min_dim = min(df.shape)
        n_components = min(n_components, min_dim)
        pca = PCA(n_components)
        X_projected = pca.fit_transform(df)
        # Convert to dataframe
        df_projected = pd.DataFrame(X_projected, index=df.index.get_level_values(0))
        # Set columns to PC-1, PC-2, ...
        df_projected.columns = [f"PC-{int(i) + 1}" for i in df_projected.columns]

        # Calculate cumulative explained variance in percent
        explained_variance = pca.explained_variance_ratio_ * 100
        cum_expl_var = np.cumsum(explained_variance)

        # Create feature contributions
        a = np.zeros((n_components, n_components), int)
        np.fill_diagonal(a, 1)
        dfc = pd.DataFrame(pca.inverse_transform(a))
        dfc.columns = df.columns
        dfc.index = [f"PC-{i + 1}" for i in range(n_components)]
        dfc.index.name = "PC"
        # convert to long format
        dfc = dfc.stack().reset_index().rename(columns={0: "Coefficient"})

        self.results = {
            "df_projected": df_projected,
            "cum_expl_var": cum_expl_var,
            "n_components": n_components,
            "type": "PCA",
            "feature_contributions": dfc,
            "class": pca,
        }


class PCA_Plotter:
    """Class for visualizing PCA results from MS-MINT analysis.

    This class provides methods to create various plots of PCA results,
    including cumulative variance plots, pairplots, and loading plots.

    Attributes:
        pca: The PrincipalComponentsAnalyser instance containing results to visualize.
    """

    def __init__(self, pca: PrincipalComponentsAnalyser) -> None:
        """Initialize a PCA_Plotter instance.

        Args:
            pca: PrincipalComponentsAnalyser instance with results to visualize.
        """
        self.pca = pca

    def cumulative_variance(
        self, interactive: bool = False, **kwargs
    ) -> Union[Figure, PlotlyFigure]:
        """Plot the cumulative explained variance of principal components.

        Args:
            interactive: If True, returns a Plotly interactive figure.
                If False, returns a static Matplotlib figure.
            **kwargs: Additional keyword arguments passed to the underlying plotting functions.

        Returns:
            Either a Matplotlib figure or a Plotly figure depending on the interactive parameter.
        """
        if interactive:
            return self.cumulative_variance_px(**kwargs)
        else:
            return self.cumulative_variance_sns(**kwargs)

    def cumulative_variance_px(self, **kwargs) -> PlotlyFigure:
        """Create an interactive Plotly plot of cumulative explained variance.

        Args:
            **kwargs: Additional keyword arguments passed to px.bar.

        Returns:
            Plotly figure showing cumulative explained variance.
        """
        n_components = self.pca.results["n_components"]
        cum_expl_var = self.pca.results["cum_expl_var"]
        df = pd.DataFrame(
            {
                "Principal Component": np.arange(n_components) + 1,
                "Explained variance [%]": cum_expl_var,
            }
        )
        fig = px.bar(
            df,
            x="Principal Component",
            y="Explained variance [%]",
            title="Cumulative explained variance",
            labels={
                "Principal Component": "Principal Component",
                "Explained variance [%]": "Explained variance [%]",
            },
            **kwargs,
        )
        fig.update_layout(autosize=True, showlegend=False)
        return fig

    def cumulative_variance_sns(self, **kwargs) -> Figure:
        """Create a static Matplotlib plot of cumulative explained variance.

        Args:
            **kwargs: Additional keyword arguments for figure customization.
                'aspect': Width-to-height ratio of the figure (default: 1).
                'height': Height of the figure in inches (default: 5).

        Returns:
            Matplotlib figure showing cumulative explained variance.
        """
        # Set default values for aspect and height
        aspect = kwargs.get("aspect", 1)
        height = kwargs.get("height", 5)

        n_components = self.pca.results["n_components"]
        cum_expl_var = self.pca.results["cum_expl_var"]

        # Calculate width based on aspect ratio and number of components
        width = height * aspect

        fig, ax = plt.subplots(figsize=(width, height))
        ax.bar(
            np.arange(n_components) + 1,
            cum_expl_var,
            facecolor="grey",
            edgecolor="none",
        )
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained variance [%]")
        ax.set_title("Cumulative explained variance")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(range(1, len(cum_expl_var) + 1))
        return fig

    def _prepare_data(
        self, n_components: int = 3, hue: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """Prepare data for pairplot visualization.

        Args:
            n_components: Number of principal components to include.
            hue: Labels used for coloring points. If a string, data is taken from
                the mint.meta DataFrame. If a list, values are used directly.

        Returns:
            DataFrame containing the prepared data for visualization.
        """
        df = self.pca.results["df_projected"].copy()
        cols = df.columns.to_list()[:n_components]
        df = df[cols]

        df = pd.merge(
            df, self.pca.mint.meta.dropna(axis=1, how="all"), left_index=True, right_index=True
        )

        if hue and (not isinstance(hue, str)):
            df["Label"] = hue
            df["Label"] = df["Label"].astype(str)

        return df

    def pairplot(
        self,
        n_components: int = 3,
        hue: Optional[Union[str, List[str]]] = None,
        fig_kws: Optional[Dict[str, Any]] = None,
        interactive: bool = False,
        **kwargs,
    ) -> Union[sns.axisgrid.PairGrid, PlotlyFigure]:
        """Create a pairplot of principal components.

        Args:
            n_components: Number of principal components to include in the plot.
            hue: Labels used for coloring points. If a string, data is taken from
                the mint.meta DataFrame. If a list, values are used directly.
            fig_kws: Keyword arguments passed to plt.figure if using seaborn.
            interactive: If True, returns a Plotly interactive figure.
                If False, returns a static Seaborn PairGrid.
            **kwargs: Additional keyword arguments passed to the underlying plotting functions.

        Returns:
            Either a Seaborn PairGrid or a Plotly figure depending on the interactive parameter.
        """
        df = self._prepare_data(n_components=n_components, hue=hue)

        if isinstance(hue, list):
            hue = "label"

        if interactive:
            return self.pairplot_plotly(df, color_col=hue, **kwargs)
        else:
            return self.pairplot_sns(df, fig_kws=fig_kws, hue=hue, **kwargs)

    def pairplot_sns(
        self, df: pd.DataFrame, fig_kws: Optional[Dict[str, Any]] = None, **kwargs
    ) -> sns.axisgrid.PairGrid:
        """Create a static Seaborn pairplot of principal components.

        Args:
            df: DataFrame containing the data to visualize.
            fig_kws: Keyword arguments passed to plt.figure.
            **kwargs: Additional keyword arguments passed to sns.pairplot.

        Returns:
            Seaborn PairGrid object.
        """
        if fig_kws is None:
            fig_kws = {}
        plt.figure(**fig_kws)
        g = sns.pairplot(df, **kwargs)
        return g

    def pairplot_plotly(
        self, df: pd.DataFrame, color_col: Optional[str] = None, **kwargs
    ) -> PlotlyFigure:
        """Create an interactive Plotly pairplot of principal components.

        Args:
            df: DataFrame containing the data to visualize.
            color_col: Column name to use for coloring points.
            **kwargs: Additional keyword arguments passed to ff.create_scatterplotmatrix.

        Returns:
            Plotly figure object.
        """
        columns = df.filter(regex=f"PC|^{color_col}$").columns
        fig = ff.create_scatterplotmatrix(
            df[columns], index=color_col, hovertext=df.index, **kwargs
        )
        # set the legendgroup equal to the marker color
        for t in fig.data:
            t.legendgroup = t.marker.color
        return fig

    def loadings(
        self, interactive: bool = False, **kwargs
    ) -> Union[sns.axisgrid.FacetGrid, PlotlyFigure]:
        """Plot PCA loadings (feature contributions to principal components).

        Args:
            interactive: If True, returns a Plotly interactive figure.
                If False, returns a static Seaborn FacetGrid.
            **kwargs: Additional keyword arguments passed to the underlying plotting functions.

        Returns:
            Either a Seaborn FacetGrid or a Plotly figure depending on the interactive parameter.
        """
        if interactive:
            return self.loadings_plotly(**kwargs)
        else:
            return self.loadings_sns(**kwargs)

    def loadings_sns(self, **kwargs) -> sns.axisgrid.FacetGrid:
        """Create a static Seaborn plot of PCA loadings.

        Args:
            **kwargs: Additional keyword arguments passed to sns.catplot.
                If 'row' is not specified, it defaults to 'PC'.

        Returns:
            Seaborn FacetGrid object.
        """
        if "row" not in kwargs:
            kwargs["row"] = "PC"
        g = sns.catplot(
            data=self.pca.results["feature_contributions"],
            x="peak_label",
            y="Coefficient",
            kind="bar",
            **kwargs,
        )
        plt.tight_layout()
        return g

    def loadings_plotly(self, **kwargs) -> PlotlyFigure:
        """Create an interactive Plotly plot of PCA loadings.

        Args:
            **kwargs: Additional keyword arguments passed to px.bar.
                If 'facet_row' is not specified, it defaults to 'PC'.

        Returns:
            Plotly figure object.
        """
        if "facet_row" not in kwargs:
            kwargs["facet_row"] = "PC"
        fig = px.bar(
            self.pca.results["feature_contributions"],
            x="peak_label",
            y="Coefficient",
            barmode="group",
            **kwargs,
        )
        return fig
