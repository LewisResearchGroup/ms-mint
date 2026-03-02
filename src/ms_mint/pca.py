"""Principal Component Analysis for metabolomics data."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .Mint import Mint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import seaborn as sns
from matplotlib.figure import Figure
from plotly import express as px
from plotly.graph_objs._figure import Figure as PlotlyFigure
from sklearn.decomposition import PCA


class PrincipalComponentsAnalyser:
    """Class for applying PCA to MS-MINT analysis results.

    This class provides functionality to perform Principal Component Analysis on
    MS-MINT metabolomics data and store the results for visualization.

    Attributes:
        mint: The Mint instance containing the data to analyze.
        results: Dictionary containing PCA results after running the analysis.
        plot: PCA_Plotter instance for visualizing the PCA results.
    """

    def __init__(self, mint: Mint | None = None) -> None:
        """Initialize a PrincipalComponentsAnalyser instance.

        Args:
            mint: Mint instance containing the data to analyze.
        """
        self.mint = mint
        self.results: dict[str, Any] | None = None
        self.plot = PCA_Plotter(self)

    def run(
        self,
        n_components: int = 3,
        on: str | None = None,
        var_name: str = "peak_max",
        fillna: str | float = "median",
        apply: str | None = None,
        groupby: str | list[str] | None = None,
        scaler: str = "standard",
        peak_labels: list[str] | None = None,
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
            peak_labels: List of peak labels to include. If None, all peaks are used.

        Raises:
            DeprecationWarning: If the deprecated 'on' parameter is used.
        """
        if on is not None:
            warnings.warn("on is deprecated, use var_name instead", DeprecationWarning)
            var_name = on

        df = self.mint.crosstab(var_name=var_name, apply=apply, scaler=scaler, groupby=groupby, peak_labels=peak_labels)

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
    ) -> Figure | PlotlyFigure:
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

    def scatter(
        self,
        x_component: int = 1,
        y_component: int = 2,
        color_by: str | None = None,
        interactive: bool = False,
        **kwargs,
    ) -> Figure | PlotlyFigure:
        """Create a scatter plot of two principal components.

        Args:
            x_component: Principal component number for x-axis (1-indexed).
            y_component: Principal component number for y-axis (1-indexed).
            color_by: Metadata column to use for coloring points.
            interactive: If True, returns a Plotly interactive figure.
            **kwargs: Additional keyword arguments passed to plotting functions.

        Returns:
            Either a Matplotlib figure or a Plotly figure depending on interactive.
        """
        if interactive:
            return self.scatter_plotly(x_component, y_component, color_by, **kwargs)
        else:
            return self.scatter_sns(x_component, y_component, color_by, **kwargs)

    def scatter_sns(
        self,
        x_component: int = 1,
        y_component: int = 2,
        color_by: str | None = None,
        **kwargs,
    ) -> Figure:
        """Create a static scatter plot of two principal components.

        Args:
            x_component: Principal component number for x-axis (1-indexed).
            y_component: Principal component number for y-axis (1-indexed).
            color_by: Metadata column to use for coloring points.
            **kwargs: Additional keyword arguments for figure customization.

        Returns:
            Matplotlib figure showing the scatter plot.
        """
        df = self.pca.results["df_projected"].copy()
        x_col = f"PC-{x_component}"
        y_col = f"PC-{y_component}"

        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Components {x_component} or {y_component} not available")

        # Merge with metadata if color_by is specified
        if color_by and color_by != "none":
            meta = self.pca.mint.meta.dropna(axis=1, how="all")
            if color_by in meta.columns:
                df = pd.merge(df, meta[[color_by]], left_index=True, right_index=True, how="left")

        height = kwargs.get("height", 6)
        width = kwargs.get("width", 8)

        fig, ax = plt.subplots(figsize=(width, height))

        if color_by and color_by != "none" and color_by in df.columns:
            # Get unique categories
            categories = df[color_by].dropna().unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
            for cat, color in zip(categories, colors):
                mask = df[color_by] == cat
                ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                          c=[color], label=str(cat), alpha=0.7, s=50)
            ax.legend(title=color_by, bbox_to_anchor=(1.02, 1), loc='upper left')
        else:
            ax.scatter(df[x_col], df[y_col], alpha=0.7, s=50, c='steelblue')

        # Get explained variance for axis labels
        cum_var = self.pca.results["cum_expl_var"]
        var_x = cum_var[x_component - 1] if x_component == 1 else cum_var[x_component - 1] - cum_var[x_component - 2]
        var_y = cum_var[y_component - 1] if y_component == 1 else cum_var[y_component - 1] - cum_var[y_component - 2]

        ax.set_xlabel(f"{x_col} ({var_x:.1f}%)")
        ax.set_ylabel(f"{y_col} ({var_y:.1f}%)")
        ax.set_title(f"PCA: {x_col} vs {y_col}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        return fig

    def scatter_plotly(
        self,
        x_component: int = 1,
        y_component: int = 2,
        color_by: str | None = None,
        **kwargs,
    ) -> PlotlyFigure:
        """Create an interactive Plotly scatter plot of two principal components.

        Args:
            x_component: Principal component number for x-axis (1-indexed).
            y_component: Principal component number for y-axis (1-indexed).
            color_by: Metadata column to use for coloring points.
            **kwargs: Additional keyword arguments passed to px.scatter.

        Returns:
            Plotly figure showing the scatter plot.
        """
        df = self.pca.results["df_projected"].copy()
        x_col = f"PC-{x_component}"
        y_col = f"PC-{y_component}"

        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Components {x_component} or {y_component} not available")

        # Merge with metadata if color_by is specified
        if color_by and color_by != "none":
            meta = self.pca.mint.meta.dropna(axis=1, how="all")
            if color_by in meta.columns:
                df = pd.merge(df, meta[[color_by]], left_index=True, right_index=True, how="left")

        # Get explained variance for axis labels
        cum_var = self.pca.results["cum_expl_var"]
        var_x = cum_var[x_component - 1] if x_component == 1 else cum_var[x_component - 1] - cum_var[x_component - 2]
        var_y = cum_var[y_component - 1] if y_component == 1 else cum_var[y_component - 1] - cum_var[y_component - 2]

        color_col = color_by if (color_by and color_by != "none" and color_by in df.columns) else None

        fig = px.scatter(
            df.reset_index(),
            x=x_col,
            y=y_col,
            color=color_col,
            hover_name="index" if "index" in df.reset_index().columns else None,
            labels={
                x_col: f"{x_col} ({var_x:.1f}%)",
                y_col: f"{y_col} ({var_y:.1f}%)",
            },
            title=f"PCA: {x_col} vs {y_col}",
            **kwargs,
        )
        fig.update_layout(autosize=True)
        return fig

    def _prepare_data(
        self, n_components: int = 3, hue: str | list[str] | None = None
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
        hue: str | list[str] | None = None,
        fig_kws: dict[str, Any] | None = None,
        interactive: bool = False,
        **kwargs,
    ) -> sns.axisgrid.PairGrid | PlotlyFigure:
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
        self, df: pd.DataFrame, fig_kws: dict[str, Any] | None = None, **kwargs
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
        # Only plot PC columns, not merged metadata columns
        pc_cols = [c for c in df.columns if c.startswith("PC-")]
        if "vars" not in kwargs:
            kwargs["vars"] = pc_cols
        plt.figure(**fig_kws)
        g = sns.pairplot(df, **kwargs)
        return g

    def pairplot_plotly(
        self, df: pd.DataFrame, color_col: str | None = None, **kwargs
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
    ) -> sns.axisgrid.FacetGrid | PlotlyFigure:
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
