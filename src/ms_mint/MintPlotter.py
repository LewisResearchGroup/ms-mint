"""Visualization tools for MS-MINT analysis results."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .Mint import Mint

import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from plotly.graph_objs._figure import Figure as PlotlyFigure
from scipy.cluster.hierarchy import ClusterWarning

from .io import ms_file_to_df
from .matplotlib_tools import (
    hierarchical_clustering,
    plot_metabolomics_hist2d,
    plot_peak_shapes,
)
from .plotly_tools import plotly_heatmap, plotly_peak_shapes
from .tools import mz_mean_width_to_min_max, unwrap_reactive


class MintPlotter:
    """Plot generator for visualizing MS-MINT analysis results.

    This class provides various visualization methods for metabolomics data processed
    by MS-MINT, including heatmaps, chromatograms, peak shapes, and 2D histograms.

    Attributes:
        mint: The Mint instance containing data to be visualized.
    """

    def __init__(self, mint: Mint) -> None:
        """Initialize the MintPlotter with a Mint instance.

        Args:
            mint: Mint instance containing the data to visualize.
        """
        self.mint = mint

    def hierarchical_clustering(
        self,
        data: pd.DataFrame | None = None,
        peak_labels: list[str] | None = None,
        ms_files: list[str] | None = None,
        title: str | None = None,
        figsize: tuple[int, int] = (8, 8),
        targets_var: str | None = None,
        var_name: str = "peak_max",
        vmin: int = -3,
        vmax: int = 3,
        xmaxticks: int | None = None,
        ymaxticks: int | None = None,
        apply: str = "log2p1",
        metric: str = "cosine",
        scaler: str = "standard",
        groupby: str | None = None,
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

    # Alias for hierarchical_clustering
    hc = hierarchical_clustering

    def peak_shapes(
        self,
        fns: str | list[str] | None = None,
        peak_labels: str | list[str] | None = None,
        interactive: bool = False,
        **kwargs,
    ) -> sns.axisgrid.FacetGrid | PlotlyFigure:
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
    ) -> PlotlyFigure | None:
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
        transform_filenames_func = lambda x: Path(x).with_suffix("").name
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
        peak_label: str | None = None,
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
            plt.title(f"{Path(fn).with_suffix('').name}")
        else:
            plt.title(f"{Path(fn).with_suffix('').name}\n{peak_label}")
        return fig

    def chromatogram(
        self,
        fns: str | list[str] | None = None,
        peak_labels: str | list[str] | None = None,
        interactive: bool = False,
        filters: list[Any] | None = None,
        ax: plt.Axes | None = None,
        nthreads: int | None = None,
        **kwargs,
    ) -> sns.axisgrid.FacetGrid | sns.axes._base.AxesBase | PlotlyFigure:
        """Plot chromatograms extracted from one or more files.

        Args:
            fns: File name(s) to extract chromatograms from. If None, all files are used.
            peak_labels: Target(s) from Mint.targets.peak_label to use for extraction parameters.
                If None, all targets are used.
            interactive: If True, returns an interactive Plotly figure instead of a static Matplotlib figure.
            filters: List of filters to apply to the chromatograms before plotting.
            ax: Matplotlib axes to plot on. If None, a new figure is created.
            nthreads: Number of threads for parallel chromatogram extraction.
            **kwargs: Additional keyword arguments passed to the underlying plotting functions.

        Returns:
            Either a seaborn FacetGrid, a single Axes, or a Plotly figure depending on
            the 'interactive' parameter and whether an 'ax' is provided.
        """
        # Handle Reactive objects
        fns = unwrap_reactive(fns)
        peak_labels = unwrap_reactive(peak_labels)
        nthreads = unwrap_reactive(nthreads)

        if isinstance(fns, str):
            fns = [fns]

        if fns is not None:
            fns = tuple(fns)

        if isinstance(peak_labels, str):
            peak_labels = [peak_labels]

        if peak_labels is None:
            peak_labels = self.mint.peak_labels

        # Ensure peak_labels is a list/tuple, not Reactive
        peak_labels = unwrap_reactive(peak_labels)

        if peak_labels is not None:
            peak_labels = tuple(peak_labels)

        data = self.mint.get_chromatograms(fns=fns, peak_labels=peak_labels, filters=filters, nthreads=nthreads)

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

    def distribution(
        self,
        var_name: str = "peak_max",
        apply: str | None = "log2p1",
        style: str = "boxplot",
        hue: str | None = None,
        col: str | None = None,
        row: str | None = None,
        col_wrap: int | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        peak_labels: list[str] | None = None,
        interactive: bool = False,
        **kwargs,
    ) -> matplotlib.figure.Figure | PlotlyFigure:
        """Create distribution plots (histogram, boxplot, violin) for peak values.

        Args:
            var_name: Column from results to plot.
            apply: Transformation to apply ("log2p1", "log10p1", or None).
            style: Plot style - "boxplot", "violin", or "histogram".
            hue: Metadata column for grouping/coloring.
            col: Variable for facet columns (e.g., "peak_label", "ms_file", or metadata column).
            row: Variable for facet rows (e.g., "peak_label", "ms_file", or metadata column).
            col_wrap: Number of columns before wrapping (only used if col is set and row is None).
            x_label: Custom x-axis label (default: auto-generated).
            y_label: Custom y-axis label (default: auto-generated from var_name and apply).
            peak_labels: List of peak labels to include. If None, all peaks are included.
            interactive: If True, returns Plotly figure; otherwise Matplotlib.
            **kwargs: Additional arguments passed to plotting functions.

        Returns:
            Matplotlib figure or Plotly figure depending on interactive parameter.
        """
        data = self.mint.crosstab(var_name=var_name)
        if data is None or len(data) == 0:
            return None

        # Filter to specified peak labels
        if peak_labels is not None and len(peak_labels) > 0:
            available_cols = [c for c in peak_labels if c in data.columns]
            data = data[available_cols]

        # Apply transform
        if apply == "log2p1":
            data = np.log2(data + 1)
        elif apply == "log10p1":
            data = np.log10(data + 1)

        # Melt to long format
        df_long = data.reset_index().melt(
            id_vars=data.index.name or "index",
            var_name="peak_label",
            value_name="value"
        )
        df_long = df_long.rename(columns={data.index.name or "index": "ms_file"})

        # Use target order for peak_label (Categorical preserves order in plots)
        peak_order = list(self.mint.peak_labels) if self.mint.targets is not None else None
        if peak_order:
            df_long["peak_label"] = pd.Categorical(
                df_long["peak_label"], categories=peak_order, ordered=True
            )

        # Add metadata columns needed for hue, col, or row
        meta_cols_needed = set()
        for var in [hue, col, row]:
            if var and var not in ["peak_label", "ms_file"] and self.mint.meta is not None:
                if var in self.mint.meta.columns:
                    meta_cols_needed.add(var)

        if meta_cols_needed and self.mint.meta is not None:
            df_long = df_long.merge(
                self.mint.meta[list(meta_cols_needed)], left_on="ms_file", right_index=True, how="left"
            )

        if interactive:
            return self._distribution_plotly(df_long, var_name, apply, style, hue, **kwargs)
        else:
            return self._distribution_matplotlib(
                df_long, var_name, apply, style, hue, col, row, col_wrap, x_label, y_label, **kwargs
            )

    def _get_distribution_labels(
        self,
        hue: str | None,
        x_label: str | None,
        y_label: str | None,
        var_name: str,
        apply: str | None,
    ) -> tuple[str, str, str]:
        """Determine x-column and labels for distribution plots.

        Args:
            hue: Hue variable for grouping.
            x_label: Custom x-axis label or None.
            y_label: Custom y-axis label or None.
            var_name: Variable being plotted.
            apply: Transformation applied.

        Returns:
            Tuple of (x_col, x_label, y_label).
        """
        if y_label is None:
            y_label = f"{var_name} ({apply})" if apply else var_name

        if hue == "peak_label":
            x_col = "ms_file"
            if x_label is None:
                x_label = "Sample"
        else:
            x_col = "peak_label"
            if x_label is None:
                x_label = "Target"

        return x_col, x_label, y_label

    def _histogram_matplotlib(
        self,
        df_long: pd.DataFrame,
        var_name: str,
        y_label: str,
        hue: str | None,
        col: str | None,
        row: str | None,
        col_wrap: int | None,
        height: float,
        aspect: float,
    ) -> matplotlib.figure.Figure:
        """Create matplotlib histogram plot."""
        use_facet = col is not None or row is not None

        if use_facet:
            g = sns.FacetGrid(
                df_long, col=col, row=row, col_wrap=col_wrap if row is None else None,
                height=height, aspect=aspect, sharex=True, sharey=True
            )
            g.map(plt.hist, "value", bins=50, alpha=0.7, color="steelblue")
            g.set_axis_labels(y_label, "Count")
            return g.fig

        fig, ax = plt.subplots(figsize=(10, 6))
        if hue and hue in df_long.columns:
            for group in df_long[hue].dropna().unique():
                subset = df_long[df_long[hue] == group]["value"].dropna()
                ax.hist(subset, bins=50, alpha=0.5, label=str(group))
            ax.legend(title=hue)
        else:
            ax.hist(df_long["value"].dropna(), bins=50, alpha=0.7, color="steelblue")
        ax.set_xlabel(y_label)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {var_name}")
        plt.tight_layout()
        return fig

    def _catplot_matplotlib(
        self,
        df_long: pd.DataFrame,
        style: str,
        x_col: str,
        x_label: str,
        y_label: str,
        hue: str | None,
        col: str | None,
        row: str | None,
        col_wrap: int | None,
        height: float,
        aspect: float,
        figsize: tuple[int, int],
        var_name: str,
    ) -> matplotlib.figure.Figure:
        """Create matplotlib boxplot or violin plot."""
        kind_map = {"boxplot": "box", "violin": "violin"}
        kind = kind_map.get(style, "box")
        use_facet = col is not None or row is not None

        if use_facet:
            g = sns.catplot(
                data=df_long,
                x=x_col,
                y="value",
                hue=hue if hue and hue != x_col else None,
                col=col,
                row=row,
                col_wrap=col_wrap if row is None else None,
                kind=kind,
                height=height,
                aspect=aspect,
                sharex=False,
                sharey=True,
            )
            g.set_axis_labels(x_label, y_label)
            for ax in g.axes.flatten():
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha("right")
                    label.set_fontsize(8)
            g.tight_layout()
            return g.fig

        fig, ax = plt.subplots(figsize=figsize)
        hue_arg = hue if hue and hue != x_col else None
        if style == "boxplot":
            sns.boxplot(data=df_long, x=x_col, y="value", hue=hue_arg, ax=ax)
        else:  # violin
            sns.violinplot(data=df_long, x=x_col, y="value", hue=hue_arg, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"Distribution of {var_name} by {x_label.lower()}")
        plt.tight_layout()
        return fig

    def _distribution_matplotlib(
        self,
        df_long: pd.DataFrame,
        var_name: str,
        apply: str | None,
        style: str,
        hue: str | None,
        col: str | None = None,
        row: str | None = None,
        col_wrap: int | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        **kwargs,
    ) -> matplotlib.figure.Figure:
        """Create matplotlib distribution plot with optional faceting."""
        figsize = kwargs.get("figsize", (12, 6))
        height = kwargs.get("height", 4)
        aspect = kwargs.get("aspect", 1.5)

        x_col, x_label, y_label = self._get_distribution_labels(
            hue, x_label, y_label, var_name, apply
        )

        if style == "histogram":
            return self._histogram_matplotlib(
                df_long, var_name, y_label, hue, col, row, col_wrap, height, aspect
            )

        return self._catplot_matplotlib(
            df_long, style, x_col, x_label, y_label, hue,
            col, row, col_wrap, height, aspect, figsize, var_name
        )

    def _distribution_plotly(
        self,
        df_long: pd.DataFrame,
        var_name: str,
        apply: str | None,
        style: str,
        hue: str | None,
        **kwargs,
    ) -> PlotlyFigure:
        """Create plotly distribution plot."""
        y_label = f"{var_name} ({apply})" if apply else var_name

        # Determine x-axis: if hue is peak_label, use ms_file as x
        if hue == "peak_label":
            x_col = "ms_file"
            x_label = "Sample"
        else:
            x_col = "peak_label"
            x_label = "Target"

        if style == "histogram":
            fig = px.histogram(
                df_long,
                x="value",
                color=hue,
                title=f"Distribution of {var_name}",
                labels={"value": y_label},
                **kwargs,
            )

        elif style == "boxplot":
            fig = px.box(
                df_long,
                x=x_col,
                y="value",
                color=hue if hue != x_col else None,
                title=f"Distribution of {var_name} by {x_label.lower()}",
                labels={x_col: x_label, "value": y_label},
                **kwargs,
            )

        elif style == "violin":
            fig = px.violin(
                df_long,
                x=x_col,
                y="value",
                color=hue if hue != x_col else None,
                title=f"Distribution of {var_name} by {x_label.lower()}",
                labels={x_col: x_label, "value": y_label},
                **kwargs,
            )

        else:
            raise ValueError(f"Unknown style: {style}")

        fig.update_layout(autosize=True)
        return fig
