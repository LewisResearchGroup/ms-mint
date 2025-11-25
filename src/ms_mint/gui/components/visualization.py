"""Visualization panel component with various plot types."""

from __future__ import annotations

import io
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import solara

if TYPE_CHECKING:
    from ms_mint.Mint import Mint


@contextmanager
def no_display():
    """Context manager to suppress matplotlib auto-display."""
    was_interactive = plt.isinteractive()
    plt.ioff()
    try:
        yield
    finally:
        if was_interactive:
            plt.ion()


def save_figure_to_file(
    fig,
    mint: "Mint",
    plot_name: str,
    settings: dict,
    dpi: int = 150,
) -> str:
    """Save figure to analyses/figures/ with readme.

    Args:
        fig: Matplotlib figure (or seaborn grid with .fig attribute)
        mint: Mint instance for wdir
        plot_name: Name for the figure file
        settings: Dict of settings to write to readme
        dpi: Resolution for saving

    Returns:
        Path to saved figure or error message
    """
    try:
        # Handle seaborn grid objects
        if hasattr(fig, 'fig'):
            actual_fig = fig.fig
        elif hasattr(fig, 'figure'):
            actual_fig = fig.figure
        else:
            actual_fig = fig

        # Create output directory
        date_str = datetime.now().strftime("%y%m%d")
        out_dir = mint.wdir / "analyses" / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        fig_name = f"{date_str}-{plot_name}"
        fig_path = out_dir / f"{fig_name}.png"
        readme_path = out_dir / f"{fig_name}.md"

        # Save figure
        actual_fig.savefig(fig_path, format='png', dpi=dpi, bbox_inches='tight')

        # Write readme
        settings_str = "\n".join(f"- **{k}:** {v}" for k, v in settings.items())
        readme_content = f"""# {plot_name.replace('_', ' ').title()}

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Settings
{settings_str}

## Output
- `{fig_name}.png`
"""
        readme_path.write_text(readme_content)
        return f"Saved: {fig_path.name}"
    except Exception as e:
        return f"Error: {e}"


@solara.component
def PCAPlot(mint: "Mint", plot_type: str, interactive: bool, n_components: int,
            var_name: str, apply: str, scaler: str, fillna: str, color_by: str = "none",
            x_component: int = 1, y_component: int = 2):
    """Render PCA plot."""
    save_message = solara.use_reactive("")

    # Convert "none" to None for apply parameter
    apply_val = apply if apply != "none" else None

    def save_figure():
        with no_display():
            mint.pca.run(n_components=n_components, var_name=var_name, apply=apply_val, scaler=scaler, fillna=fillna)
            if plot_type == "scatter":
                fig = mint.pca.plot.scatter(x_component=x_component, y_component=y_component,
                                            color_by=color_by if color_by != "none" else None, interactive=False)
            elif plot_type == "cumulative_variance":
                fig = mint.pca.plot.cumulative_variance(interactive=False)
            elif plot_type == "pairplot":
                fig = mint.pca.plot.pairplot(interactive=False, hue=color_by if color_by != "none" else None)
            elif plot_type == "loadings":
                fig = mint.pca.plot.loadings(interactive=False)
            else:
                save_message.set("Unknown plot type")
                return

            if fig is not None:
                settings = {
                    "Plot type": plot_type,
                    "Variable": var_name,
                    "Transform": apply,
                    "Components": n_components,
                    "Scaler": scaler,
                    "Fill NA": fillna,
                    "Color by": color_by,
                }
                if plot_type == "scatter":
                    settings["X component"] = x_component
                    settings["Y component"] = y_component
                msg = save_figure_to_file(fig, mint, f"pca_{plot_type}-{var_name}", settings)
                save_message.set(msg)
                plt.close('all')

    try:
        with no_display():
            # Run PCA with current settings
            mint.pca.run(
                n_components=n_components,
                var_name=var_name,
                apply=apply_val,
                scaler=scaler,
                fillna=fillna,
            )

            # Generate the requested plot
            if plot_type == "scatter":
                fig = mint.pca.plot.scatter(
                    x_component=x_component,
                    y_component=y_component,
                    color_by=color_by if color_by != "none" else None,
                    interactive=interactive,
                )
            elif plot_type == "cumulative_variance":
                fig = mint.pca.plot.cumulative_variance(interactive=interactive)
            elif plot_type == "pairplot":
                hue = color_by if color_by != "none" else None
                fig = mint.pca.plot.pairplot(interactive=interactive, hue=hue)
            elif plot_type == "loadings":
                fig = mint.pca.plot.loadings(interactive=interactive)
            else:
                solara.Warning(f"Unknown PCA plot type: {plot_type}")
                return

            if fig is not None:
                if interactive:
                    solara.FigurePlotly(fig)
                else:
                    # Handle seaborn grid objects (PairGrid, FacetGrid, etc.)
                    if hasattr(fig, 'fig'):
                        actual_fig = fig.fig
                    elif hasattr(fig, 'figure'):
                        actual_fig = fig.figure
                    else:
                        actual_fig = fig
                    solara.FigureMatplotlib(actual_fig, dependencies=[plot_type, n_components, var_name, apply, scaler, fillna, color_by, x_component, y_component])
                    plt.close(actual_fig)

                with solara.Row():
                    solara.Button("Save Figure", on_click=save_figure, color="primary")
                    if save_message.value:
                        solara.Text(save_message.value, style={"fontSize": "12px"})
            else:
                solara.Warning("Could not generate PCA plot")
    except Exception as e:
        solara.Error(f"PCA error: {e}")




@solara.component
def PeakShapesPlot(mint: "Mint", interactive: bool):
    """Render peak shapes plot."""
    save_message = solara.use_reactive("")

    def save_figure():
        with no_display():
            fig = mint.plot.peak_shapes(interactive=False)
            if fig is not None:
                settings = {"Interactive": False}
                msg = save_figure_to_file(fig, mint, "peak_shapes", settings)
                save_message.set(msg)
                plt.close('all')

    try:
        with no_display():
            fig = mint.plot.peak_shapes(interactive=interactive)
            if fig is not None:
                if interactive:
                    solara.FigurePlotly(fig)
                else:
                    # Handle seaborn grid objects (FacetGrid, etc.)
                    if hasattr(fig, 'fig'):
                        actual_fig = fig.fig
                    elif hasattr(fig, 'figure'):
                        actual_fig = fig.figure
                    else:
                        actual_fig = fig
                    solara.FigureMatplotlib(actual_fig)
                    plt.close(actual_fig)

                with solara.Row():
                    solara.Button("Save Figure", on_click=save_figure, color="primary")
                    if save_message.value:
                        solara.Text(save_message.value, style={"fontSize": "12px"})
            else:
                solara.Warning("No peak shape data available")
    except Exception as e:
        solara.Error(f"Peak shapes error: {e}")


@solara.component
def ChromatogramPlot(mint: "Mint", interactive: bool, height: float = 1.5, aspect: float = 5,
                     rt_unit: str = "seconds", peak_labels: list = None):
    """Render chromatogram plot.

    Args:
        mint: Mint instance
        interactive: Whether to use plotly
        height: Height of each subplot
        aspect: Width/height ratio
        rt_unit: Display unit for RT ("seconds" or "minutes")
        peak_labels: List of targets to plot (None or empty = all)
    """
    save_message = solara.use_reactive("")

    rt_label = "min" if rt_unit == "minutes" else "s"
    # Use all targets if none specified
    targets = peak_labels if peak_labels else None

    def save_figure():
        from matplotlib.ticker import FuncFormatter
        with no_display():
            fig = mint.plot.chromatogram(interactive=False, height=height, aspect=aspect, peak_labels=targets)
            if fig is not None:
                # Convert x-axis labels to minutes if needed
                if rt_unit == "minutes":
                    if hasattr(fig, 'fig'):
                        actual_fig = fig.fig
                    elif hasattr(fig, 'figure'):
                        actual_fig = fig.figure
                    else:
                        actual_fig = fig
                    for ax in actual_fig.axes:
                        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x/60:.1f}"))
                        ax.set_xlabel(f"RT ({rt_label})")
                    fig = actual_fig
                settings = {"Height": height, "Aspect": aspect, "RT unit": rt_unit, "Targets": len(targets) if targets else "all"}
                msg = save_figure_to_file(fig, mint, "chromatogram", settings)
                save_message.set(msg)
                plt.close('all')

    try:
        with no_display():
            fig = mint.plot.chromatogram(interactive=interactive, height=height, aspect=aspect, peak_labels=targets)
            if fig is not None:
                if interactive:
                    # For plotly, update x-axis labels if minutes
                    if rt_unit == "minutes":
                        fig.update_xaxes(title_text=f"RT ({rt_label})")
                        # Scale the data in each trace
                        for trace in fig.data:
                            if hasattr(trace, 'x') and trace.x is not None:
                                trace.x = tuple(x / 60 for x in trace.x)
                    solara.FigurePlotly(fig)
                else:
                    # Handle seaborn grid objects (FacetGrid, etc.)
                    if hasattr(fig, 'fig'):
                        actual_fig = fig.fig
                    elif hasattr(fig, 'figure'):
                        actual_fig = fig.figure
                    else:
                        actual_fig = fig

                    # Convert x-axis labels to minutes if needed (keep data in seconds)
                    if rt_unit == "minutes":
                        from matplotlib.ticker import FuncFormatter
                        for ax in actual_fig.axes:
                            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x/60:.1f}"))
                            ax.set_xlabel(f"RT ({rt_label})")

                    solara.FigureMatplotlib(actual_fig, dependencies=[height, aspect, rt_unit, str(peak_labels)])
                    plt.close(actual_fig)

                with solara.Row():
                    solara.Button("Save Figure", on_click=save_figure, color="primary")
                    if save_message.value:
                        solara.Text(save_message.value, style={"fontSize": "12px"})
            else:
                solara.Warning("No chromatogram data available")
    except Exception as e:
        solara.Error(f"Chromatogram error: {e}")


@solara.component
def DistributionPlot(mint: "Mint", var_name: str = "peak_max", apply: str = "log2p1",
                     plot_style: str = "boxplot", color_by: str = "none", interactive: bool = False):
    """Render distribution plots (histogram, boxplot, violin).

    Args:
        mint: Mint instance
        var_name: Variable to plot
        apply: Transform to apply
        plot_style: One of "histogram", "boxplot", "violin"
        color_by: Metadata column for grouping
        interactive: Whether to use plotly
    """
    save_message = solara.use_reactive("")

    # Convert "none" to None for hue
    hue = color_by if color_by != "none" else None
    apply_val = apply if apply != "none" else None

    def save_figure():
        with no_display():
            fig = mint.plot.distribution(
                var_name=var_name,
                apply=apply_val,
                style=plot_style,
                hue=hue,
                interactive=False,
            )
            if fig is not None:
                settings = {
                    "Variable": var_name,
                    "Transform": apply,
                    "Style": plot_style,
                    "Color by": color_by,
                }
                msg = save_figure_to_file(fig, mint, f"distribution_{plot_style}-{var_name}", settings)
                save_message.set(msg)
                plt.close('all')

    try:
        with no_display():
            fig = mint.plot.distribution(
                var_name=var_name,
                apply=apply_val,
                style=plot_style,
                hue=hue,
                interactive=interactive,
            )
            if fig is None:
                solara.Warning("No data for distribution plot")
                return

            if interactive:
                solara.FigurePlotly(fig)
            else:
                solara.FigureMatplotlib(fig, dependencies=[var_name, apply, plot_style, color_by])
                plt.close(fig)

            with solara.Row():
                solara.Button("Save Figure", on_click=save_figure, color="primary")
                if save_message.value:
                    solara.Text(save_message.value, style={"fontSize": "12px"})
    except Exception as e:
        solara.Error(f"Distribution error: {e}")


def _create_heatmap_figure(mint: "Mint", var_name: str, apply: str, scaler: str, transposed: bool):
    """Helper to create heatmap figure."""
    import seaborn as sns
    import numpy as np

    data = mint.crosstab(var_name=var_name)
    if data is None or len(data) == 0:
        return None

    row_labels = list(data.index)
    col_labels = list(data.columns)

    if apply == "log2p1":
        data = np.log2(data + 1)
    elif apply == "log10p1":
        data = np.log10(data + 1)

    if scaler == "standard":
        data = (data - data.mean()) / data.std()
    elif scaler == "robust":
        median = data.median()
        iqr = data.quantile(0.75) - data.quantile(0.25)
        data = (data - median) / iqr
    elif scaler == "minmax":
        data = (data - data.min()) / (data.max() - data.min())

    if transposed:
        data = data.T
        row_labels, col_labels = col_labels, row_labels

    n_rows, n_cols = data.shape
    figsize = (max(12, n_cols * 0.6), max(8, n_rows * 0.4))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data, ax=ax, cmap="RdBu_r", center=0,
               xticklabels=col_labels, yticklabels=row_labels)
    ax.set_title(f"{var_name} ({apply}, {scaler})")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xlabel("Targets")
    ax.set_ylabel("Samples")
    plt.tight_layout()
    return fig


@solara.component
def SimpleHeatmapPlot(mint: "Mint", var_name: str = "peak_max",
                      apply: str = "log2p1", scaler: str = "standard",
                      transposed: bool = False):
    """Render simple heatmap (seaborn heatmap without clustering)."""
    save_message = solara.use_reactive("")

    def save_figure():
        with no_display():
            fig = _create_heatmap_figure(mint, var_name, apply, scaler, transposed)
            if fig is not None:
                settings = {
                    "Variable": var_name,
                    "Transform": apply,
                    "Scaler": scaler,
                    "Transposed": transposed,
                }
                msg = save_figure_to_file(fig, mint, f"heatmap-{var_name}", settings)
                save_message.set(msg)
                plt.close('all')

    try:
        with no_display():
            fig = _create_heatmap_figure(mint, var_name, apply, scaler, transposed)
            if fig is None:
                solara.Warning("No data for heatmap")
                return

            solara.FigureMatplotlib(fig, dependencies=[var_name, apply, scaler, transposed])
            plt.close(fig)

            with solara.Row():
                solara.Button("Save Figure", on_click=save_figure, color="primary")
                if save_message.value:
                    solara.Text(save_message.value, style={"fontSize": "12px"})
    except Exception as e:
        solara.Error(f"Heatmap error: {e}")


def _create_clustering_figure(mint: "Mint", var_name: str, apply: str, scaler: str,
                               metric: str, transposed: bool, width: int, height: int):
    """Helper to create hierarchical clustering figure."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*")
        data = mint.crosstab(var_name=var_name)
        if data is None or len(data) == 0:
            return None

        n_rows, n_cols = data.shape
        if transposed:
            n_rows, n_cols = n_cols, n_rows

        fig = mint.plot.hierarchical_clustering(
            var_name=var_name,
            apply=apply if apply != "none" else None,
            scaler=scaler,
            metric=metric,
            transposed=transposed,
            figsize=(width, height),
            xmaxticks=n_cols,
            ymaxticks=n_rows,
        )
        return fig


@solara.component
def HierarchicalClusteringPlot(mint: "Mint", var_name: str = "peak_max",
                                apply: str = "log2p1", scaler: str = "standard",
                                metric: str = "cosine", transposed: bool = False,
                                width: int = 12, height: int = 10, dpi: int = 150):
    """Render hierarchical clustering heatmap (seaborn clustermap)."""
    save_message = solara.use_reactive("")

    def save_figure():
        with no_display():
            fig = _create_clustering_figure(mint, var_name, apply, scaler, metric, transposed, width, height)
            if fig is not None:
                settings = {
                    "Variable": var_name,
                    "Transform": apply,
                    "Scaler": scaler,
                    "Metric": metric,
                    "Transposed": transposed,
                    "Figure size": f"{width} x {height}",
                    "DPI": dpi,
                }
                msg = save_figure_to_file(fig, mint, f"hierarchical_clustering-{var_name}", settings, dpi=dpi)
                save_message.set(msg)
                plt.close('all')

    try:
        with no_display():
            fig = _create_clustering_figure(mint, var_name, apply, scaler, metric, transposed, width, height)
            if fig is None:
                solara.Warning("No data for clustering")
                return

            if hasattr(fig, 'fig'):
                fig = fig.fig
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)

            solara.Image(buf.getvalue())
            with solara.Row():
                solara.Button("Save Figure", on_click=save_figure, color="primary")
                if save_message.value:
                    solara.Text(save_message.value, style={"fontSize": "12px"})
    except Exception as e:
        solara.Error(f"Clustering error: {e}")


@solara.component
def VisualizationPanel(
    mint: "Mint",
    results: solara.Reactive[pd.DataFrame],
    rt_unit: solara.Reactive[str] = None,
):
    """Component for interactive visualizations.

    Provides access to various visualization methods from MintPlotter.

    Args:
        mint: The Mint instance with plotting methods.
        results: Reactive DataFrame of results.
        rt_unit: Reactive string for RT display unit ("seconds" or "minutes").
    """
    # Fallback for rt_unit
    fallback_rt_unit = solara.use_reactive("seconds")
    active_rt_unit = rt_unit if rt_unit is not None else fallback_rt_unit
    plot_type = solara.use_reactive("heatmap")

    # Heatmap settings (simple)
    heatmap_var = solara.use_reactive("peak_max")
    heatmap_scaler = solara.use_reactive("standard")
    heatmap_apply = solara.use_reactive("log2p1")
    heatmap_transposed = solara.use_reactive(False)

    # Hierarchical clustering settings
    clust_var = solara.use_reactive("peak_max")
    clust_scaler = solara.use_reactive("standard")
    clust_apply = solara.use_reactive("log2p1")
    clust_metric = solara.use_reactive("cosine")
    clust_transposed = solara.use_reactive(False)
    clust_width = solara.use_reactive(12)
    clust_height = solara.use_reactive(10)
    clust_dpi = solara.use_reactive(150)

    # Peak shapes settings
    peak_shapes_interactive = solara.use_reactive(False)

    # Chromatogram settings
    chrom_interactive = solara.use_reactive(False)
    chrom_height = solara.use_reactive(2.0)
    chrom_aspect = solara.use_reactive(5.0)
    chrom_targets = solara.use_reactive([])  # Empty means all

    # Distribution plot settings
    dist_var = solara.use_reactive("peak_max")
    dist_apply = solara.use_reactive("log2p1")
    dist_style = solara.use_reactive("boxplot")
    dist_color_by = solara.use_reactive("none")
    dist_interactive = solara.use_reactive(False)

    # PCA settings
    pca_plot_type = solara.use_reactive("scatter")  # scatter is the default (PC1 vs PC2)
    pca_interactive = solara.use_reactive(False)
    pca_n_components = solara.use_reactive(3)
    pca_var_name = solara.use_reactive("peak_max")
    pca_apply = solara.use_reactive("log2p1")
    pca_scaler = solara.use_reactive("standard")
    pca_fillna = solara.use_reactive("median")
    pca_color_by = solara.use_reactive("none")
    pca_x_component = solara.use_reactive(1)
    pca_y_component = solara.use_reactive(2)

    # Get available metadata columns for color_by options
    metadata_columns = ["none"]
    if mint.meta is not None and len(mint.meta) > 0:
        # Only include columns that have at least some non-null values
        valid_cols = mint.meta.dropna(axis=1, how="all").columns.tolist()
        metadata_columns = ["none"] + valid_cols

    # Get available peak labels for target selection
    available_targets = list(mint.peak_labels) if mint.targets is not None else []

    if len(results.value) == 0:
        solara.Info("No results to visualize. Run processing first.")
        return

    with solara.Column():
        # Plot type selector
        with solara.Row():
            solara.Select(
                label="Plot Type",
                value=plot_type.value,
                values=[
                    "heatmap",
                    "hierarchical_clustering",
                    "distribution",
                    "peak_shapes",
                    "chromatogram",
                    "pca",
                ],
                on_value=plot_type.set,
            )

            # Plot-specific inline settings
            if plot_type.value == "pca":
                solara.Select(
                    label="PCA Plot",
                    value=pca_plot_type.value,
                    values=["scatter", "cumulative_variance", "pairplot", "loadings"],
                    on_value=pca_plot_type.set,
                )

        # Additional settings per plot type
        if plot_type.value == "heatmap":
            with solara.Row():
                solara.Select(
                    label="Variable",
                    value=heatmap_var.value,
                    values=["peak_max", "peak_area", "peak_area_top3", "peak_mean"],
                    on_value=heatmap_var.set,
                )
                solara.Select(
                    label="Transform",
                    value=heatmap_apply.value,
                    values=["log2p1", "log10p1", "none"],
                    on_value=heatmap_apply.set,
                )
                solara.Select(
                    label="Scaler",
                    value=heatmap_scaler.value,
                    values=["standard", "robust", "minmax", "none"],
                    on_value=heatmap_scaler.set,
                )
                solara.Checkbox(label="Transposed", value=heatmap_transposed)

        elif plot_type.value == "hierarchical_clustering":
            with solara.Row():
                solara.Select(
                    label="Variable",
                    value=clust_var.value,
                    values=["peak_max", "peak_area", "peak_area_top3", "peak_mean"],
                    on_value=clust_var.set,
                )
                solara.Select(
                    label="Transform",
                    value=clust_apply.value,
                    values=["log2p1", "log10p1", "none"],
                    on_value=clust_apply.set,
                )
                solara.Select(
                    label="Scaler",
                    value=clust_scaler.value,
                    values=["standard", "robust", "minmax"],
                    on_value=clust_scaler.set,
                )
            with solara.Row():
                solara.Select(
                    label="Metric",
                    value=clust_metric.value,
                    values=["cosine", "euclidean", "correlation"],
                    on_value=clust_metric.set,
                )
                solara.Checkbox(label="Transposed", value=clust_transposed)
            with solara.Row():
                solara.SliderInt(label="Width", value=clust_width, min=6, max=24)
                solara.SliderInt(label="Height", value=clust_height, min=6, max=24)
                solara.SliderInt(label="DPI", value=clust_dpi, min=72, max=300, step=10)

        elif plot_type.value == "distribution":
            # Color options: none, peak_label, or metadata columns
            dist_color_options = ["none", "peak_label"] + metadata_columns[1:]  # Skip "none" from metadata_columns
            with solara.Row():
                solara.Select(
                    label="Variable",
                    value=dist_var.value,
                    values=["peak_max", "peak_area", "peak_area_top3", "peak_mean"],
                    on_value=dist_var.set,
                )
                solara.Select(
                    label="Transform",
                    value=dist_apply.value,
                    values=["log2p1", "log10p1", "none"],
                    on_value=dist_apply.set,
                )
                solara.Select(
                    label="Style",
                    value=dist_style.value,
                    values=["boxplot", "violin", "histogram"],
                    on_value=dist_style.set,
                )
                solara.Select(
                    label="Color by",
                    value=dist_color_by.value,
                    values=dist_color_options,
                    on_value=dist_color_by.set,
                )

        elif plot_type.value == "chromatogram":
            with solara.Row():
                solara.SliderFloat(label="Height", value=chrom_height, min=1.0, max=5.0, step=0.5)
                solara.SliderFloat(label="Aspect", value=chrom_aspect, min=2.0, max=10.0, step=0.5)
                solara.Select(
                    label="RT Unit",
                    value=active_rt_unit.value,
                    values=["seconds", "minutes"],
                    on_value=active_rt_unit.set,
                )
            if available_targets:
                solara.SelectMultiple(
                    label="Targets (empty = all)",
                    values=chrom_targets,
                    all_values=available_targets,
                )

        elif plot_type.value == "pca":
            with solara.Row():
                solara.SliderInt(label="Components", value=pca_n_components, min=2, max=10)
                solara.Select(
                    label="Variable",
                    value=pca_var_name.value,
                    values=["peak_max", "peak_area", "peak_area_top3", "peak_mean"],
                    on_value=pca_var_name.set,
                )
                solara.Select(
                    label="Transform",
                    value=pca_apply.value,
                    values=["log2p1", "log10p1", "none"],
                    on_value=pca_apply.set,
                )
            with solara.Row():
                solara.Select(
                    label="Scaler",
                    value=pca_scaler.value,
                    values=["standard", "robust", "minmax"],
                    on_value=pca_scaler.set,
                )
                solara.Select(
                    label="Fill NA",
                    value=pca_fillna.value,
                    values=["median", "mean", "zero"],
                    on_value=pca_fillna.set,
                )
            # Color by metadata (for scatter and pairplot)
            if pca_plot_type.value in ["scatter", "pairplot"]:
                with solara.Row():
                    solara.Select(
                        label="Color by",
                        value=pca_color_by.value,
                        values=metadata_columns,
                        on_value=pca_color_by.set,
                    )
                    # Component selectors for scatter plot
                    if pca_plot_type.value == "scatter":
                        component_options = [i for i in range(1, pca_n_components.value + 1)]
                        solara.Select(
                            label="X axis",
                            value=pca_x_component.value,
                            values=component_options,
                            on_value=pca_x_component.set,
                        )
                        solara.Select(
                            label="Y axis",
                            value=pca_y_component.value,
                            values=component_options,
                            on_value=pca_y_component.set,
                        )

        # Render the plot
        solara.Markdown("---")

        if plot_type.value == "heatmap":
            SimpleHeatmapPlot(
                mint=mint,
                var_name=heatmap_var.value,
                apply=heatmap_apply.value,
                scaler=heatmap_scaler.value,
                transposed=heatmap_transposed.value,
            )
        elif plot_type.value == "hierarchical_clustering":
            HierarchicalClusteringPlot(
                mint=mint,
                var_name=clust_var.value,
                apply=clust_apply.value,
                scaler=clust_scaler.value,
                metric=clust_metric.value,
                transposed=clust_transposed.value,
                width=clust_width.value,
                height=clust_height.value,
                dpi=clust_dpi.value,
            )
        elif plot_type.value == "distribution":
            DistributionPlot(
                mint=mint,
                var_name=dist_var.value,
                apply=dist_apply.value,
                plot_style=dist_style.value,
                color_by=dist_color_by.value,
                interactive=dist_interactive.value,
            )
        elif plot_type.value == "peak_shapes":
            PeakShapesPlot(mint=mint, interactive=peak_shapes_interactive.value)
        elif plot_type.value == "chromatogram":
            ChromatogramPlot(
                mint=mint,
                interactive=chrom_interactive.value,
                height=chrom_height.value,
                aspect=chrom_aspect.value,
                rt_unit=active_rt_unit.value,
                peak_labels=chrom_targets.value if chrom_targets.value else None,
            )
        elif plot_type.value == "pca":
            PCAPlot(
                mint=mint,
                plot_type=pca_plot_type.value,
                interactive=pca_interactive.value,
                n_components=pca_n_components.value,
                var_name=pca_var_name.value,
                apply=pca_apply.value,
                scaler=pca_scaler.value,
                fillna=pca_fillna.value,
                color_by=pca_color_by.value,
                x_component=pca_x_component.value,
                y_component=pca_y_component.value,
            )
