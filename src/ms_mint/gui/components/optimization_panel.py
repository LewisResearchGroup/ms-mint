"""Optimization panel component for target retention time optimization."""

from __future__ import annotations

import io
import traceback
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns
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


def extract_all_chromatograms(
    mint: "Mint",
    nthreads: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Extract chromatogram data for ALL peaks at once (memoizable).

    This extracts data for all peaks so each file is only processed once.
    """
    ms_files_list = list(mint.ms_files)
    if not ms_files_list:
        return None
    peak_labels_list = list(mint.peak_labels) if mint.targets is not None else []
    if not peak_labels_list:
        return None
    return mint.get_chromatograms(
        fns=ms_files_list,
        peak_labels=peak_labels_list,
        nthreads=nthreads,
    )


@solara.component
def OptimizationPanel(
    mint: "Mint",
    ms_files: solara.Reactive[list[str]],
    targets: solara.Reactive[pd.DataFrame],
    on_targets_updated: Callable[[], None],
    rt_unit: solara.Reactive[str] = None,
    nthreads: solara.Reactive[int] = None,
):
    """Component for target optimization with peak preview.

    Args:
        mint: The Mint instance with optimization methods.
        ms_files: Reactive list of loaded MS files.
        targets: Reactive DataFrame of targets.
        on_targets_updated: Callback when targets are updated after optimization.
        rt_unit: Reactive string for RT display unit ("seconds" or "minutes").
        nthreads: Reactive int for number of threads for chromatogram extraction.
    """
    # Fallback reactive for rt_unit (always created to satisfy hooks rules)
    fallback_rt_unit = solara.use_reactive("seconds")
    # Use provided rt_unit or fallback
    active_rt_unit = rt_unit if rt_unit is not None else fallback_rt_unit

    # Conversion helpers
    def to_display(value_seconds: float) -> float:
        """Convert from internal seconds to display unit."""
        if active_rt_unit.value == "minutes":
            return value_seconds / 60.0
        return value_seconds

    def from_display(value_display: float) -> float:
        """Convert from display unit to internal seconds."""
        if active_rt_unit.value == "minutes":
            return value_display * 60.0
        return value_display

    unit_label = "min" if active_rt_unit.value == "minutes" else "s"
    error_message = solara.use_reactive("")
    success_message = solara.use_reactive("")

    # Selected peak for preview/editing
    selected_peak = solara.use_reactive("")

    # Manual RT adjustment values
    manual_rt_range = solara.use_reactive([0.0, 0.0])  # [min, max] tuple for range slider
    manual_rt = solara.use_reactive(0.0)

    # Zoom level for peak preview (margin around RT window in seconds)
    zoom_margin = solara.use_reactive(30.0)  # 30 seconds default margin

    # Y-axis zoom (percentage of max intensity to show, 100 = full range)
    y_zoom_percent = solara.use_reactive(100.0)

    # Batch optimization parameters
    minimum_intensity = solara.use_reactive(1e4)
    sigma = solara.use_reactive(20.0)
    rel_height = solara.use_reactive(0.9)

    # Effect to reload values when rt_unit changes (must be before early returns)
    def on_rt_unit_change():
        if selected_peak.value and mint.targets is not None:
            try:
                row = mint.targets.loc[selected_peak.value]
                rt_min_display = to_display(float(row.get("rt_min", 0)))
                rt_max_display = to_display(float(row.get("rt_max", 0)))
                manual_rt_range.set([rt_min_display, rt_max_display])
                manual_rt.set(to_display(float(row.get("rt", 0))))
            except KeyError:
                pass

    solara.use_effect(on_rt_unit_change, [active_rt_unit.value])

    # Memoize chromatogram data extraction for ALL peaks - must be before early returns (rules of hooks)
    # Re-extract when files or targets change (use lengths as proxy for change detection)
    n_files = len(ms_files.value)
    n_targets = len(targets.value)
    all_chrom_data = solara.use_memo(
        lambda: extract_all_chromatograms(mint, nthreads=4) if n_files > 0 and n_targets > 0 else None,
        dependencies=[n_files, n_targets],
    )

    if len(ms_files.value) == 0:
        solara.Info("Load MS files first for optimization.")
        return

    if len(targets.value) == 0:
        solara.Info("Load targets first for optimization.")
        return

    # Get peak labels
    peak_labels = list(mint.peak_labels) if mint.targets is not None else []

    def load_peak_values(peak_label: str):
        """Load RT values from mint.targets for the given peak."""
        if peak_label and mint.targets is not None:
            try:
                # Use .loc for direct index access
                row = mint.targets.loc[peak_label]
                # Convert to display units
                rt_min_display = to_display(float(row.get("rt_min", 0)))
                rt_max_display = to_display(float(row.get("rt_max", 0)))
                manual_rt_range.set([rt_min_display, rt_max_display])
                manual_rt.set(to_display(float(row.get("rt", 0))))
            except KeyError:
                pass  # Peak not found

    def on_peak_selected(peak_label: str):
        """Handle peak selection - load current RT values."""
        selected_peak.set(peak_label)
        load_peak_values(peak_label)
        error_message.set("")
        success_message.set("")

    def apply_manual_rt():
        """Apply manually adjusted RT values to the selected peak."""
        if not selected_peak.value:
            error_message.set("Select a peak first")
            return
        try:
            # Convert from display units back to seconds and update the target in mint
            rt_min_sec = from_display(manual_rt_range.value[0])
            rt_max_sec = from_display(manual_rt_range.value[1])
            mint.targets.loc[selected_peak.value, "rt_min"] = rt_min_sec
            mint.targets.loc[selected_peak.value, "rt_max"] = rt_max_sec
            mint.targets.loc[selected_peak.value, "rt"] = from_display(manual_rt.value)
            on_targets_updated()
            success_message.set(f"Updated RT for {selected_peak.value}")
            error_message.set("")
        except Exception as e:
            error_message.set(f"Error: {e}")
            success_message.set("")

    def auto_optimize_peak():
        """Auto-optimize RT for the selected peak only."""
        if not selected_peak.value:
            error_message.set("Select a peak first")
            return
        try:
            mint.opt.rt_min_max(
                peak_labels=[selected_peak.value],
                minimum_intensity=minimum_intensity.value,
                sigma=sigma.value,
                rel_height=rel_height.value,
            )
            # Reload the values from updated mint.targets
            load_peak_values(selected_peak.value)
            on_targets_updated()
            success_message.set(f"Optimized RT for {selected_peak.value}")
            error_message.set("")
        except Exception as e:
            error_message.set(f"Optimization error: {e}")
            success_message.set("")

    def batch_optimize():
        """Optimize all peaks."""
        try:
            mint.opt.rt_min_max(
                minimum_intensity=minimum_intensity.value,
                sigma=sigma.value,
                rel_height=rel_height.value,
            )
            on_targets_updated()
            # Reload selected peak values if one is selected
            if selected_peak.value:
                load_peak_values(selected_peak.value)
            success_message.set("Optimized all peaks")
            error_message.set("")
        except Exception as e:
            error_message.set(f"Batch optimization error: {e}")
            success_message.set("")

    with solara.Column():
        # Peak selector and preview side by side
        with solara.Row():
            # Left: Peak selector
            with solara.Column(style={"flex": "1", "min-width": "200px"}):
                solara.Markdown("### Select Peak")
                solara.Select(
                    label="Peak Label",
                    value=selected_peak.value,
                    values=[""] + peak_labels,
                    on_value=on_peak_selected,
                )

                if selected_peak.value:
                    # Get slider range based on target RT (with margin)
                    try:
                        row = mint.targets.loc[selected_peak.value]
                        rt_center = float(row.get("rt", 0))
                        # Set range to +/- 2 minutes around RT center (in display units)
                        margin = 120  # 2 minutes in seconds
                        slider_min = to_display(max(0, rt_center - margin))
                        slider_max = to_display(rt_center + margin)
                        step = 0.1 if active_rt_unit.value == "minutes" else 1.0
                    except:
                        slider_min = 0.0
                        slider_max = to_display(600)  # 10 min default
                        step = 0.1 if active_rt_unit.value == "minutes" else 1.0

                    solara.Markdown(f"### RT Adjustment ({unit_label})")

                    # RT expected slider
                    solara.SliderFloat(
                        label=f"RT expected ({unit_label})",
                        value=manual_rt,
                        min=slider_min,
                        max=slider_max,
                        step=step,
                    )

                    # RT window range slider
                    solara.SliderRangeFloat(
                        label=f"RT Window ({unit_label})",
                        value=manual_rt_range,
                        min=slider_min,
                        max=slider_max,
                        step=step,
                    )

                    with solara.Row():
                        solara.Button(
                            "Apply",
                            on_click=apply_manual_rt,
                            color="primary",
                        )
                        solara.Button(
                            "Auto-Optimize",
                            on_click=auto_optimize_peak,
                            color="secondary",
                        )

            # Right: Peak preview
            with solara.Column(style={"flex": "2", "min-width": "600px"}):
                if selected_peak.value:
                    with solara.Row():
                        solara.Markdown(f"### Peak: {selected_peak.value}")
                        # X-axis zoom slider
                        zoom_label = "min" if active_rt_unit.value == "minutes" else "s"
                        solara.SliderFloat(
                            label=f"RT margin ({zoom_label})",
                            value=zoom_margin,
                            min=5.0,
                            max=120.0,
                            step=5.0,
                        )
                        # Y-axis zoom slider
                        solara.SliderFloat(
                            label="Y max (%)",
                            value=y_zoom_percent,
                            min=1.0,
                            max=100.0,
                            step=1.0,
                        )
                    # Filter memoized chromatogram data for selected peak
                    peak_label_str = str(selected_peak.value)
                    chrom_data = None
                    if all_chrom_data is not None and len(all_chrom_data) > 0:
                        chrom_data = all_chrom_data[all_chrom_data["peak_label"] == peak_label_str]

                    if chrom_data is not None and len(chrom_data) > 0:
                        try:
                            with no_display():
                                # Calculate xlim based on current RT window + zoom margin
                                rt_min_sec = from_display(manual_rt_range.value[0])
                                rt_max_sec = from_display(manual_rt_range.value[1])
                                xlim = (
                                    max(0, rt_min_sec - zoom_margin.value),
                                    rt_max_sec + zoom_margin.value
                                )

                                # Create plot from cached data
                                fig = sns.relplot(
                                    data=chrom_data,
                                    x="scan_time",
                                    y="intensity",
                                    hue="ms_file_label",
                                    height=5,
                                    aspect=1.5,
                                    marker=".",
                                    linewidth=0,
                                )

                                # Apply zoom and add RT indicators
                                for ax in fig.axes.flatten():
                                    ax.set_xlim(xlim)

                                    # Apply Y-axis zoom (percentage of current max, always start from 0)
                                    y_min, y_max = ax.get_ylim()
                                    new_y_max = y_max * (y_zoom_percent.value / 100.0)
                                    ax.set_ylim(0, new_y_max)

                                    # Add green background for RT range
                                    ax.axvspan(rt_min_sec, rt_max_sec, alpha=0.2, color='green', zorder=0)

                                    # Add vertical line for expected RT
                                    rt_expected_sec = from_display(manual_rt.value)
                                    ax.axvline(rt_expected_sec, color='green', linestyle='--', linewidth=2, label='RT expected')

                                    # Update x-label for unit
                                    if active_rt_unit.value == "minutes":
                                        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x/60:.1f}"))
                                        ax.set_xlabel(f"RT ({zoom_label})")

                                plt.tight_layout()

                                # Render to PNG for full-width display
                                buf = io.BytesIO()
                                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                                buf.seek(0)
                                plt.close('all')

                                solara.Image(buf.read(), width="100%")
                        except Exception as e:
                            tb = traceback.format_exc()
                            solara.Error(f"Preview error: {e}\n\nTraceback:\n{tb}")
                    else:
                        solara.Warning("No data for this peak")
                else:
                    solara.Info("Select a peak to preview and adjust its retention time window.")

        # Messages
        if success_message.value:
            solara.Success(success_message.value)
        if error_message.value:
            solara.Error(error_message.value)

        # Batch optimization section
        solara.Markdown("---")
        with solara.Details("Batch Optimization (All Peaks)", expand=False):
            with solara.Column():
                solara.InputFloat(
                    label="Minimum intensity",
                    value=minimum_intensity,
                    on_value=minimum_intensity.set,
                )
                solara.InputFloat(
                    label="Sigma (Gaussian width, s)",
                    value=sigma,
                    on_value=sigma.set,
                )
                solara.InputFloat(
                    label="Relative height",
                    value=rel_height,
                    on_value=rel_height.set,
                )
                solara.Button(
                    "Optimize All Peaks",
                    on_click=batch_optimize,
                    color="primary",
                )
