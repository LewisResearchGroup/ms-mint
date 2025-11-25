"""Optimization panel component for target retention time optimization."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable

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


@solara.component
def OptimizationPanel(
    mint: "Mint",
    ms_files: solara.Reactive[list[str]],
    targets: solara.Reactive[pd.DataFrame],
    on_targets_updated: Callable[[], None],
    rt_unit: solara.Reactive[str] = None,
):
    """Component for target optimization with peak preview.

    Args:
        mint: The Mint instance with optimization methods.
        ms_files: Reactive list of loaded MS files.
        targets: Reactive DataFrame of targets.
        on_targets_updated: Callback when targets are updated after optimization.
        rt_unit: Reactive string for RT display unit ("seconds" or "minutes").
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
    manual_rt_min = solara.use_reactive(0.0)
    manual_rt_max = solara.use_reactive(0.0)
    manual_rt = solara.use_reactive(0.0)

    # Counter to force refresh after optimization
    refresh_counter = solara.use_reactive(0)

    # Batch optimization parameters
    minimum_intensity = solara.use_reactive(1e4)
    sigma = solara.use_reactive(20.0)
    rel_height = solara.use_reactive(0.9)

    # Effect to reload values when rt_unit changes (must be before early returns)
    def on_rt_unit_change():
        if selected_peak.value and mint.targets is not None:
            try:
                row = mint.targets.loc[selected_peak.value]
                manual_rt_min.set(to_display(float(row.get("rt_min", 0))))
                manual_rt_max.set(to_display(float(row.get("rt_max", 0))))
                manual_rt.set(to_display(float(row.get("rt", 0))))
            except KeyError:
                pass

    solara.use_effect(on_rt_unit_change, [active_rt_unit.value])

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
                manual_rt_min.set(to_display(float(row.get("rt_min", 0))))
                manual_rt_max.set(to_display(float(row.get("rt_max", 0))))
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
            mint.targets.loc[selected_peak.value, "rt_min"] = from_display(manual_rt_min.value)
            mint.targets.loc[selected_peak.value, "rt_max"] = from_display(manual_rt_max.value)
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
            # Increment counter to force plot refresh
            refresh_counter.set(refresh_counter.value + 1)
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
            # Increment counter to force plot refresh
            refresh_counter.set(refresh_counter.value + 1)
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
                    solara.Markdown(f"### RT Adjustment ({unit_label})")
                    solara.InputFloat(
                        label=f"RT expected ({unit_label})",
                        value=manual_rt,
                        on_value=manual_rt.set,
                    )
                    solara.InputFloat(
                        label=f"RT Min ({unit_label})",
                        value=manual_rt_min,
                        on_value=manual_rt_min.set,
                    )
                    solara.InputFloat(
                        label=f"RT Max ({unit_label})",
                        value=manual_rt_max,
                        on_value=manual_rt_max.set,
                    )

                    with solara.Row():
                        solara.Button(
                            "Apply Manual",
                            on_click=apply_manual_rt,
                            color="primary",
                        )
                        solara.Button(
                            "Auto-Optimize",
                            on_click=auto_optimize_peak,
                            color="secondary",
                        )

            # Right: Peak preview
            with solara.Column(style={"flex": "2"}):
                if selected_peak.value:
                    solara.Markdown(f"### Peak: {selected_peak.value}")
                    try:
                        with no_display():
                            fig = mint.plot.peak_shapes(
                                peak_labels=[selected_peak.value],
                                interactive=False,
                            )
                            if fig is not None:
                                if hasattr(fig, 'fig'):
                                    actual_fig = fig.fig
                                elif hasattr(fig, 'figure'):
                                    actual_fig = fig.figure
                                else:
                                    actual_fig = fig
                                solara.FigureMatplotlib(
                                    actual_fig,
                                    dependencies=[selected_peak.value, manual_rt_min.value, manual_rt_max.value, refresh_counter.value]
                                )
                                plt.close(actual_fig)
                            else:
                                solara.Warning("No data for this peak")
                    except Exception as e:
                        solara.Error(f"Preview error: {e}")
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
