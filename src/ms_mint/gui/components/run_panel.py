"""Run panel component with processing controls and progress bar."""

from __future__ import annotations

from collections.abc import Callable

import solara


@solara.component
def RunPanel(
    status: solara.Reactive[str],
    progress: solara.Reactive[float],
    nthreads: solara.Reactive[int | None],
    rt_margin: solara.Reactive[float],
    mode: solara.Reactive[str],
    can_run: bool,
    can_export: bool,
    on_run: Callable[[], None],
    on_export: Callable[[str], None],
):
    """Component for run controls, parameters, and progress display.

    Args:
        status: Reactive status string (waiting/running/done/error).
        progress: Reactive progress value (0-100).
        nthreads: Reactive number of threads.
        rt_margin: Reactive retention time margin.
        mode: Reactive processing mode.
        can_run: Whether run button should be enabled.
        can_export: Whether export button should be enabled.
        on_run: Callback to start processing.
        on_export: Callback to export results with filename.
    """
    export_filename = solara.use_reactive("results.xlsx")
    show_params = solara.use_reactive(False)

    def handle_run():
        """Handle run button click."""
        on_run()

    def handle_export():
        """Handle export button click."""
        on_export(export_filename.value)

    with solara.Card("Processing", margin=0):
        with solara.Column():
            # Status indicator
            status_color = {
                "waiting": "info",
                "running": "warning",
                "done": "success",
                "error": "error",
            }.get(status.value, "info")

            solara.Text(
                f"Status: {status.value.upper()}",
                style={"color": status_color, "fontWeight": "bold"},
            )

            # Progress bar
            solara.ProgressLinear(
                value=progress.value if status.value == "running" else (100 if status.value == "done" else 0),
                color=status_color,
            )

            # Parameters toggle
            solara.Button(
                "Show Parameters" if not show_params.value else "Hide Parameters",
                on_click=lambda: show_params.set(not show_params.value),
                text=True,
                small=True,
            )

            if show_params.value:
                with solara.Card("Parameters", elevation=1, margin=0):
                    with solara.Column():
                        # Number of threads
                        solara.Select(
                            label="Threads",
                            value=str(nthreads.value) if nthreads.value else "auto",
                            values=["auto", "1", "2", "4", "8"],
                            on_value=lambda v: nthreads.set(None if v == "auto" else int(v)),
                        )

                        # RT margin
                        solara.InputFloat(
                            label="RT margin (s)",
                            value=rt_margin,
                            on_value=rt_margin.set,
                        )

                        # Processing mode
                        solara.Select(
                            label="Mode",
                            value=mode.value,
                            values=["standard", "express"],
                            on_value=mode.set,
                        )

            # Action buttons
            with solara.Row():
                solara.Button(
                    "Run",
                    on_click=handle_run,
                    color="primary",
                    disabled=not can_run or status.value == "running",
                )
                solara.Button(
                    "Export",
                    on_click=handle_export,
                    color="success",
                    disabled=not can_export,
                )

            # Export filename input
            if can_export:
                solara.InputText(
                    label="Export filename",
                    value=export_filename,
                    on_value=export_filename.set,
                )
