"""Target loader component for uploading and managing target files."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import solara


@solara.component
def TargetLoader(
    targets: solara.Reactive[pd.DataFrame],
    on_targets_loaded: Callable[[bytes, str], None],
    on_clear: Callable[[], None],
    wdir: Path = None,
):
    """Component for loading target files (CSV/XLSX) and previewing targets.

    Args:
        targets: Reactive DataFrame of currently loaded targets.
        on_targets_loaded: Callback when targets are loaded (content, filename).
        on_clear: Callback to clear targets.
        wdir: Working directory for resolving relative paths.
    """
    error_message = solara.use_reactive("")
    file_path_input = solara.use_reactive("")

    def handle_file_upload(file_info: dict):
        """Handle uploaded file."""
        try:
            content = file_info["data"]
            filename = file_info["name"]
            if not (filename.endswith(".csv") or filename.endswith(".xlsx")):
                error_message.set("Please upload a CSV or XLSX file")
                return
            error_message.set("")
            on_targets_loaded(content, filename)
        except Exception as e:
            error_message.set(f"Error loading file: {e}")

    def handle_load_from_path():
        """Load targets from a file path on the server."""
        try:
            path = Path(file_path_input.value.strip())
            # Resolve relative paths against wdir
            if wdir is not None and not path.is_absolute():
                path = (wdir / path).resolve()

            if not path.exists():
                error_message.set(f"File not found: {path}")
                return
            if not (path.suffix.lower() in {".csv", ".xlsx"}):
                error_message.set("Please provide a CSV or XLSX file")
                return

            content = path.read_bytes()
            error_message.set("")
            on_targets_loaded(content, path.name)
        except Exception as e:
            error_message.set(f"Error: {e}")

    def handle_clear():
        """Clear loaded targets."""
        error_message.value = ""
        on_clear()

    with solara.Card("Targets", margin=0):
        with solara.Column():
            # File path input for server-side files
            solara.InputText(
                label="Target file path (CSV/XLSX)",
                value=file_path_input,
                on_value=file_path_input.set,
            )

            with solara.Row():
                solara.Button(
                    "Load from Path",
                    on_click=handle_load_from_path,
                    color="primary",
                    disabled=not file_path_input.value,
                )
                solara.Button(
                    "Clear",
                    on_click=handle_clear,
                    color="secondary",
                    disabled=len(targets.value) == 0,
                )

            # File upload for client-side files
            solara.FileDrop(
                label="Or drop a CSV/XLSX file here",
                on_file=handle_file_upload,
                lazy=False,
            )

            # Error message
            if error_message.value:
                solara.Error(error_message.value)

            # Target summary
            if len(targets.value) > 0:
                solara.Info(f"{len(targets.value)} target(s) loaded")

                # Preview table
                with solara.Details("Target preview", expand=False):
                    # Show key columns if available
                    display_cols = []
                    for col in ["peak_label", "mz_mean", "rt", "rt_min", "rt_max"]:
                        if col in targets.value.columns:
                            display_cols.append(col)

                    if display_cols:
                        preview_df = targets.value[display_cols].head(10)
                        solara.DataFrame(preview_df)
                        if len(targets.value) > 10:
                            solara.Text(
                                f"... and {len(targets.value) - 10} more rows",
                                style={"fontStyle": "italic", "fontSize": "12px"},
                            )
                    else:
                        solara.Text("No standard columns found")
