"""Target loader component for uploading and managing target files."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import solara


@solara.component
def TargetLoader(
    targets: solara.Reactive[pd.DataFrame],
    on_targets_loaded: Callable[[bytes, str], None],
    on_targets_reordered: Optional[Callable[[list[str]], None]] = None,
    on_targets_activation_changed: Optional[Callable[[list[str]], None]] = None,
    on_clear: Callable[[], None] = None,
    wdir: Path = None,
):
    """Component for loading target files (CSV/XLSX) and previewing targets.

    Args:
        targets: Reactive DataFrame of currently loaded targets.
        on_targets_loaded: Callback when targets are loaded (content, filename).
        on_targets_reordered: Callback when targets are reordered (new order of peak_labels).
        on_targets_activation_changed: Callback when target activation changes (list of inactive peak_labels).
        on_clear: Callback to clear targets.
        wdir: Working directory for resolving relative paths.
    """
    error_message = solara.use_reactive("")
    file_path_input = solara.use_reactive("")
    sort_column = solara.use_reactive("peak_label")
    sort_ascending = solara.use_reactive(True)
    inactive_targets = solara.use_reactive([])

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
        if on_clear:
            on_clear()

    def handle_sort():
        """Sort targets by selected column."""
        if len(targets.value) == 0:
            return
        try:
            df = targets.value.copy()
            # Get the column to sort by
            col = sort_column.value

            # Sort the dataframe
            if col in df.columns:
                df = df.sort_values(by=col, ascending=sort_ascending.value)
            else:
                error_message.set(f"Column {col} not found")
                return

            # Get the new order of peak labels from the peak_label column
            if "peak_label" in df.columns:
                new_order = df["peak_label"].tolist()
            else:
                error_message.set("peak_label column not found")
                return

            if on_targets_reordered:
                on_targets_reordered(new_order)
            error_message.set("")
        except Exception as e:
            error_message.set(f"Sort error: {e}")

    def handle_activation_change(new_inactive: list[str]):
        """Handle changes to target activation."""
        inactive_targets.set(list(new_inactive) if new_inactive else [])
        if on_targets_activation_changed:
            on_targets_activation_changed(list(new_inactive) if new_inactive else [])

    def activate_all():
        """Activate all targets."""
        handle_activation_change([])

    def deactivate_all():
        """Deactivate all targets."""
        if len(targets.value) > 0:
            all_labels = list(targets.value.index)
            handle_activation_change(all_labels)

    # Get sortable columns
    sortable_columns = ["peak_label"]
    if len(targets.value) > 0:
        for col in ["mz_mean", "rt", "rt_min", "rt_max", "intensity_threshold"]:
            if col in targets.value.columns:
                sortable_columns.append(col)

    # Get all target labels for activation selection
    all_target_labels = list(targets.value.index) if len(targets.value) > 0 else []
    n_active = len(all_target_labels) - len(inactive_targets.value)

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
                solara.Info(f"{n_active}/{len(targets.value)} target(s) active")

                # Sorting controls
                if on_targets_reordered:
                    with solara.Row():
                        solara.Select(
                            label="Sort by",
                            value=sort_column.value,
                            values=sortable_columns,
                            on_value=sort_column.set,
                        )
                        solara.Checkbox(
                            label="Ascending",
                            value=sort_ascending,
                        )
                        solara.Button(
                            "Apply Sort",
                            on_click=handle_sort,
                            color="primary",
                        )

                # Activation controls
                if on_targets_activation_changed and len(all_target_labels) > 0:
                    with solara.Details("Target Activation", expand=False):
                        with solara.Row():
                            solara.Button("Activate All", on_click=activate_all, color="success")
                            solara.Button("Deactivate All", on_click=deactivate_all, color="warning")
                        solara.SelectMultiple(
                            label="Inactive targets",
                            values=inactive_targets.value,
                            all_values=all_target_labels,
                            on_value=handle_activation_change,
                        )

                # Preview table
                with solara.Details("Target preview", expand=False):
                    # Show all columns
                    preview_df = targets.value.head(10).reset_index()
                    solara.DataFrame(preview_df)
                    if len(targets.value) > 10:
                        solara.Text(
                            f"... and {len(targets.value) - 10} more rows",
                            style={"fontStyle": "italic", "fontSize": "12px"},
                        )
