"""Metadata panel component for viewing and editing sample metadata."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import pandas as pd
import solara

if TYPE_CHECKING:
    from ms_mint.Mint import Mint


@solara.component
def MetadataPanel(
    mint: "Mint",
    ms_files: solara.Reactive[list[str]],
    on_metadata_updated: Callable[[], None],
):
    """Component for viewing and editing sample metadata.

    Args:
        mint: The Mint instance with metadata.
        ms_files: Reactive list of loaded MS files.
        on_metadata_updated: Callback when metadata is updated.
    """
    error_message = solara.use_reactive("")
    success_message = solara.use_reactive("")
    file_path_input = solara.use_reactive("")

    if len(ms_files.value) == 0:
        solara.Info("Load MS files first to view metadata.")
        return

    def handle_load_metadata():
        """Load metadata from CSV file."""
        try:
            input_path = file_path_input.value.strip()
            path = Path(input_path)

            # Try to resolve the path
            if not path.is_absolute():
                # First try relative to current directory
                resolved = path.resolve()
                if not resolved.exists():
                    # Try relative to mint working directory
                    resolved = (mint.wdir / path).resolve()
                if not resolved.exists() and len(ms_files.value) > 0:
                    # Try relative to first MS file's directory
                    ms_dir = Path(ms_files.value[0]).parent
                    resolved = (ms_dir / path).resolve()
                path = resolved
            else:
                path = path.resolve()

            if not path.exists():
                error_message.set(f"File not found: {path}")
                success_message.set("")
                return
            mint.load_metadata(str(path))
            error_message.set("")
            success_message.set(f"Loaded metadata: {len(mint.meta)} rows")
            on_metadata_updated()
        except Exception as e:
            error_message.set(f"Error loading metadata: {e}")
            success_message.set("")

    def handle_save_metadata():
        """Save metadata to CSV file."""
        try:
            if not file_path_input.value:
                error_message.set("Enter a file path to save metadata")
                return
            mint.save_metadata(file_path_input.value.strip())
            error_message.set("")
        except Exception as e:
            error_message.set(f"Error saving metadata: {e}")

    with solara.Card("Sample Metadata", margin=0):
        with solara.Column():
            # File path input
            with solara.Row():
                solara.InputText(
                    label="Metadata CSV path",
                    value=file_path_input,
                    on_value=file_path_input.set,
                )
                solara.Button(
                    "Load",
                    on_click=handle_load_metadata,
                    color="primary",
                    disabled=not file_path_input.value,
                )
                solara.Button(
                    "Save",
                    on_click=handle_save_metadata,
                    color="secondary",
                    disabled=not file_path_input.value,
                )

            if error_message.value:
                solara.Error(error_message.value)
            if success_message.value:
                solara.Success(success_message.value)

            # Display metadata table
            meta_df = mint.meta
            if len(meta_df) > 0:
                solara.Info(f"{len(meta_df)} samples with {len(meta_df.columns)} metadata columns")
                solara.DataFrame(meta_df, items_per_page=20)
            else:
                solara.Warning("No metadata loaded. Load a CSV with 'ms_file' column matching your files.")
