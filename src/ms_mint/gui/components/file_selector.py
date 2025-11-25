"""MS File selector component for browsing and uploading MS files."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import solara
import solara.lab

# Supported MS file extensions
MS_FILE_EXTENSIONS = {".mzml", ".mzxml", ".mzhdf"}


def is_ms_file(path: Path) -> bool:
    """Check if a path is a supported MS file.

    Args:
        path: Path to check.

    Returns:
        True if the file has a supported MS extension.
    """
    return path.suffix.lower() in MS_FILE_EXTENSIONS


def find_ms_files(path_str: str, wdir: Path = None) -> list[str]:
    """Find MS files from a path string (supports glob patterns).

    Args:
        path_str: Path string, can be a directory or glob pattern.
        wdir: Working directory for resolving relative paths.

    Returns:
        List of absolute file paths.
    """
    from glob import glob

    files = []
    path = Path(path_str)

    # Only resolve relative paths against wdir
    if wdir is not None and not path.is_absolute():
        path = (wdir / path).resolve()
        path_str = str(path)

    # Check if it's a glob pattern (contains * or ?)
    if "*" in path_str or "?" in path_str:
        # Use glob pattern directly
        matched = glob(path_str)
        files = [f for f in matched if is_ms_file(Path(f))]
    elif path.is_dir():
        # It's a directory - search for MS files
        for ext in MS_FILE_EXTENSIONS:
            files.extend(str(f) for f in path.glob(f"*{ext}"))
            files.extend(str(f) for f in path.glob(f"*{ext.upper()}"))
    elif path.is_file() and is_ms_file(path):
        # It's a single MS file
        files = [str(path)]

    return sorted(files)


@solara.component
def MSFileSelector(
    ms_files: solara.Reactive[list[str]],
    on_files_loaded: Callable[[list[str]], None],
    wdir: Optional[Path] = None,
):
    """Component for selecting MS files via directory browser or file upload.

    Provides two modes of file selection:
    1. Browse server directories and load all MS files from a folder
    2. Upload files from the client (for smaller files)

    Args:
        ms_files: Reactive list of currently loaded MS file paths.
        on_files_loaded: Callback when files are loaded.
        wdir: Working directory for resolving relative paths.
    """
    # Local state
    default_dir = wdir if wdir is not None else Path.cwd()
    path_input = solara.use_reactive(str(default_dir))
    error_message = solara.use_reactive("")

    def handle_load_from_path():
        """Load MS files from the path (directory, file, or glob pattern)."""
        try:
            files = find_ms_files(path_input.value, wdir=wdir)
            if not files:
                error_message.set(f"No MS files found: {path_input.value}")
                return

            error_message.set("")
            on_files_loaded(files)
        except Exception as e:
            error_message.set(f"Error: {e}")

    def handle_clear():
        """Clear loaded files."""
        on_files_loaded([])
        error_message.set("")

    with solara.Card("MS Files", margin=0):
        with solara.Column():
            # Path input (supports directory, file, or glob pattern)
            solara.InputText(
                label="Path (directory, file, or glob pattern like *.mzML)",
                value=path_input,
                on_value=path_input.set,
            )

            # Action buttons
            with solara.Row():
                solara.Button(
                    "Load MS Files",
                    on_click=handle_load_from_path,
                    color="primary",
                )
                solara.Button(
                    "Clear",
                    on_click=handle_clear,
                    color="secondary",
                    disabled=len(ms_files.value) == 0,
                )

            # Error message
            if error_message.value:
                solara.Error(error_message.value)

            # File count and list
            if ms_files.value:
                solara.Info(f"{len(ms_files.value)} MS file(s) loaded")

                # Show first few files
                with solara.Details("Loaded files", expand=False):
                    for f in ms_files.value[:10]:
                        solara.Text(Path(f).name, style={"fontSize": "12px"})
                    if len(ms_files.value) > 10:
                        solara.Text(
                            f"... and {len(ms_files.value) - 10} more",
                            style={"fontStyle": "italic", "fontSize": "12px"},
                        )
