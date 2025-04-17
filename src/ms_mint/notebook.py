#src/ms_mint/notebook.py

"""Experimental module to run Mint interactively inside the Jupyter notebook.

Example usage:
    ```python
    from ms_mint.notebook import Mint

    mint = Mint()

    mint.display()
    ```
"""

import os
import io
import ipywidgets as W
from glob import glob
from pathlib import Path as P
from typing import Optional, Union, List, Dict, Any, Tuple, Callable, ByteString, cast

from ipyfilechooser import FileChooser
from IPython.display import display
from IPython.core.display import HTML

from .Mint import Mint as _Mint_
from tqdm.notebook import tqdm


HOME = str(P.home())


class Mint(_Mint_):
    """Interactive MINT for Jupyter Notebook environment (experimental).

    This class extends the base Mint class with interactive widgets and controls
    for use in Jupyter notebooks, allowing for a graphical user interface to
    manage MS files, target lists, and process data.

    Attributes:
        progress_callback: Function to update progress bar.
        ms_storage_path: File chooser widget for MS file directory.
        target_files_button: Upload widget for target files.
        load_ms_button: Button to load MS files from selected directory.
        message_box: Text area for displaying messages.
        run_button: Button to start processing.
        download_button: Button to export results.
        progress_bar: Progress indicator for processing.
        layout: Main container for all widgets.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the interactive Mint instance.

        Args:
            *args: Positional arguments passed to the parent Mint class.
            **kwargs: Keyword arguments passed to the parent Mint class.
        """
        self.progress_callback = self._set_progress_

        super().__init__(progress_callback=self.progress_callback, *args, **kwargs)

        # Initialize file chooser for MS files directory
        fc = FileChooser()
        fc.show_only_dirs = True
        fc.default_path = os.getcwd()
        self.ms_storage_path = fc

        # Target file upload widget
        self.target_files_button = W.FileUpload(
            description="Peaklists", accept="csv,xlsx", multiple=False
        )
        self.target_files_button.observe(self._load_target_from_bytes_, names="value")

        # Button to load MS files
        self.load_ms_button = W.Button(description="Load MS-files")
        self.load_ms_button.on_click(self._search_files_)

        # Message display area
        self.message_box = W.Textarea(
            value="",
            placeholder="Please, select ms-files define a target list.",
            description="",
            disabled=True,
            layout={"width": "90%", "height": "500px", "font_family": "monospace"},
        )

        # Processing buttons
        self.run_button = W.Button(description="Run")
        self.run_button.on_click(self._run_)
        self.run_button.style.button_color = "lightgray"

        self.download_button = W.Button(description="Export")
        self.download_button.on_click(self._export_action_)
        self.download_button.style.button_color = "lightgray"

        # Progress indicator
        self.progress_bar = W.IntProgress(
            min=0,
            max=100,
            layout=W.Layout(width="90%"),
            description="Progress:",
            bar_style="info",
        )

        self.output = W.Output()

        # Create tabs for file selection
        tabs = W.Tab()
        tabs.children = [
            W.HBox([self.ms_storage_path, self.load_ms_button]),
            W.HBox(
                [
                    self.target_files_button,
                ]
            ),
        ]

        tabs.set_title(0, "MS-Files")
        tabs.set_title(1, "Peaklists")

        # Main layout
        self.layout = W.VBox(
            [
                tabs,
                self.message_box,
                W.HBox([self.run_button, self.download_button]),
                self.progress_bar,
            ]
        )

        self.tqdm = tqdm

    def _load_target_from_bytes_(self, value: Dict[str, Any]) -> None:
        """Load target list from uploaded file bytes.

        Args:
            value: Dictionary containing upload widget's value information.
        """
        for data in value["new"].values():
            self.load(io.BytesIO(data["content"]))
        self._message_(f"{len(self.targets)} targets loaded.")

    @property
    def messages(self) -> List[str]:
        """Get the list of messages displayed in the message box.

        Returns:
            List of messages.
        """
        return self._messages

    def _message_(self, text: str) -> None:
        """Add a message to the message box.

        Args:
            text: Message text to add.
        """
        self.message_box.value = f"{text}\n" + self.message_box.value

    def _clear_messages_(self) -> None:
        """Clear all messages from the message box."""
        self.message_box.value = ""

    def _search_files_(self, b: Optional[W.Button] = None) -> None:
        """Search for MS files in the selected directory.

        Args:
            b: Button that triggered the action (not used).
        """
        self.ms_files = (
            glob(os.path.join(self.ms_storage_path.selected_path, "*mzXML"))
            + glob(os.path.join(self.ms_storage_path.selected_path, "*mzML"))
            + glob(os.path.join(self.ms_storage_path.selected_path, "*mzHDF"))
            + glob(os.path.join(self.ms_storage_path.selected_path, "*mzxml"))
            + glob(os.path.join(self.ms_storage_path.selected_path, "*mzml"))
            + glob(os.path.join(self.ms_storage_path.selected_path, "*mzhdf"))
        )
        self.message(
            f"{self.n_files} MS-files loaded."
        )  # This should be self._message_ instead of self.message

    def display(self) -> W.VBox:
        """Display control elements in Jupyter notebook.

        Returns:
            The main widget layout container.
        """
        display(HTML("<style>textarea, input { font-family: monospace; }</style>"))
        return self.layout

    def _run_(self, b: Optional[W.Button] = None, **kwargs: Any) -> None:
        """Run data processing with the current settings.

        Args:
            b: Button that triggered the action (not used).
            **kwargs: Additional keyword arguments passed to the run method.
        """
        self._message_("Start processing...")
        self.progress = 0
        self.run(**kwargs)
        self._message_("...finished processing.")
        if self.results is not None:
            self.download_button.style.button_color = "lightgreen"

    def _set_progress_(self, value: int) -> None:
        """Update the progress bar value.

        Args:
            value: Progress value (0-100).
        """
        self.progress_bar.value = value

    def _export_action_(self, b: Optional[W.Button] = None, filename: Optional[str] = None) -> None:
        """Export results to an Excel file.

        Args:
            b: Button that triggered the action (not used).
            filename: Output filename. If None, uses a default name.
        """
        if filename is None:
            filename = "MINT__results.xlsx"
            filename = os.path.join(os.getcwd(), filename)
        self.export(filename)
        self._message_(f"\nExported results to: {filename}")
